from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy and pandas are required in the project environment."
    ) from exc


ALL_SUBJECTS = (101, 102, 103, 104, 105, 106, 107, 108)
OUTER_SUBJECTS = (101, 102, 105, 106, 108)
NUM_CLASSES = 12
CLIENTS_PER_SUBJECT = 4
ALPHAS = (1.0, 0.1)

BASE_PARTITION_SEED = 20260706
MIN_CLIENT_WINDOWS = 128
MAX_RESAMPLE_ATTEMPTS = 2000

WINDOW_LENGTH_SECONDS = 2.56
OVERLAP_TOLERANCE_SECONDS = 1e-6

ACTIVITY_NAMES = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "Nordic walking",
    "ascending stairs",
    "descending stairs",
    "vacuum cleaning",
    "ironing",
    "rope jumping",
]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def alpha_label(alpha: float) -> str:
    if math.isclose(alpha, 1.0):
        return "alpha1p0"
    if math.isclose(alpha, 0.1):
        return "alpha0p1"
    return str(alpha).replace(".", "p")


def derive_subject_seed(subject_id: int, alpha: float) -> int:
    alpha_code = 1000 if math.isclose(alpha, 1.0) else 100
    return BASE_PARTITION_SEED + subject_id * 100 + alpha_code


def parse_subject_list(text: str) -> tuple[int, ...]:
    subjects: list[int] = []

    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue

        if token.lower().startswith("subject"):
            token = token[7:]

        subjects.append(int(token))

    return tuple(subjects)


def largest_remainder_counts(
    total: int,
    proportions: np.ndarray,
) -> np.ndarray:
    raw = proportions * total
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())

    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1

    if int(counts.sum()) != total:
        raise RuntimeError("Largest-remainder allocation failed.")

    return counts


def normalized_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0

    probabilities = counts[counts > 0].astype(float) / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(NUM_CLASSES)


def js_divergence(
    client_counts: np.ndarray,
    reference_counts: np.ndarray,
) -> float:
    p = client_counts.astype(float)
    q = reference_counts.astype(float)

    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass(frozen=True)
class DatasetIndex:
    row_index: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    start_timestamp_s: np.ndarray


def load_all_metadata(processed_root: Path) -> DatasetIndex:
    split_dir = processed_root / "splits"

    row_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []

    global_offset = 0

    for split in ("train", "validation", "test"):
        y = np.asarray(
            np.load(split_dir / f"{split}_y.npy", mmap_mode="r"),
            dtype=np.int64,
        ).copy()
        subject = np.asarray(
            np.load(
                split_dir / f"{split}_subject_id.npy",
                mmap_mode="r",
            ),
            dtype=np.int64,
        ).copy()
        timestamp = np.asarray(
            np.load(
                split_dir / f"{split}_start_timestamp_s.npy",
                mmap_mode="r",
            ),
            dtype=np.float64,
        ).copy()

        n = int(y.shape[0])

        row_parts.append(
            np.arange(global_offset, global_offset + n, dtype=np.int64)
        )
        y_parts.append(y)
        subject_parts.append(subject)
        time_parts.append(timestamp)

        global_offset += n

    rows = np.concatenate(row_parts)
    y = np.concatenate(y_parts)
    subject = np.concatenate(subject_parts)
    timestamp = np.concatenate(time_parts)

    if rows.shape != (14972,):
        raise RuntimeError(f"Unexpected row shape: {rows.shape}")
    if y.shape != rows.shape:
        raise RuntimeError("Unexpected label shape.")
    if subject.shape != rows.shape:
        raise RuntimeError("Unexpected subject shape.")
    if timestamp.shape != rows.shape:
        raise RuntimeError("Unexpected timestamp shape.")

    observed_subjects = tuple(sorted(int(v) for v in np.unique(subject)))
    if observed_subjects != ALL_SUBJECTS:
        raise RuntimeError(f"Unexpected subjects: {observed_subjects}")

    if not np.isfinite(timestamp).all():
        raise RuntimeError("Non-finite timestamps found.")

    return DatasetIndex(
        row_index=rows,
        y=y,
        subject=subject,
        start_timestamp_s=timestamp,
    )


def allocate_subject_once(
    *,
    subject_rows: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return local-client assignment for each subject row and guard-drop mask.

    Assignment is class-wise and chronological. Dirichlet proportions control
    class allocation. At boundaries between two non-empty client slices, the
    first row of the later slice is dropped only when its 2.56 s window would
    overlap the final retained window of the previous slice.
    """
    n = int(subject_rows.shape[0])
    assignment = np.full(n, -1, dtype=np.int64)
    dropped = np.zeros(n, dtype=bool)

    for class_index in sorted(int(v) for v in np.unique(labels)):
        positions = np.where(labels == class_index)[0]
        positions = positions[
            np.argsort(timestamps[positions], kind="stable")
        ]

        proportions = rng.dirichlet(
            np.full(CLIENTS_PER_SUBJECT, alpha, dtype=float)
        )
        counts = largest_remainder_counts(
            int(positions.shape[0]),
            proportions,
        )

        client_order = rng.permutation(CLIENTS_PER_SUBJECT)

        cursor = 0
        previous_last_position: int | None = None

        for client_id in client_order:
            count = int(counts[client_id])
            if count <= 0:
                continue

            current_positions = positions[cursor : cursor + count]
            cursor += count

            if current_positions.size == 0:
                continue

            if previous_last_position is not None:
                first_position = int(current_positions[0])
                time_gap = (
                    float(timestamps[first_position])
                    - float(timestamps[previous_last_position])
                )

                if time_gap < (
                    WINDOW_LENGTH_SECONDS - OVERLAP_TOLERANCE_SECONDS
                ):
                    dropped[first_position] = True
                    current_positions = current_positions[1:]

            if current_positions.size == 0:
                continue

            assignment[current_positions] = int(client_id)
            previous_last_position = int(current_positions[-1])

        if cursor != int(positions.shape[0]):
            raise RuntimeError("Class allocation cursor mismatch.")

    assignment[dropped] = -1
    return assignment, dropped


def check_no_cross_client_overlap(
    *,
    assignments: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    retained_positions = np.where(assignments >= 0)[0]
    order = retained_positions[
        np.argsort(timestamps[retained_positions], kind="stable")
    ]

    for left, right in zip(order[:-1], order[1:]):
        if assignments[left] == assignments[right]:
            continue

        time_gap = float(timestamps[right]) - float(timestamps[left])

        if time_gap < (
            WINDOW_LENGTH_SECONDS - OVERLAP_TOLERANCE_SECONDS
        ):
            raise RuntimeError(
                "Overlapping windows from the same subject were assigned "
                "to different clients."
            )


def build_subject_partition(
    *,
    subject_id: int,
    alpha: float,
    dataset: DatasetIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_mask = dataset.subject == subject_id

    subject_rows = dataset.row_index[subject_mask]
    labels = dataset.y[subject_mask]
    timestamps = dataset.start_timestamp_s[subject_mask]

    seed = derive_subject_seed(subject_id, alpha)

    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        rng = np.random.default_rng(seed + attempt)

        assignment, dropped = allocate_subject_once(
            subject_rows=subject_rows,
            labels=labels,
            timestamps=timestamps,
            alpha=alpha,
            rng=rng,
        )

        sizes = np.array(
            [
                int(np.sum(assignment == client_id))
                for client_id in range(CLIENTS_PER_SUBJECT)
            ],
            dtype=np.int64,
        )

        if int(sizes.min()) < MIN_CLIENT_WINDOWS:
            continue

        check_no_cross_client_overlap(
            assignments=assignment,
            timestamps=timestamps,
        )

        rows: list[dict[str, object]] = []
        dropped_rows: list[dict[str, object]] = []

        for local_position in range(subject_rows.shape[0]):
            if dropped[local_position]:
                dropped_rows.append(
                    {
                        "alpha": alpha,
                        "alpha_label": alpha_label(alpha),
                        "subject_id": subject_id,
                        "row_index": int(subject_rows[local_position]),
                        "activity_index": int(labels[local_position]),
                        "activity_name": ACTIVITY_NAMES[
                            int(labels[local_position])
                        ],
                        "start_timestamp_s": float(
                            timestamps[local_position]
                        ),
                        "reason": "cross_client_overlap_guard",
                    }
                )
                continue

            client_id = int(assignment[local_position])
            if client_id < 0:
                raise RuntimeError(
                    "Retained subject row has no client assignment."
                )

            rows.append(
                {
                    "alpha": alpha,
                    "alpha_label": alpha_label(alpha),
                    "subject_id": subject_id,
                    "local_client_id": client_id,
                    "global_client_id": (
                        f"subject{subject_id}_client{client_id}"
                    ),
                    "row_index": int(subject_rows[local_position]),
                    "activity_index": int(labels[local_position]),
                    "activity_name": ACTIVITY_NAMES[
                        int(labels[local_position])
                    ],
                    "start_timestamp_s": float(
                        timestamps[local_position]
                    ),
                }
            )

        assignment_df = pd.DataFrame(rows)
        dropped_df = pd.DataFrame(dropped_rows)

        if assignment_df["row_index"].duplicated().any():
            raise RuntimeError("Duplicate row assignment detected.")

        return assignment_df, dropped_df

    raise RuntimeError(
        f"Could not create a valid partition for subject{subject_id}, "
        f"alpha={alpha} after {MAX_RESAMPLE_ATTEMPTS} attempts."
    )


def client_metrics(
    assignment_df: pd.DataFrame,
    reference_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        subject_id,
        local_client_id,
        global_client_id,
    ), group in assignment_df.groupby(
        ["subject_id", "local_client_id", "global_client_id"],
        sort=True,
    ):
        counts = np.bincount(
            group["activity_index"].to_numpy(dtype=np.int64),
            minlength=NUM_CLASSES,
        )

        rows.append(
            {
                "subject_id": int(subject_id),
                "local_client_id": int(local_client_id),
                "global_client_id": str(global_client_id),
                "windows": int(counts.sum()),
                "classes_present": int(np.sum(counts > 0)),
                "normalized_label_entropy": normalized_entropy(counts),
                "dominant_class_fraction": float(
                    counts.max() / counts.sum()
                ),
                "js_divergence_from_fold": js_divergence(
                    counts,
                    reference_counts,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit subject-pure PAMAP2 federated client partitions "
            "for alpha=1.0 and alpha=0.1 before any FL training."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser().resolve()

    try:
        project_root = processed_root.parents[3]
    except IndexError:
        project_root = Path.cwd()

    protocol_root = (
        args.protocol_root.expanduser().resolve()
        if args.protocol_root is not None
        else project_root
        / "outputs"
        / "protocols"
        / "pamap2_evaluation_v2"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_partitions_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    outer_manifest_path = protocol_root / "outer_fold_manifest.csv"
    evaluation_protocol_path = protocol_root / "EVALUATION_PROTOCOL_V2.json"

    for path in (outer_manifest_path, evaluation_protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    outer_manifest = pd.read_csv(outer_manifest_path)

    print("=== Build and audit PAMAP2 federated partitions ===")
    print(f"Processed dataset: {processed_root}")
    print(f"Outer protocol:    {protocol_root}")
    print(f"Output:            {output_root}")
    print()
    print(f"Clients per subject: {CLIENTS_PER_SUBJECT}")
    print(f"Outer-fold clients:  {7 * CLIENTS_PER_SUBJECT}")
    print(f"Alphas:              {list(ALPHAS)}")
    print(f"Minimum client size: {MIN_CLIENT_WINDOWS} windows")
    print()

    dataset = load_all_metadata(processed_root)

    all_assignment_frames: list[pd.DataFrame] = []
    all_dropped_frames: list[pd.DataFrame] = []

    for alpha in ALPHAS:
        print(f"Building master subject partitions for alpha={alpha}...")

        alpha_assignments: list[pd.DataFrame] = []
        alpha_dropped: list[pd.DataFrame] = []

        for subject_id in ALL_SUBJECTS:
            assignment_df, dropped_df = build_subject_partition(
                subject_id=subject_id,
                alpha=alpha,
                dataset=dataset,
            )

            alpha_assignments.append(assignment_df)

            if not dropped_df.empty:
                alpha_dropped.append(dropped_df)

            sizes = (
                assignment_df.groupby("global_client_id")
                .size()
                .sort_values()
                .to_numpy()
            )

            print(
                f"  subject{subject_id}: "
                f"clients={len(sizes)}; "
                f"min={int(sizes.min())}; "
                f"max={int(sizes.max())}; "
                f"retained={len(assignment_df)}"
            )

        alpha_assignment_df = pd.concat(
            alpha_assignments,
            ignore_index=True,
        )

        alpha_assignment_df.to_csv(
            output_root
            / f"master_assignments_{alpha_label(alpha)}.csv",
            index=False,
        )

        all_assignment_frames.append(alpha_assignment_df)

        if alpha_dropped:
            all_dropped_frames.append(
                pd.concat(alpha_dropped, ignore_index=True)
            )

        print()

    all_assignments = pd.concat(
        all_assignment_frames,
        ignore_index=True,
    )

    if all_dropped_frames:
        dropped_all = pd.concat(
            all_dropped_frames,
            ignore_index=True,
        )
    else:
        dropped_all = pd.DataFrame(
            columns=[
                "alpha",
                "alpha_label",
                "subject_id",
                "row_index",
                "activity_index",
                "activity_name",
                "start_timestamp_s",
                "reason",
            ]
        )

    dropped_all.to_csv(
        output_root / "dropped_guard_windows.csv",
        index=False,
    )

    outer_client_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    class_count_rows: list[dict[str, object]] = []

    for outer_row in outer_manifest.itertuples(index=False):
        outer_fold = int(outer_row.outer_fold)
        outer_test_subject = int(
            str(outer_row.outer_test_subject).replace("subject", "")
        )
        training_subjects = parse_subject_list(
            outer_row.outer_training_subjects
        )

        if len(training_subjects) != 7:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 7 training subjects."
            )
        if outer_test_subject in training_subjects:
            raise RuntimeError(
                f"Outer fold {outer_fold}: test subject leaked into training."
            )

        for alpha in ALPHAS:
            fold_assignments = all_assignments[
                (all_assignments["alpha"] == alpha)
                & (
                    all_assignments["subject_id"].isin(
                        training_subjects
                    )
                )
            ].copy()

            client_ids = sorted(
                fold_assignments["global_client_id"].unique().tolist()
            )

            if len(client_ids) != 28:
                raise RuntimeError(
                    f"Outer fold {outer_fold}, alpha={alpha}: "
                    f"expected 28 clients, found {len(client_ids)}."
                )

            reference_counts = np.bincount(
                fold_assignments["activity_index"].to_numpy(
                    dtype=np.int64
                ),
                minlength=NUM_CLASSES,
            )

            metrics_df = client_metrics(
                fold_assignments,
                reference_counts,
            )
            metrics_df.insert(0, "alpha", alpha)
            metrics_df.insert(0, "outer_test_subject", outer_test_subject)
            metrics_df.insert(0, "outer_fold", outer_fold)

            outer_client_rows.append(metrics_df)

            for (
                client_id,
                client_group,
            ) in fold_assignments.groupby(
                "global_client_id",
                sort=True,
            ):
                counts = np.bincount(
                    client_group["activity_index"].to_numpy(
                        dtype=np.int64
                    ),
                    minlength=NUM_CLASSES,
                )

                for class_index in range(NUM_CLASSES):
                    class_count_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "outer_test_subject": outer_test_subject,
                            "alpha": alpha,
                            "global_client_id": client_id,
                            "class_index": class_index,
                            "activity_name": ACTIVITY_NAMES[class_index],
                            "windows": int(counts[class_index]),
                        }
                    )

            summary_rows.append(
                {
                    "outer_fold": outer_fold,
                    "outer_test_subject": outer_test_subject,
                    "alpha": alpha,
                    "clients": int(len(metrics_df)),
                    "total_retained_windows": int(
                        metrics_df["windows"].sum()
                    ),
                    "min_client_windows": int(
                        metrics_df["windows"].min()
                    ),
                    "median_client_windows": float(
                        metrics_df["windows"].median()
                    ),
                    "max_client_windows": int(
                        metrics_df["windows"].max()
                    ),
                    "mean_classes_present": float(
                        metrics_df["classes_present"].mean()
                    ),
                    "mean_normalized_label_entropy": float(
                        metrics_df[
                            "normalized_label_entropy"
                        ].mean()
                    ),
                    "mean_dominant_class_fraction": float(
                        metrics_df[
                            "dominant_class_fraction"
                        ].mean()
                    ),
                    "mean_js_divergence_from_fold": float(
                        metrics_df[
                            "js_divergence_from_fold"
                        ].mean()
                    ),
                }
            )

    outer_client_df = pd.concat(
        outer_client_rows,
        ignore_index=True,
    )
    outer_client_df.to_csv(
        output_root / "outer_fold_client_manifest.csv",
        index=False,
    )

    class_counts_df = pd.DataFrame(class_count_rows)
    class_counts_df.to_csv(
        output_root / "outer_fold_client_class_counts.csv",
        index=False,
    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["outer_fold", "alpha"],
        ascending=[True, False],
    )
    summary_df.to_csv(
        output_root / "outer_fold_partition_summary.csv",
        index=False,
    )

    alpha_summary = (
        summary_df.groupby("alpha", as_index=False)
        .agg(
            folds=("outer_fold", "count"),
            mean_clients=("clients", "mean"),
            mean_min_client_windows=("min_client_windows", "mean"),
            mean_classes_present=("mean_classes_present", "mean"),
            mean_normalized_label_entropy=(
                "mean_normalized_label_entropy",
                "mean",
            ),
            mean_dominant_class_fraction=(
                "mean_dominant_class_fraction",
                "mean",
            ),
            mean_js_divergence_from_fold=(
                "mean_js_divergence_from_fold",
                "mean",
            ),
        )
        .sort_values("alpha", ascending=False)
    )

    alpha_summary.to_csv(
        output_root / "alpha_partition_summary.csv",
        index=False,
    )

    alpha1 = alpha_summary[
        np.isclose(alpha_summary["alpha"], 1.0)
    ].iloc[0]
    alpha01 = alpha_summary[
        np.isclose(alpha_summary["alpha"], 0.1)
    ].iloc[0]

    if not (
        float(alpha01["mean_normalized_label_entropy"])
        < float(alpha1["mean_normalized_label_entropy"])
    ):
        raise RuntimeError(
            "Audit failure: alpha=0.1 did not reduce mean label entropy."
        )

    if not (
        float(alpha01["mean_js_divergence_from_fold"])
        > float(alpha1["mean_js_divergence_from_fold"])
    ):
        raise RuntimeError(
            "Audit failure: alpha=0.1 did not increase mean JSD."
        )

    if int(outer_client_df["windows"].min()) < MIN_CLIENT_WINDOWS:
        raise RuntimeError("Audit failure: a client is below minimum size.")

    expected_outer_rows = 5 * 2 * 28
    if len(outer_client_df) != expected_outer_rows:
        raise RuntimeError(
            f"Audit failure: expected {expected_outer_rows} outer-client rows, "
            f"found {len(outer_client_df)}."
        )

    protocol = {
        "protocol_name": "PAMAP2 Federated Client Partitions v1",
        "status": "FROZEN_AFTER_PARTITION_AUDIT_BEFORE_FL_TRAINING",
        "source_evaluation_protocol": str(evaluation_protocol_path),
        "source_evaluation_protocol_sha256": sha256_file(
            evaluation_protocol_path
        ),
        "processed_root": str(processed_root),
        "subjects": list(ALL_SUBJECTS),
        "outer_test_subjects": list(OUTER_SUBJECTS),
        "clients_per_subject": CLIENTS_PER_SUBJECT,
        "clients_per_outer_fold": 28,
        "subject_pure_clients": True,
        "heterogeneity": {
            "alphas": list(ALPHAS),
            "partition_unit": (
                "within-subject, within-activity chronological window slices"
            ),
            "dirichlet_allocation": True,
            "overlap_guard": (
                "drop the first window of a later client slice only when it "
                "would overlap the previous client's final retained window"
            ),
            "minimum_client_windows": MIN_CLIENT_WINDOWS,
            "base_partition_seed": BASE_PARTITION_SEED,
            "master_subject_partitions_reused_across_outer_folds": True,
        },
        "leakage_controls": {
            "outer_test_subject_excluded_from_outer_training_clients": True,
            "cross_client_overlapping_windows_within_subject_forbidden": True,
            "client_contains_only_one_physical_subject": True,
        },
        "audit": {
            "alpha_0p1_lower_entropy_than_alpha_1p0": True,
            "alpha_0p1_higher_jsd_than_alpha_1p0": True,
            "all_clients_above_minimum_size": True,
        },
    }

    protocol_path = output_root / "FL_PARTITION_PROTOCOL.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    dropped_fraction = (
        len(dropped_all)
        / (len(dataset.row_index) * len(ALPHAS))
    )

    report_lines = [
        "PAMAP2 FEDERATED CLIENT PARTITION AUDIT",
        "=" * 78,
        f"Processed dataset: {processed_root}",
        f"Outer protocol: {protocol_root}",
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "Partitions frozen before any FL training.",
        "",
        "CLIENT DESIGN",
        "-" * 78,
        "4 pseudo-clients per physical training subject.",
        "7 training subjects per outer fold.",
        "28 clients per outer fold.",
        "Every client contains data from exactly one physical subject.",
        "",
        "HETEROGENEITY REGIMES",
        "-" * 78,
        "alpha=1.0: moderate label skew",
        "alpha=0.1: strong label skew",
        "",
        "ALPHA-LEVEL AUDIT",
        "-" * 78,
    ]

    for row in alpha_summary.itertuples(index=False):
        report_lines.append(
            f"alpha={row.alpha}: "
            f"mean classes/client={row.mean_classes_present:.2f}; "
            f"mean normalized entropy={row.mean_normalized_label_entropy:.3f}; "
            f"mean dominant-class fraction={row.mean_dominant_class_fraction:.3f}; "
            f"mean JSD={row.mean_js_divergence_from_fold:.3f}"
        )

    report_lines.extend(
        [
            "",
            "SIZE AND OVERLAP CONTROLS",
            "-" * 78,
            f"Minimum allowed client windows: {MIN_CLIENT_WINDOWS}",
            f"Observed minimum client windows: {int(outer_client_df['windows'].min())}",
            f"Guard windows dropped across both alpha masters: {len(dropped_all)}",
            f"Dropped fraction of subject-alpha rows: {dropped_fraction:.4%}",
            "No overlapping windows from one subject are assigned to different clients.",
            "",
            "LEAKAGE CONTROLS",
            "-" * 78,
            "Outer test subject excluded from all 28 training clients in its fold.",
            "Master client assignments are subject-local and reused across outer folds.",
            "No client mixes physical subjects.",
            "",
            "FILES",
            "-" * 78,
            "FL_PARTITION_PROTOCOL.json",
            "PARTITION_AUDIT_REPORT.txt",
            "master_assignments_alpha1p0.csv",
            "master_assignments_alpha0p1.csv",
            "dropped_guard_windows.csv",
            "outer_fold_client_manifest.csv",
            "outer_fold_client_class_counts.csv",
            "outer_fold_partition_summary.csv",
            "alpha_partition_summary.csv",
        ]
    )

    report_path = output_root / "PARTITION_AUDIT_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("[OK] All outer folds contain exactly 28 subject-pure clients.")
    print(
        f"[OK] Minimum observed client size: "
        f"{int(outer_client_df['windows'].min())}"
    )
    print(
        "[OK] alpha=0.1 has lower entropy and higher JSD than alpha=1.0."
    )
    print(
        f"[OK] Cross-client overlap guard windows dropped: "
        f"{len(dropped_all)}"
    )
    print()
    print("=== Federated partition audit completed successfully ===")
    print(f"Protocol: {protocol_path}")
    print(f"Report:   {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPartition audit interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
