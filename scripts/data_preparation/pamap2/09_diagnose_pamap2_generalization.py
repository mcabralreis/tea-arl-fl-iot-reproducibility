from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "ERROR: required packages are missing from the project environment."
    ) from exc


NUM_CLASSES = 12
BATCH_SIZE = 128

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

MODEL_VARIANTS = {
    "tiny": {
        "channels": (24, 48, 72),
        "dropout": 0.15,
    },
    "small": {
        "channels": (32, 64, 96),
        "dropout": 0.20,
    },
    "medium": {
        "channels": (48, 96, 128),
        "dropout": 0.25,
    },
}


def select_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class LightweightCNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            ConvBNReLU(input_channels, c1, kernel_size=7, stride=2),
            ConvBNReLU(c1, c1, kernel_size=5, stride=1),
            ConvBNReLU(c1, c2, kernel_size=5, stride=2),
            ConvBNReLU(c2, c2, kernel_size=3, stride=1),
            ConvBNReLU(c2, c3, kernel_size=3, stride=2),
            ConvBNReLU(c3, c3, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(variant: str) -> LightweightCNN1D:
    config = MODEL_VARIANTS[variant]
    return LightweightCNN1D(
        input_channels=27,
        num_classes=12,
        channels=tuple(config["channels"]),
        dropout=float(config["dropout"]),
    )


def load_core27_split(
    processed_root: Path,
    split: str,
    core27_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_dir = processed_root / "splits"
    full = np.load(
        split_dir / f"{split}_X_full36.npy",
        mmap_mode="r",
    )
    x_np = np.ascontiguousarray(
        full[:, :, core27_indices].transpose(0, 2, 1),
        dtype=np.float32,
    )
    y_np = np.asarray(
        np.load(split_dir / f"{split}_y.npy", mmap_mode="r"),
        dtype=np.int64,
    ).copy()

    return torch.from_numpy(x_np), torch.from_numpy(y_np)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return (
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        y_true,
        y_pred,
    )


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain model_state_dict: {path}"
        )
    return checkpoint["model_state_dict"]


def top_confusions(
    matrix: np.ndarray,
    top_k: int = 15,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for true_index in range(NUM_CLASSES):
        true_total = int(matrix[true_index].sum())
        if true_total == 0:
            continue

        for pred_index in range(NUM_CLASSES):
            if pred_index == true_index:
                continue

            count = int(matrix[true_index, pred_index])
            if count == 0:
                continue

            rows.append(
                {
                    "true_class": true_index,
                    "true_activity": ACTIVITY_NAMES[true_index],
                    "predicted_class": pred_index,
                    "predicted_activity": ACTIVITY_NAMES[pred_index],
                    "count": count,
                    "fraction_of_true_class": count / true_total,
                }
            )

    rows.sort(
        key=lambda item: (
            item["fraction_of_true_class"],
            item["count"],
        ),
        reverse=True,
    )
    return rows[:top_k]


def class_shift_profile(
    x_development: torch.Tensor,
    y_development: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> pd.DataFrame:
    """
    Quantify subject-level class shift using 54-D window summary features:
    per-channel temporal mean + temporal standard deviation.

    Distances are normalized by the pooled development feature scale.
    """
    dev_np = x_development.numpy()
    test_np = x_test.numpy()
    y_dev = y_development.numpy()
    y_tst = y_test.numpy()

    dev_features = np.concatenate(
        (
            dev_np.mean(axis=2),
            dev_np.std(axis=2),
        ),
        axis=1,
    )
    test_features = np.concatenate(
        (
            test_np.mean(axis=2),
            test_np.std(axis=2),
        ),
        axis=1,
    )

    global_scale = dev_features.std(axis=0, ddof=0)
    global_scale = np.where(global_scale < 1e-6, 1.0, global_scale)

    rows: list[dict[str, object]] = []

    for class_index in range(NUM_CLASSES):
        dev_class = dev_features[y_dev == class_index]
        test_class = test_features[y_tst == class_index]

        dev_centroid = dev_class.mean(axis=0)
        test_centroid = test_class.mean(axis=0)

        standardized_difference = (
            test_centroid - dev_centroid
        ) / global_scale

        centroid_rms_distance = float(
            np.sqrt(np.mean(np.square(standardized_difference)))
        )
        centroid_max_abs_shift = float(
            np.max(np.abs(standardized_difference))
        )

        rows.append(
            {
                "class_index": class_index,
                "activity_name": ACTIVITY_NAMES[class_index],
                "development_windows": int(dev_class.shape[0]),
                "test_windows": int(test_class.shape[0]),
                "centroid_rms_shift": centroid_rms_distance,
                "centroid_max_abs_shift": centroid_max_abs_shift,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "centroid_rms_shift",
        ascending=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the PAMAP2 subject108 generalization collapse without "
            "training new models."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--screening-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--final-root",
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

    screening_root = (
        args.screening_root.expanduser().resolve()
        if args.screening_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / "screening_v1"
    )

    final_root = (
        args.final_root.expanduser().resolve()
        if args.final_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / "final_test_v1"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "diagnostics"
        / "pamap2_generalization_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device()

    print("=== PAMAP2 generalization diagnosis ===")
    print(f"Processed dataset: {processed_root}")
    print(f"Screening source:  {screening_root}")
    print(f"Final source:      {final_root}")
    print(f"Output:            {output_root}")
    print(f"Device:            {device}")
    print()

    core27_indices = np.load(
        processed_root / "statistics" / "core27_indices.npy"
    ).astype(np.int64)

    x_train, y_train = load_core27_split(
        processed_root,
        "train",
        core27_indices,
    )
    x_validation, y_validation = load_core27_split(
        processed_root,
        "validation",
        core27_indices,
    )
    x_test, y_test = load_core27_split(
        processed_root,
        "test",
        core27_indices,
    )

    x_development = torch.cat((x_train, x_validation), dim=0)
    y_development = torch.cat((y_train, y_validation), dim=0)

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Loaded arrays:")
    print(f"  train:       {tuple(x_train.shape)}")
    print(f"  validation:  {tuple(x_validation.shape)}")
    print(f"  development: {tuple(x_development.shape)}")
    print(f"  test:        {tuple(x_test.shape)}")
    print()

    # ------------------------------------------------------------------
    # A. Evaluate the original selected screening checkpoints on subject108
    # ------------------------------------------------------------------
    selected = json.loads(
        (
            screening_root / "selected_configuration.json"
        ).read_text(encoding="utf-8")
    )

    selected_model = str(selected["selected_model_variant"])
    selected_loss = str(selected["selected_loss_variant"])

    screening_runs = pd.read_csv(
        screening_root / "screening_runs.csv"
    )

    selected_rows = screening_runs[
        (screening_runs["model_variant"] == selected_model)
        & (screening_runs["loss_variant"] == selected_loss)
    ].sort_values("seed")

    if len(selected_rows) != 3:
        raise RuntimeError(
            "Expected 3 selected screening checkpoints."
        )

    comparison_rows: list[dict[str, object]] = []
    screening_confusions: list[np.ndarray] = []

    print("A. Evaluating original screening checkpoints on subject108...")

    for row in selected_rows.itertuples(index=False):
        seed = int(row.seed)
        checkpoint_path = Path(row.checkpoint_path)

        model = build_model(selected_model)
        model.load_state_dict(
            load_checkpoint_state(checkpoint_path)
        )
        model.to(device)

        metrics, y_true, y_pred = evaluate(
            model,
            test_loader,
            device,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        )
        screening_confusions.append(matrix)

        comparison_rows.append(
            {
                "source": "screening_checkpoint",
                "seed": seed,
                "training_data": "train_only",
                "epochs": int(row.best_epoch),
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
            }
        )

        print(
            f"  seed={seed}: "
            f"test_macro_f1={metrics['macro_f1']:.4f}; "
            f"test_bal_acc={metrics['balanced_accuracy']:.4f}; "
            f"test_acc={metrics['accuracy']:.4f}"
        )

    print()

    # ------------------------------------------------------------------
    # B. Load the already-computed final retraining results
    # ------------------------------------------------------------------
    print("B. Loading final train+validation retraining results...")

    final_runs = pd.read_csv(
        final_root / "final_test_runs.csv"
    )

    for row in final_runs.itertuples(index=False):
        comparison_rows.append(
            {
                "source": "final_retrained",
                "seed": int(row.seed),
                "training_data": "train_plus_validation",
                "epochs": int(row.fixed_epochs),
                "macro_f1": float(row.test_macro_f1),
                "balanced_accuracy": float(
                    row.test_balanced_accuracy
                ),
                "accuracy": float(row.test_accuracy),
            }
        )

    print(
        f"  loaded {len(final_runs)} final retrained runs."
    )
    print()

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        output_root / "checkpoint_vs_retraining_test_results.csv",
        index=False,
    )

    source_summary = (
        comparison_df.groupby(
            ["source", "training_data"],
            as_index=False,
        )
        .agg(
            runs=("seed", "count"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )
    source_summary.to_csv(
        output_root / "checkpoint_vs_retraining_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # C. Class-level shift profile
    # ------------------------------------------------------------------
    print("C. Computing class-level development-to-test shift profile...")

    shift_df = class_shift_profile(
        x_development,
        y_development,
        x_test,
        y_test,
    )
    shift_df.to_csv(
        output_root / "class_shift_profile.csv",
        index=False,
    )

    print("[OK] Class shift profile computed.")
    print()

    # ------------------------------------------------------------------
    # D. Aggregate screening-checkpoint confusion analysis
    # ------------------------------------------------------------------
    aggregate_screening_confusion = np.sum(
        np.stack(screening_confusions, axis=0),
        axis=0,
    )

    pd.DataFrame(
        aggregate_screening_confusion,
        index=[
            f"true_{i}_{ACTIVITY_NAMES[i]}"
            for i in range(NUM_CLASSES)
        ],
        columns=[
            f"pred_{i}_{ACTIVITY_NAMES[i]}"
            for i in range(NUM_CLASSES)
        ],
    ).to_csv(
        output_root
        / "aggregate_screening_checkpoints_test_confusion_matrix.csv"
    )

    confusions = top_confusions(
        aggregate_screening_confusion,
        top_k=20,
    )
    pd.DataFrame(confusions).to_csv(
        output_root / "top_screening_checkpoint_confusions.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # E. Final decision logic
    # ------------------------------------------------------------------
    screening_summary_row = source_summary[
        source_summary["source"] == "screening_checkpoint"
    ].iloc[0]
    final_summary_row = source_summary[
        source_summary["source"] == "final_retrained"
    ].iloc[0]

    screening_mean_f1 = float(
        screening_summary_row["mean_macro_f1"]
    )
    final_mean_f1 = float(
        final_summary_row["mean_macro_f1"]
    )
    delta = screening_mean_f1 - final_mean_f1

    if delta >= 0.10:
        diagnosis = (
            "FINAL_RETRAINING_PROTOCOL_IS_A_MAJOR_FAILURE_MODE"
        )
        interpretation = (
            "The original validation-selected checkpoints generalize "
            "substantially better to subject108 than the train+validation "
            "fixed-epoch retraining protocol. The final retraining design "
            "should be discarded."
        )
    elif screening_mean_f1 < 0.45:
        diagnosis = (
            "SINGLE_SUBJECT_GENERALIZATION_IS_THE_PRIMARY_FAILURE_MODE"
        )
        interpretation = (
            "The original screening checkpoints also generalize poorly to "
            "subject108. The main problem is not final retraining; it is "
            "instability across held-out subjects and the inadequacy of "
            "single-subject validation for model selection."
        )
    else:
        diagnosis = (
            "MIXED_FAILURE_MODE_REQUIRES_MODEL_AND_PROTOCOL_REDESIGN"
        )
        interpretation = (
            "Both subject shift and final retraining contribute materially. "
            "A revised subject-wise evaluation protocol is required."
        )

    report_lines = [
        "PAMAP2 GENERALIZATION DIAGNOSTIC REPORT",
        "=" * 78,
        f"Processed dataset: {processed_root}",
        f"Selected model: {selected_model}",
        f"Selected loss: {selected_loss}",
        "",
        "IMPORTANT STATUS",
        "-" * 78,
        "The subject108 test result has already been observed.",
        "Therefore, subject108 can no longer be treated as an untouched",
        "hold-out set for any future tuning decision.",
        "",
        "A. ORIGINAL SCREENING CHECKPOINTS ON SUBJECT108",
        "-" * 78,
        f"Mean Macro-F1: "
        f"{screening_mean_f1:.4f} +/- "
        f"{float(screening_summary_row['std_macro_f1']):.4f}",
        f"Mean balanced accuracy: "
        f"{float(screening_summary_row['mean_balanced_accuracy']):.4f}",
        f"Mean accuracy: "
        f"{float(screening_summary_row['mean_accuracy']):.4f}",
        "",
        "B. FINAL TRAIN+VALIDATION RETRAINING ON SUBJECT108",
        "-" * 78,
        f"Mean Macro-F1: "
        f"{final_mean_f1:.4f} +/- "
        f"{float(final_summary_row['std_macro_f1']):.4f}",
        f"Mean balanced accuracy: "
        f"{float(final_summary_row['mean_balanced_accuracy']):.4f}",
        f"Mean accuracy: "
        f"{float(final_summary_row['mean_accuracy']):.4f}",
        "",
        "COMPARISON",
        "-" * 78,
        f"Screening checkpoint minus final retraining Macro-F1: {delta:+.4f}",
        "",
        "AUTOMATIC DIAGNOSIS",
        "-" * 78,
        diagnosis,
        interpretation,
        "",
        "LARGEST DEVELOPMENT-TO-TEST CLASS SHIFTS",
        "-" * 78,
    ]

    for row in shift_df.head(6).itertuples(index=False):
        report_lines.append(
            f"{row.activity_name}: "
            f"RMS shift={row.centroid_rms_shift:.3f}; "
            f"max abs shift={row.centroid_max_abs_shift:.3f}"
        )

    report_lines.extend(
        [
            "",
            "TOP AGGREGATE CONFUSIONS OF ORIGINAL SCREENING CHECKPOINTS",
            "-" * 78,
        ]
    )

    for item in confusions[:10]:
        report_lines.append(
            f"{item['true_activity']} -> "
            f"{item['predicted_activity']}: "
            f"{item['fraction_of_true_class']:.1%} "
            f"({item['count']} windows across 3 checkpoints)"
        )

    report_lines.extend(
        [
            "",
            "NEXT DECISION",
            "-" * 78,
            "Do not start federated-learning experiments yet.",
            "Use this diagnosis to decide whether the main correction is:",
            "1. discard the final retraining protocol;",
            "2. redesign subject-wise validation/evaluation;",
            "3. redesign the normalization/model for cross-subject robustness.",
        ]
    )

    report_path = output_root / "GENERALIZATION_DIAGNOSTIC_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=== Diagnosis completed successfully ===")
    print(f"Screening-checkpoint mean test Macro-F1: {screening_mean_f1:.4f}")
    print(f"Final-retraining mean test Macro-F1:    {final_mean_f1:.4f}")
    print(f"Difference:                              {delta:+.4f}")
    print(f"Automatic diagnosis: {diagnosis}")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDiagnosis interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
