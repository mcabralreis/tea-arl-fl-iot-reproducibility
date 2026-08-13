from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        silhouette_score,
    )
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "ERROR: required packages are missing from the project environment."
    ) from exc


# =============================================================================
# Frozen FedLE-adapted smoke condition
# =============================================================================

OUTER_FOLD = 1
ALPHA = 1.0
SCENARIO = "clean"
METHOD = "fedle_adapted"
FL_SEED = 123

SMOKE_ROUNDS = 3

CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8
NUM_CLUSTERS = CLIENTS_PER_ROUND

NUM_CLASSES = 12
MODEL_CHANNELS = (32, 64, 96)
MODEL_DROPOUT = 0.20
MODEL_GROUPS = 8
EXPECTED_PARAMETER_COUNT = 77004

LOCAL_BATCH_SIZE = 64
LOCAL_EPOCHS = 1
LOCAL_LEARNING_RATE = 1e-3
LOCAL_WEIGHT_DECAY = 1e-4

PREFLIGHT_LOCAL_EPOCHS = 1

CRITICAL_ENERGY = 0.10
STANDBY_COST = 0.0005
COMMUNICATION_COST = 0.004
COMPUTE_COEFFICIENT = 0.016

BASE_LOCAL_SEED = 20260706
BASE_FEDLE_SEED = 20260706


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % (2**31 - 1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed_all(seed)
        except Exception:
            pass


def select_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "xpu":
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def alpha_label(alpha: float) -> str:
    if math.isclose(alpha, 1.0):
        return "alpha1p0"
    if math.isclose(alpha, 0.1):
        return "alpha0p1"
    return str(alpha).replace(".", "p")


def parse_subject(value: object) -> int:
    text = str(value).strip()
    if text.lower().startswith("subject"):
        text = text[7:]
    return int(text)


# =============================================================================
# Data
# =============================================================================

@dataclass(frozen=True)
class FullDataset:
    x_raw_full36: np.ndarray
    y: np.ndarray
    subject: np.ndarray


def load_all_raw_scale_windows(processed_root: Path) -> FullDataset:
    split_dir = processed_root / "splits"
    stats_dir = processed_root / "statistics"

    mean = np.load(
        stats_dir / "training_mean_full36.npy"
    ).astype(np.float32)
    std = np.load(
        stats_dir / "training_std_full36.npy"
    ).astype(np.float32)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []

    for split in ("train", "validation", "test"):
        x_norm = np.load(
            split_dir / f"{split}_X_full36.npy",
            mmap_mode="r",
        )
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

        x_raw = (
            np.asarray(x_norm, dtype=np.float32)
            * std[None, None, :]
            + mean[None, None, :]
        ).astype(np.float32, copy=False)

        x_parts.append(x_raw)
        y_parts.append(y)
        subject_parts.append(subject)

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject = np.concatenate(subject_parts, axis=0)

    if x.shape != (14972, 256, 36):
        raise RuntimeError(f"Unexpected X shape: {x.shape}")
    if y.shape != (14972,):
        raise RuntimeError(f"Unexpected y shape: {y.shape}")
    if subject.shape != (14972,):
        raise RuntimeError(f"Unexpected subject shape: {subject.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite reconstructed raw-scale values.")

    return FullDataset(
        x_raw_full36=x,
        y=y,
        subject=subject,
    )


def vector_magnitude_block(
    x_raw: np.ndarray,
    start_index: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []

    for position in range(3):
        start = start_index + 3 * position
        triple = x_raw[:, :, start : start + 3]
        magnitude = np.sqrt(
            np.sum(
                np.square(triple, dtype=np.float32),
                axis=2,
            )
        )
        parts.append(magnitude[:, :, None])

    return np.concatenate(parts, axis=2)


def build_magnitude6(x_raw: np.ndarray) -> np.ndarray:
    acc16 = vector_magnitude_block(x_raw, 0)
    gyro = vector_magnitude_block(x_raw, 18)

    return np.ascontiguousarray(
        np.concatenate((acc16, gyro), axis=2),
        dtype=np.float32,
    )


def fit_zscore(
    x_representation: np.ndarray,
    retained_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = x_representation[retained_rows]

    mean = train_values.mean(
        axis=(0, 1),
        dtype=np.float64,
    ).astype(np.float32)
    std = train_values.std(
        axis=(0, 1),
        dtype=np.float64,
    ).astype(np.float32)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Non-finite normalization statistics.")
    if np.any(std <= 1e-8):
        raise RuntimeError("Near-zero normalization standard deviation.")

    return mean, std


def normalize_rows(
    x_representation: np.ndarray,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    x = (
        (x_representation[rows] - mean[None, None, :])
        / std[None, None, :]
    ).astype(np.float32, copy=False)

    x = np.ascontiguousarray(
        x.transpose(0, 2, 1),
        dtype=np.float32,
    )

    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite normalized values.")

    return torch.from_numpy(x)


# =============================================================================
# Model
# =============================================================================

class ConvGroupNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        if out_channels % MODEL_GROUPS != 0:
            raise RuntimeError("GroupNorm channel/group mismatch.")

        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(MODEL_GROUPS, out_channels),
            nn.ReLU(inplace=True),
        )


class LightweightCNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        c1, c2, c3 = MODEL_CHANNELS

        self.features = nn.Sequential(
            ConvGroupNormReLU(6, c1, 7, 2),
            ConvGroupNormReLU(c1, c1, 5, 1),
            ConvGroupNormReLU(c1, c2, 5, 2),
            ConvGroupNormReLU(c2, c2, 3, 1),
            ConvGroupNormReLU(c2, c3, 3, 2),
            ConvGroupNormReLU(c3, c3, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(c3, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def extract_partial_update_vector(
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
) -> np.ndarray:
    """
    FedLE-adapted clustering representation:
    normalized partial local update delta = theta_local - theta_global
    for:
    - first convolution weight
    - final linear weight and bias

    Using deltas removes the common global initialization that otherwise
    dominates cosine similarity between full local parameter vectors.
    """
    keys = (
        "features.0.0.weight",
        "classifier.2.weight",
        "classifier.2.bias",
    )

    parts: list[np.ndarray] = []

    for key in keys:
        if key not in local_state:
            raise KeyError(f"Missing local partial-model key: {key}")
        if key not in global_state:
            raise KeyError(f"Missing global partial-model key: {key}")

        delta = (
            local_state[key].detach().cpu().to(dtype=torch.float64)
            - global_state[key].detach().cpu().to(dtype=torch.float64)
        )

        parts.append(
            delta.numpy().astype(np.float32, copy=False).ravel()
        )

    vector = np.concatenate(parts).astype(np.float32, copy=False)

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Invalid or near-zero partial-update vector norm.")

    return vector / norm


# =============================================================================
# Metrics and training
# =============================================================================

@dataclass
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float,
) -> Metrics:
    return Metrics(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Metrics:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_examples += batch_size

        prediction = logits.argmax(dim=1)

        all_true.append(y_batch.cpu().numpy())
        all_pred.append(prediction.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return compute_metrics(
        y_true,
        y_pred,
        total_loss / max(total_examples, 1),
    )


@dataclass
class LocalTrainResult:
    client_id: str
    windows: int
    train_loss: float
    train_macro_f1: float
    wall_seconds: float
    state_dict: dict[str, torch.Tensor]


def train_one_client(
    *,
    global_state: dict[str, torch.Tensor],
    x_client: torch.Tensor,
    y_client: torch.Tensor,
    client_id: str,
    local_seed: int,
    local_epochs: int,
    device: torch.device,
) -> LocalTrainResult:
    set_seed(local_seed)

    model = LightweightCNN1D()
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LOCAL_LEARNING_RATE,
        weight_decay=LOCAL_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(local_seed)

    loader = DataLoader(
        TensorDataset(x_client, y_client),
        batch_size=LOCAL_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )

    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    start = time.perf_counter()

    for _ in range(local_epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(y_batch.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size

            prediction = logits.detach().argmax(dim=1)
            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(prediction.cpu().numpy())

    synchronize(device)
    wall_seconds = time.perf_counter() - start

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    result = LocalTrainResult(
        client_id=client_id,
        windows=int(y_client.shape[0]),
        train_loss=total_loss / max(total_examples, 1),
        train_macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        wall_seconds=wall_seconds,
        state_dict=cpu_state_dict(model),
    )

    del model
    del optimizer
    clear_device_cache(device)

    return result


def weighted_fedavg_state(
    *,
    global_state: dict[str, torch.Tensor],
    local_results: list[LocalTrainResult],
) -> dict[str, torch.Tensor]:
    total_windows = sum(result.windows for result in local_results)

    if total_windows <= 0:
        raise RuntimeError("FedAvg received zero total client windows.")

    new_state: dict[str, torch.Tensor] = {}

    for key, global_tensor in global_state.items():
        global64 = global_tensor.to(dtype=torch.float64)
        accumulator = torch.zeros_like(global64)

        for result in local_results:
            weight = result.windows / total_windows
            local64 = result.state_dict[key].to(dtype=torch.float64)
            accumulator.add_(local64 - global64, alpha=weight)

        new_state[key] = (
            global64 + accumulator
        ).to(dtype=global_tensor.dtype)

    return new_state


# =============================================================================
# FedLE-adapted orchestration
# =============================================================================

def participation_cost(
    *,
    client_id: str,
    client_sizes: dict[str, int],
    compute_factor: dict[str, float],
    median_client_windows: float,
) -> float:
    return (
        COMMUNICATION_COST
        + COMPUTE_COEFFICIENT
        * compute_factor[client_id]
        * (
            client_sizes[client_id]
            / median_client_windows
        )
    )


def run_preflight(
    *,
    client_ids: list[str],
    client_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    global_state: dict[str, torch.Tensor],
    residual_energy: dict[str, float],
    client_sizes: dict[str, int],
    compute_factor: dict[str, float],
    median_client_windows: float,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []

    print("Running one-time FedLE-adapted preflight over all 28 clients...")

    for index, client_id in enumerate(client_ids, start=1):
        x_client, y_client = client_tensors[client_id]

        local_seed = deterministic_seed(
            BASE_FEDLE_SEED,
            "fedle_preflight",
            OUTER_FOLD,
            FL_SEED,
            client_id,
        )

        result = train_one_client(
            global_state=global_state,
            x_client=x_client,
            y_client=y_client,
            client_id=client_id,
            local_seed=local_seed,
            local_epochs=PREFLIGHT_LOCAL_EPOCHS,
            device=device,
        )

        vector = extract_partial_update_vector(
            result.state_dict,
            global_state,
        )
        vectors.append(vector)

        cost = participation_cost(
            client_id=client_id,
            client_sizes=client_sizes,
            compute_factor=compute_factor,
            median_client_windows=median_client_windows,
        )

        residual_energy[client_id] = max(
            0.0,
            residual_energy[client_id] - cost,
        )

        rows.append(
            {
                "global_client_id": client_id,
                "client_windows": client_sizes[client_id],
                "preflight_seed": local_seed,
                "preflight_train_loss": result.train_loss,
                "preflight_train_macro_f1": result.train_macro_f1,
                "preflight_wall_seconds": result.wall_seconds,
                "preflight_energy_cost": cost,
                "residual_energy_after_preflight": (
                    residual_energy[client_id]
                ),
            }
        )

        print(
            f"  [{index:02d}/28] {client_id}: "
            f"n={client_sizes[client_id]}; "
            f"local_f1={result.train_macro_f1:.4f}; "
            f"E={residual_energy[client_id]:.4f}"
        )

    matrix = np.stack(vectors, axis=0)

    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite FedLE preflight feature matrix.")

    return pd.DataFrame(rows), matrix


def cluster_preflight_vectors(
    *,
    client_ids: list[str],
    feature_matrix: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, float | None]:
    kmeans_seed = deterministic_seed(
        BASE_FEDLE_SEED,
        "fedle_kmeans",
        OUTER_FOLD,
        FL_SEED,
    )

    model = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=kmeans_seed,
        n_init=20,
    )

    labels = model.fit_predict(feature_matrix)

    unique_clusters = sorted(int(value) for value in np.unique(labels))
    if len(unique_clusters) != NUM_CLUSTERS:
        raise RuntimeError(
            f"Expected {NUM_CLUSTERS} non-empty clusters, "
            f"found {len(unique_clusters)}."
        )

    similarity_matrix = np.clip(
        feature_matrix @ feature_matrix.T,
        -1.0,
        1.0,
    )

    np.fill_diagonal(similarity_matrix, np.nan)

    silhouette: float | None = None
    counts = np.bincount(labels, minlength=NUM_CLUSTERS)

    if (
        len(np.unique(labels)) > 1
        and np.all(counts >= 2)
        and feature_matrix.shape[0] > len(np.unique(labels))
    ):
        silhouette = float(
            silhouette_score(
                feature_matrix,
                labels,
                metric="cosine",
            )
        )

    rows: list[dict[str, object]] = []

    for index, client_id in enumerate(client_ids):
        cluster_id = int(labels[index])
        peers = [
            peer_index
            for peer_index in range(len(client_ids))
            if peer_index != index and int(labels[peer_index]) == cluster_id
        ]

        mean_within_similarity = (
            float(np.nanmean(similarity_matrix[index, peers]))
            if peers
            else float("nan")
        )

        others = [
            peer_index
            for peer_index in range(len(client_ids))
            if int(labels[peer_index]) != cluster_id
        ]

        mean_outside_similarity = (
            float(np.nanmean(similarity_matrix[index, others]))
            if others
            else float("nan")
        )

        rows.append(
            {
                "global_client_id": client_id,
                "cluster_id": cluster_id,
                "cluster_size": int(counts[cluster_id]),
                "mean_within_cluster_cosine": mean_within_similarity,
                "mean_outside_cluster_cosine": mean_outside_similarity,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(
            ["cluster_id", "global_client_id"]
        ),
        similarity_matrix,
        silhouette,
    )


def select_fedle_clients(
    *,
    round_index: int,
    client_ids: list[str],
    residual_energy: dict[str, float],
    initial_energy: dict[str, float],
    cluster_assignment: dict[str, int],
    cluster_sizes: dict[int, int],
) -> tuple[list[str], pd.DataFrame]:
    eligible = [
        client_id
        for client_id in client_ids
        if residual_energy[client_id] >= CRITICAL_ENERGY
    ]

    if len(eligible) < CLIENTS_PER_ROUND:
        raise RuntimeError(
            f"Round {round_index}: only {len(eligible)} eligible clients."
        )

    scores = np.asarray(
        [
            (
                residual_energy[client_id]
                / initial_energy[client_id]
            )
            / cluster_sizes[cluster_assignment[client_id]]
            for client_id in eligible
        ],
        dtype=np.float64,
    )

    if not np.isfinite(scores).all() or np.any(scores <= 0):
        raise RuntimeError("Invalid FedLE selection scores.")

    probabilities = scores / scores.sum()

    rng = np.random.default_rng(
        deterministic_seed(
            BASE_FEDLE_SEED,
            "fedle_selection",
            OUTER_FOLD,
            FL_SEED,
            round_index,
        )
    )

    chosen_indices = rng.choice(
        np.arange(len(eligible)),
        size=CLIENTS_PER_ROUND,
        replace=False,
        p=probabilities,
    )

    selected = [
        eligible[int(index)]
        for index in chosen_indices
    ]

    audit_rows: list[dict[str, object]] = []

    for client_id, score, probability in zip(
        eligible,
        scores,
        probabilities,
    ):
        audit_rows.append(
            {
                "round": round_index,
                "global_client_id": client_id,
                "cluster_id": cluster_assignment[client_id],
                "cluster_size": cluster_sizes[
                    cluster_assignment[client_id]
                ],
                "residual_energy_fraction": (
                    residual_energy[client_id]
                    / initial_energy[client_id]
                ),
                "selection_score": float(score),
                "selection_probability": float(probability),
                "selected": client_id in set(selected),
            }
        )

    return selected, pd.DataFrame(audit_rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-round FedLE-adapted smoke test with charged preflight, "
            "partial-model clustering, and energy-aware client selection."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()

    processed_root = (
        project_root
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128"
    )
    fl_protocol_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_experiment_v1"
    )
    partition_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_partitions_v1"
    )
    evaluation_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_evaluation_v2"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "federated"
        / "pamap2"
        / "smoke_fedle_adapted_delta_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_path = (
        fl_protocol_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json"
    )
    condition_path = (
        fl_protocol_root / "matched_condition_manifest.csv"
    )
    energy_path = (
        fl_protocol_root / "client_energy_profile.csv"
    )
    client_manifest_path = (
        partition_root / "outer_fold_client_manifest.csv"
    )
    assignment_path = (
        partition_root
        / f"master_assignments_{alpha_label(ALPHA)}.csv"
    )
    outer_manifest_path = (
        evaluation_root / "outer_fold_manifest.csv"
    )

    for path in (
        protocol_path,
        condition_path,
        energy_path,
        client_manifest_path,
        assignment_path,
        outer_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    if protocol.get("status") != "FROZEN_BEFORE_ANY_FL_TRAINING":
        raise RuntimeError("Unexpected FL protocol status.")

    conditions = pd.read_csv(condition_path)
    energy_profiles = pd.read_csv(energy_path)
    client_manifest = pd.read_csv(client_manifest_path)
    assignments = pd.read_csv(assignment_path)
    outer_manifest = pd.read_csv(outer_manifest_path)

    condition = conditions[
        (conditions["outer_fold"].astype(int) == OUTER_FOLD)
        & np.isclose(conditions["alpha"].astype(float), ALPHA)
        & (conditions["scenario"].astype(str) == SCENARIO)
        & (conditions["fl_seed"].astype(int) == FL_SEED)
    ]

    if len(condition) != 1:
        raise RuntimeError(
            f"Expected one matched condition, found {len(condition)}."
        )

    condition_row = condition.iloc[0]
    model_seed = int(condition_row["model_seed"])

    outer_row = outer_manifest[
        outer_manifest["outer_fold"].astype(int) == OUTER_FOLD
    ]
    if len(outer_row) != 1:
        raise RuntimeError("Expected one outer-fold row.")

    outer_test_subject = parse_subject(
        outer_row.iloc[0]["outer_test_subject"]
    )

    fold_clients = client_manifest[
        (client_manifest["outer_fold"].astype(int) == OUTER_FOLD)
        & np.isclose(client_manifest["alpha"].astype(float), ALPHA)
    ].copy()

    client_ids = sorted(
        fold_clients["global_client_id"].astype(str).unique().tolist()
    )

    if len(client_ids) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 clients, found {len(client_ids)}."
        )

    fold_assignments = assignments[
        assignments["global_client_id"].astype(str).isin(client_ids)
    ].copy()

    if outer_test_subject in set(
        fold_assignments["subject_id"].astype(int)
    ):
        raise RuntimeError("Outer test subject leaked into training.")

    retained_rows = np.sort(
        fold_assignments["row_index"].astype(np.int64).unique()
    )

    config = {
        "status": "FROZEN_FEDLE_ADAPTED_SMOKE_CONFIGURATION",
        "fl_protocol_path": str(protocol_path),
        "fl_protocol_sha256": sha256_file(protocol_path),
        "condition_id": int(condition_row["condition_id"]),
        "outer_fold": OUTER_FOLD,
        "outer_test_subject": outer_test_subject,
        "alpha": ALPHA,
        "scenario": SCENARIO,
        "method": METHOD,
        "fl_seed": FL_SEED,
        "model_seed": model_seed,
        "rounds": SMOKE_ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "preflight_local_epochs": PREFLIGHT_LOCAL_EPOCHS,
        "clustering_representation": "normalized partial local update delta",
        "partial_model_layers": [
            "features.0.0.weight",
            "classifier.2.weight",
            "classifier.2.bias",
        ],
        "similarity": "cosine on normalized partial update deltas",
        "clustering": "one-time KMeans",
        "num_clusters": NUM_CLUSTERS,
        "selection_score": (
            "(residual_energy / initial_energy) / cluster_size"
        ),
        "preflight_energy_and_communication_charged": True,
        "outer_test_used_for_training_selection_or_normalization": False,
    }

    (output_root / "FEDLE_SMOKE_CONFIG.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    amendment = {
        "amendment_name": "FedLE-adapted clustering representation correction",
        "status": "RECORDED_BEFORE_ANY FULL FEDERATED CAMPAIGN RUN",
        "reason": (
            "The first smoke test showed cosine similarities of full partial "
            "local parameter vectors concentrated near 1.0 because all local "
            "models shared the same dominant global initialization."
        ),
        "previous_representation": (
            "L2-normalized concatenation of selected local model parameters"
        ),
        "corrected_representation": (
            "L2-normalized concatenation of selected local update deltas "
            "(theta_local - theta_global)"
        ),
        "unchanged": [
            "preflight local epochs",
            "selected layers",
            "cosine similarity",
            "one-time KMeans clustering",
            "number of clusters",
            "selection score",
            "energy accounting",
            "aggregation",
        ],
        "scientific_runs_completed_before_amendment": 0,
    }

    (output_root / "FEDLE_TECHNICAL_AMENDMENT_V1.json").write_text(
        json.dumps(amendment, indent=2),
        encoding="utf-8",
    )

    device = select_device()

    print("=== PAMAP2 FedLE-adapted smoke test ===")
    print(f"Condition:         outer={OUTER_FOLD}, alpha={ALPHA}, clean")
    print(f"Method:            {METHOD}")
    print(f"FL seed:           {FL_SEED}")
    print(f"Smoke rounds:      {SMOKE_ROUNDS}")
    print(f"Clients:           {len(client_ids)}")
    print(f"Clusters:          {NUM_CLUSTERS}")
    print(f"Clients per round: {CLIENTS_PER_ROUND}")
    print(f"Outer test:        subject{outer_test_subject}")
    print(f"Output:            {output_root}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")
    if device.type == "xpu":
        print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    print()

    print("Loading data and building Magnitude6...")
    dataset = load_all_raw_scale_windows(processed_root)
    x_magnitude6 = build_magnitude6(dataset.x_raw_full36)

    mean, std = fit_zscore(
        x_magnitude6,
        retained_rows,
    )

    np.save(output_root / "normalization_mean.npy", mean)
    np.save(output_root / "normalization_std.npy", std)

    client_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for client_id in client_ids:
        rows = np.sort(
            fold_assignments[
                fold_assignments["global_client_id"].astype(str)
                == client_id
            ]["row_index"].astype(np.int64).to_numpy()
        )

        client_tensors[client_id] = (
            normalize_rows(
                x_magnitude6,
                rows,
                mean,
                std,
            ),
            torch.from_numpy(
                dataset.y[rows].astype(np.int64, copy=True)
            ),
        )

    test_rows = np.where(
        dataset.subject == outer_test_subject
    )[0].astype(np.int64)

    x_test = normalize_rows(
        x_magnitude6,
        test_rows,
        mean,
        std,
    )
    y_test = torch.from_numpy(
        dataset.y[test_rows].astype(np.int64, copy=True)
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    energy = energy_profiles[
        (energy_profiles["outer_fold"].astype(int) == OUTER_FOLD)
        & (energy_profiles["fl_seed"].astype(int) == FL_SEED)
        & (
            energy_profiles["global_client_id"].astype(str).isin(
                client_ids
            )
        )
    ].copy()

    if len(energy) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 energy profiles, found {len(energy)}."
        )

    energy = (
        energy.set_index("global_client_id")
        .loc[client_ids]
        .reset_index()
    )

    initial_energy = {
        str(row.global_client_id): float(row.initial_energy)
        for row in energy.itertuples(index=False)
    }
    residual_energy = dict(initial_energy)
    compute_factor = {
        str(row.global_client_id): float(row.compute_factor)
        for row in energy.itertuples(index=False)
    }
    client_sizes = {
        client_id: int(client_tensors[client_id][1].shape[0])
        for client_id in client_ids
    }
    median_client_windows = float(
        np.median(list(client_sizes.values()))
    )

    print(f"[OK] Retained training windows: {len(retained_rows)}")
    print(
        f"[OK] Client size range: "
        f"{min(client_sizes.values())}–{max(client_sizes.values())}"
    )
    print(f"[OK] Outer-test windows: {len(y_test)}")
    print()

    set_seed(model_seed)
    global_model = LightweightCNN1D().to(device)

    if count_trainable_parameters(global_model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Unexpected model parameter count.")

    global_state = cpu_state_dict(global_model)

    initial_metrics = evaluate_model(
        global_model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------------
    # Charged one-time preflight and clustering
    # ------------------------------------------------------------------
    preflight_df, feature_matrix = run_preflight(
        client_ids=client_ids,
        client_tensors=client_tensors,
        global_state=global_state,
        residual_energy=residual_energy,
        client_sizes=client_sizes,
        compute_factor=compute_factor,
        median_client_windows=median_client_windows,
        device=device,
    )

    preflight_df.to_csv(
        output_root / "preflight_client_log.csv",
        index=False,
    )

    cluster_df, similarity_matrix, silhouette = (
        cluster_preflight_vectors(
            client_ids=client_ids,
            feature_matrix=feature_matrix,
        )
    )

    cluster_df.to_csv(
        output_root / "cluster_assignment.csv",
        index=False,
    )

    pd.DataFrame(
        similarity_matrix,
        index=client_ids,
        columns=client_ids,
    ).to_csv(
        output_root / "cosine_similarity_matrix.csv"
    )

    cluster_counts = (
        cluster_df.groupby("cluster_id")
        .size()
        .sort_index()
    )

    print()
    print("Preflight clustering:")
    for cluster_id, count in cluster_counts.items():
        print(f"  cluster {int(cluster_id)}: {int(count)} clients")

    if silhouette is None:
        print("  cosine silhouette: not defined (singleton cluster present)")
    else:
        print(f"  cosine silhouette: {silhouette:.4f}")
    print()

    cluster_assignment = {
        str(row.global_client_id): int(row.cluster_id)
        for row in cluster_df.itertuples(index=False)
    }
    cluster_sizes = {
        int(cluster_id): int(count)
        for cluster_id, count in cluster_counts.items()
    }

    # ------------------------------------------------------------------
    # Three FedLE-adapted communication rounds
    # ------------------------------------------------------------------
    learning_rows: list[dict[str, object]] = [
        {
            "round": 0,
            "phase": "before_preflight",
            "test_loss": initial_metrics.loss,
            "test_accuracy": initial_metrics.accuracy,
            "test_balanced_accuracy": (
                initial_metrics.balanced_accuracy
            ),
            "test_macro_f1": initial_metrics.macro_f1,
            "active_clients": CLIENTS_PER_FOLD,
            "mean_residual_energy": float(
                np.mean(list(initial_energy.values()))
            ),
            "min_residual_energy": float(
                np.min(list(initial_energy.values()))
            ),
        }
    ]

    selection_rows: list[dict[str, object]] = []
    selection_probability_frames: list[pd.DataFrame] = []

    for round_index in range(1, SMOKE_ROUNDS + 1):
        round_start = time.perf_counter()

        active_at_start = [
            client_id
            for client_id in client_ids
            if residual_energy[client_id] >= CRITICAL_ENERGY
        ]

        for client_id in active_at_start:
            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - STANDBY_COST,
            )

        selected, selection_audit_df = select_fedle_clients(
            round_index=round_index,
            client_ids=client_ids,
            residual_energy=residual_energy,
            initial_energy=initial_energy,
            cluster_assignment=cluster_assignment,
            cluster_sizes=cluster_sizes,
        )

        selection_probability_frames.append(selection_audit_df)

        local_results: list[LocalTrainResult] = []

        print(
            f"Round {round_index}/{SMOKE_ROUNDS}: "
            f"active={len(active_at_start)}; selected={len(selected)}"
        )

        for selection_rank, client_id in enumerate(selected, start=1):
            x_client, y_client = client_tensors[client_id]

            local_seed = deterministic_seed(
                BASE_LOCAL_SEED,
                OUTER_FOLD,
                FL_SEED,
                round_index,
                client_id,
            )

            result = train_one_client(
                global_state=global_state,
                x_client=x_client,
                y_client=y_client,
                client_id=client_id,
                local_seed=local_seed,
                local_epochs=LOCAL_EPOCHS,
                device=device,
            )
            local_results.append(result)

            cost = participation_cost(
                client_id=client_id,
                client_sizes=client_sizes,
                compute_factor=compute_factor,
                median_client_windows=median_client_windows,
            )

            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - cost,
            )

            selection_rows.append(
                {
                    "round": round_index,
                    "selection_rank": selection_rank,
                    "global_client_id": client_id,
                    "cluster_id": cluster_assignment[client_id],
                    "cluster_size": cluster_sizes[
                        cluster_assignment[client_id]
                    ],
                    "client_windows": client_sizes[client_id],
                    "local_seed": local_seed,
                    "local_train_loss": result.train_loss,
                    "local_train_macro_f1": result.train_macro_f1,
                    "local_wall_seconds": result.wall_seconds,
                    "participation_cost": cost,
                    "residual_energy_after_round": (
                        residual_energy[client_id]
                    ),
                }
            )

            print(
                f"  [{selection_rank}/8] {client_id}: "
                f"cluster={cluster_assignment[client_id]}; "
                f"n={client_sizes[client_id]}; "
                f"local_f1={result.train_macro_f1:.4f}; "
                f"E={residual_energy[client_id]:.4f}"
            )

        global_state = weighted_fedavg_state(
            global_state=global_state,
            local_results=local_results,
        )

        global_model.load_state_dict(global_state)
        global_model.to(device)

        metrics = evaluate_model(
            global_model,
            test_loader,
            device,
        )

        active_after = sum(
            1
            for client_id in client_ids
            if residual_energy[client_id] >= CRITICAL_ENERGY
        )

        synchronize(device)
        wall_seconds = time.perf_counter() - round_start

        learning_rows.append(
            {
                "round": round_index,
                "phase": "post_round",
                "test_loss": metrics.loss,
                "test_accuracy": metrics.accuracy,
                "test_balanced_accuracy": (
                    metrics.balanced_accuracy
                ),
                "test_macro_f1": metrics.macro_f1,
                "active_clients": active_after,
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
            }
        )

        print(
            f"  [GLOBAL] Macro-F1={metrics.macro_f1:.4f}; "
            f"BalAcc={metrics.balanced_accuracy:.4f}; "
            f"Acc={metrics.accuracy:.4f}; "
            f"active={active_after}; "
            f"wall={wall_seconds:.1f}s"
        )
        print()

    learning_df = pd.DataFrame(learning_rows)
    selection_df = pd.DataFrame(selection_rows)
    selection_probability_df = pd.concat(
        selection_probability_frames,
        ignore_index=True,
    )

    learning_df.to_csv(
        output_root / "learning_curve.csv",
        index=False,
    )
    selection_df.to_csv(
        output_root / "selected_client_log.csv",
        index=False,
    )
    selection_probability_df.to_csv(
        output_root / "selection_probability_audit.csv",
        index=False,
    )

    final_energy_df = pd.DataFrame(
        [
            {
                "global_client_id": client_id,
                "cluster_id": cluster_assignment[client_id],
                "initial_energy": initial_energy[client_id],
                "residual_energy": residual_energy[client_id],
                "active": (
                    residual_energy[client_id] >= CRITICAL_ENERGY
                ),
            }
            for client_id in client_ids
        ]
    )
    final_energy_df.to_csv(
        output_root / "final_client_state.csv",
        index=False,
    )

    torch.save(
        {
            "model_state_dict": global_state,
            "method": METHOD,
            "rounds_completed": SMOKE_ROUNDS,
            "cluster_assignment": cluster_assignment,
            "final_metrics": learning_rows[-1],
        },
        output_root / "final_global_model.pt",
    )

    if len(preflight_df) != CLIENTS_PER_FOLD:
        raise RuntimeError("FedLE preflight did not cover all 28 clients.")

    if len(cluster_df) != CLIENTS_PER_FOLD:
        raise RuntimeError("FedLE clustering did not cover all 28 clients.")

    if selection_df.shape[0] != SMOKE_ROUNDS * CLIENTS_PER_ROUND:
        raise RuntimeError("Unexpected number of FedLE local trainings.")

    if not np.all(
        selection_df.groupby("round")["global_client_id"]
        .nunique()
        .to_numpy()
        == CLIENTS_PER_ROUND
    ):
        raise RuntimeError("Duplicate client selected within a FedLE round.")

    if not np.isfinite(
        learning_df[
            [
                "test_loss",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite FedLE smoke metrics.")

    final_metrics = learning_df.iloc[-1]

    report_lines = [
        "PAMAP2 FEDLE-ADAPTED SMOKE TEST",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "CONDITION",
        "-" * 78,
        f"Outer fold: {OUTER_FOLD}",
        f"Outer test: subject{outer_test_subject}",
        f"Alpha: {ALPHA}",
        f"Scenario: {SCENARIO}",
        f"FL seed: {FL_SEED}",
        f"Rounds: {SMOKE_ROUNDS}",
        "",
        "FEDLE-ADAPTED PREFLIGHT",
        "-" * 78,
        f"Clients preflighted: {len(preflight_df)}",
        f"Local epochs per preflight client: {PREFLIGHT_LOCAL_EPOCHS}",
        "Partial update delta: first convolution + final linear layer.",
        "Partial update deltas L2-normalized; cosine similarity audited.",
        f"One-time KMeans clusters: {NUM_CLUSTERS}",
        "Preflight communication and energy costs were charged.",
        (
            "Cosine silhouette: "
            + (
                "not defined (singleton cluster present)"
                if silhouette is None
                else f"{silhouette:.4f}"
            )
        ),
        "",
        "CLUSTER SIZES",
        "-" * 78,
    ]

    for cluster_id, count in cluster_counts.items():
        report_lines.append(
            f"Cluster {int(cluster_id)}: {int(count)} clients"
        )

    report_lines.extend(
        [
            "",
            "SELECTION",
            "-" * 78,
            "Selection score = residual-energy fraction / cluster size.",
            "Eight clients sampled without replacement from normalized scores.",
            "",
            "DATA AND LEAKAGE",
            "-" * 78,
            f"Retained training windows: {len(retained_rows)}",
            f"Outer-test windows: {len(y_test)}",
            "Z-score fitted only on retained outer-training client windows.",
            "Outer test did not contribute to preflight, clustering, selection, training, or normalization.",
            "",
            "RESULTS",
            "-" * 78,
            f"Round 0 Macro-F1: {learning_df.iloc[0]['test_macro_f1']:.4f}",
            (
                f"Round {SMOKE_ROUNDS} Macro-F1: "
                f"{float(final_metrics['test_macro_f1']):.4f}"
            ),
            (
                f"Round {SMOKE_ROUNDS} balanced accuracy: "
                f"{float(final_metrics['test_balanced_accuracy']):.4f}"
            ),
            (
                f"Round {SMOKE_ROUNDS} accuracy: "
                f"{float(final_metrics['test_accuracy']):.4f}"
            ),
            (
                f"Final active clients: "
                f"{int(final_metrics['active_clients'])}/{CLIENTS_PER_FOLD}"
            ),
            (
                "Final mean residual energy: "
                f"{float(final_metrics['mean_residual_energy']):.4f}"
            ),
            (
                "Final minimum residual energy: "
                f"{float(final_metrics['min_residual_energy']):.4f}"
            ),
            "",
            "FILES",
            "-" * 78,
            "FEDLE_SMOKE_CONFIG.json",
            "FEDLE_TECHNICAL_AMENDMENT_V1.json",
            "FEDLE_SMOKE_REPORT.txt",
            "preflight_client_log.csv",
            "cluster_assignment.csv",
            "cosine_similarity_matrix.csv",
            "selection_probability_audit.csv",
            "selected_client_log.csv",
            "learning_curve.csv",
            "final_client_state.csv",
            "normalization_mean.npy",
            "normalization_std.npy",
            "final_global_model.pt",
        ]
    )

    report_path = output_root / "FEDLE_SMOKE_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=== FedLE-adapted smoke test completed successfully ===")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nFedLE smoke test interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
