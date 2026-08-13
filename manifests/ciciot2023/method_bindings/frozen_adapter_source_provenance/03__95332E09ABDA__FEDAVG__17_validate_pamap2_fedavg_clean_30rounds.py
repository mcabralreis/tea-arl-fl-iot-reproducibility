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
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "ERROR: required packages are missing from the project environment."
    ) from exc


# =============================================================================
# Frozen technical-validation condition
# =============================================================================

OUTER_FOLD = 1
ALPHA = 1.0
SCENARIO = "clean"
METHOD = "fedavg"
FL_SEED = 123

TOTAL_ROUNDS = 30
CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8
EVALUATION_ROUNDS = {0, 1, 2, 3, 5, 10, 15, 20, 25, 30}

NUM_CLASSES = 12
MODEL_CHANNELS = (32, 64, 96)
MODEL_DROPOUT = 0.20
MODEL_GROUPS = 8
EXPECTED_PARAMETER_COUNT = 77004

LOCAL_BATCH_SIZE = 64
LOCAL_EPOCHS = 1
LOCAL_LEARNING_RATE = 1e-3
LOCAL_WEIGHT_DECAY = 1e-4

CRITICAL_ENERGY = 0.10
STANDBY_COST = 0.0005
COMMUNICATION_COST = 0.004
COMPUTE_COEFFICIENT = 0.016

BASE_LOCAL_SEED = 20260706


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

    return FullDataset(x_raw_full36=x, y=y, subject=subject)


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


def fit_retained_client_zscore(
    x_representation: np.ndarray,
    retained_row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = x_representation[retained_row_indices]

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
    row_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    x = (
        (x_representation[row_indices] - mean[None, None, :])
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


# =============================================================================
# Metrics and local training
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

    for _ in range(LOCAL_EPOCHS):
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


def fedavg_weighted_state(
    *,
    global_state: dict[str, torch.Tensor],
    local_results: list[LocalTrainResult],
) -> dict[str, torch.Tensor]:
    total_windows = sum(result.windows for result in local_results)

    if total_windows <= 0:
        raise RuntimeError("FedAvg received zero total windows.")

    new_state: dict[str, torch.Tensor] = {}

    for key, global_tensor in global_state.items():
        accumulator = torch.zeros_like(
            global_tensor,
            dtype=torch.float64,
        )

        global64 = global_tensor.to(dtype=torch.float64)

        for result in local_results:
            weight = result.windows / total_windows
            local64 = result.state_dict[key].to(dtype=torch.float64)
            accumulator.add_(local64 - global64, alpha=weight)

        new_state[key] = (
            global64 + accumulator
        ).to(dtype=global_tensor.dtype)

    return new_state


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable 30-round clean FedAvg technical validation for the "
            "frozen PAMAP2 federated protocol."
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
        / "validation_fedavg_clean_30r_v1"
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
    random_schedule_seed = int(condition_row["random_schedule_seed"])

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
        raise RuntimeError("Outer test subject leaked into training clients.")

    retained_rows = np.sort(
        fold_assignments["row_index"].astype(np.int64).unique()
    )

    config = {
        "status": "FROZEN_TECHNICAL_VALIDATION",
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
        "random_schedule_seed": random_schedule_seed,
        "total_rounds": TOTAL_ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "evaluation_rounds": sorted(EVALUATION_ROUNDS),
        "resume_checkpoint": "latest_checkpoint.pt",
        "outer_test_used_for_training_selection_or_normalization": False,
    }

    config_path = output_root / "VALIDATION_CONFIG.json"

    if config_path.is_file():
        existing = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        if existing != config:
            raise RuntimeError(
                "Existing VALIDATION_CONFIG.json does not match this run."
            )
    else:
        config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

    device = select_device()

    print("=== PAMAP2 30-round FedAvg clean validation ===")
    print(f"Condition:         outer={OUTER_FOLD}, alpha={ALPHA}, clean")
    print(f"Method:            {METHOD}")
    print(f"FL seed:           {FL_SEED}")
    print(f"Rounds:            {TOTAL_ROUNDS}")
    print(f"Clients:           {len(client_ids)}")
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

    mean, std = fit_retained_client_zscore(
        x_magnitude6,
        retained_rows,
    )

    np.save(output_root / "normalization_mean.npy", mean)
    np.save(output_root / "normalization_std.npy", std)

    client_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for client_id in client_ids:
        client_rows = np.sort(
            fold_assignments[
                fold_assignments["global_client_id"].astype(str)
                == client_id
            ]["row_index"].astype(np.int64).to_numpy()
        )

        x_client = normalize_rows(
            x_magnitude6,
            client_rows,
            mean,
            std,
        )
        y_client = torch.from_numpy(
            dataset.y[client_rows].astype(np.int64, copy=True)
        )

        client_tensors[client_id] = (x_client, y_client)

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

    print(f"[OK] Retained training windows: {len(retained_rows)}")
    print(
        f"[OK] Client size range: "
        f"{min(len(v[1]) for v in client_tensors.values())}–"
        f"{max(len(v[1]) for v in client_tensors.values())}"
    )
    print(f"[OK] Outer-test windows: {len(y_test)}")
    print()

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

    checkpoint_path = output_root / "latest_checkpoint.pt"
    learning_curve_path = output_root / "learning_curve.csv"
    selection_log_path = output_root / "selected_client_log.csv"
    round_progress_path = output_root / "round_progress.csv"

    set_seed(model_seed)
    global_model = LightweightCNN1D().to(device)

    if count_trainable_parameters(global_model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Unexpected model parameter count.")

    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        completed_round = int(checkpoint["completed_round"])
        global_state = checkpoint["global_state"]
        residual_energy = {
            str(key): float(value)
            for key, value in checkpoint["residual_energy"].items()
        }

        global_model.load_state_dict(global_state)
        global_model.to(device)

        learning_rows = (
            pd.read_csv(learning_curve_path).to_dict(orient="records")
            if learning_curve_path.is_file()
            else []
        )
        selection_rows = (
            pd.read_csv(selection_log_path).to_dict(orient="records")
            if selection_log_path.is_file()
            else []
        )
        progress_rows = (
            pd.read_csv(round_progress_path).to_dict(orient="records")
            if round_progress_path.is_file()
            else []
        )

        print(
            f"[RESUME] Loaded completed round {completed_round}. "
            f"Continuing from round {completed_round + 1}."
        )
    else:
        completed_round = 0
        global_state = cpu_state_dict(global_model)
        residual_energy = dict(initial_energy)
        learning_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        progress_rows: list[dict[str, object]] = []

        initial_metrics = evaluate_model(
            global_model,
            test_loader,
            device,
        )

        learning_rows.append(
            {
                "round": 0,
                "test_loss": initial_metrics.loss,
                "test_accuracy": initial_metrics.accuracy,
                "test_balanced_accuracy": (
                    initial_metrics.balanced_accuracy
                ),
                "test_macro_f1": initial_metrics.macro_f1,
                "active_clients": CLIENTS_PER_FOLD,
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
            }
        )

        pd.DataFrame(learning_rows).to_csv(
            learning_curve_path,
            index=False,
        )

        torch.save(
            {
                "completed_round": 0,
                "global_state": global_state,
                "residual_energy": residual_energy,
            },
            checkpoint_path,
        )

        print(
            f"Round 0: Macro-F1={initial_metrics.macro_f1:.4f}; "
            f"BalAcc={initial_metrics.balanced_accuracy:.4f}; "
            f"Acc={initial_metrics.accuracy:.4f}"
        )

    if completed_round >= TOTAL_ROUNDS:
        print("[OK] Validation already complete. Regenerating final report.")
    else:
        for round_index in range(completed_round + 1, TOTAL_ROUNDS + 1):
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

            eligible = [
                client_id
                for client_id in client_ids
                if residual_energy[client_id] >= CRITICAL_ENERGY
            ]

            if len(eligible) < CLIENTS_PER_ROUND:
                raise RuntimeError(
                    f"Round {round_index}: only {len(eligible)} "
                    "eligible clients."
                )

            permutation_rng = np.random.default_rng(
                deterministic_seed(
                    random_schedule_seed,
                    "round_permutation",
                    round_index,
                )
            )
            full_permutation = permutation_rng.permutation(
                np.asarray(client_ids, dtype=object)
            ).tolist()

            eligible_set = set(eligible)
            selected = [
                client_id
                for client_id in full_permutation
                if client_id in eligible_set
            ][:CLIENTS_PER_ROUND]

            if len(selected) != CLIENTS_PER_ROUND:
                raise RuntimeError("Could not select 8 eligible clients.")

            local_results: list[LocalTrainResult] = []

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
                    device=device,
                )
                local_results.append(result)

                participation_cost = (
                    COMMUNICATION_COST
                    + COMPUTE_COEFFICIENT
                    * compute_factor[client_id]
                    * (
                        client_sizes[client_id]
                        / median_client_windows
                    )
                )

                residual_energy[client_id] = max(
                    0.0,
                    residual_energy[client_id] - participation_cost,
                )

                selection_rows.append(
                    {
                        "round": round_index,
                        "selection_rank": selection_rank,
                        "global_client_id": client_id,
                        "client_windows": client_sizes[client_id],
                        "local_seed": local_seed,
                        "local_train_loss": result.train_loss,
                        "local_train_macro_f1": result.train_macro_f1,
                        "local_wall_seconds": result.wall_seconds,
                        "participation_cost": participation_cost,
                        "residual_energy_after_round": residual_energy[
                            client_id
                        ],
                    }
                )

            global_state = fedavg_weighted_state(
                global_state=global_state,
                local_results=local_results,
            )

            global_model.load_state_dict(global_state)
            global_model.to(device)

            active_after = sum(
                1
                for client_id in client_ids
                if residual_energy[client_id] >= CRITICAL_ENERGY
            )

            round_wall_seconds = time.perf_counter() - round_start

            evaluated = round_index in EVALUATION_ROUNDS
            metrics: Metrics | None = None

            if evaluated:
                metrics = evaluate_model(
                    global_model,
                    test_loader,
                    device,
                )

                learning_rows.append(
                    {
                        "round": round_index,
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

            progress_rows.append(
                {
                    "round": round_index,
                    "selected_clients": len(selected),
                    "active_clients": active_after,
                    "mean_residual_energy": float(
                        np.mean(list(residual_energy.values()))
                    ),
                    "min_residual_energy": float(
                        np.min(list(residual_energy.values()))
                    ),
                    "round_wall_seconds": round_wall_seconds,
                    "evaluated": evaluated,
                    "test_macro_f1": (
                        None if metrics is None else metrics.macro_f1
                    ),
                }
            )

            pd.DataFrame(learning_rows).to_csv(
                learning_curve_path,
                index=False,
            )
            pd.DataFrame(selection_rows).to_csv(
                selection_log_path,
                index=False,
            )
            pd.DataFrame(progress_rows).to_csv(
                round_progress_path,
                index=False,
            )

            torch.save(
                {
                    "completed_round": round_index,
                    "global_state": global_state,
                    "residual_energy": residual_energy,
                },
                checkpoint_path,
            )

            if metrics is None:
                print(
                    f"Round {round_index:02d}/{TOTAL_ROUNDS}: "
                    f"active={active_after}; "
                    f"Emean={np.mean(list(residual_energy.values())):.4f}; "
                    f"wall={round_wall_seconds:.1f}s"
                )
            else:
                print(
                    f"Round {round_index:02d}/{TOTAL_ROUNDS}: "
                    f"Macro-F1={metrics.macro_f1:.4f}; "
                    f"BalAcc={metrics.balanced_accuracy:.4f}; "
                    f"Acc={metrics.accuracy:.4f}; "
                    f"active={active_after}; "
                    f"Emean={np.mean(list(residual_energy.values())):.4f}; "
                    f"wall={round_wall_seconds:.1f}s"
                )

    learning_df = pd.read_csv(learning_curve_path)
    progress_df = pd.read_csv(round_progress_path)
    selection_df = pd.read_csv(selection_log_path)

    latest = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    completed_round = int(latest["completed_round"])

    if completed_round != TOTAL_ROUNDS:
        raise RuntimeError(
            f"Validation incomplete: checkpoint is at round {completed_round}."
        )

    if len(selection_df) != TOTAL_ROUNDS * CLIENTS_PER_ROUND:
        raise RuntimeError(
            f"Expected {TOTAL_ROUNDS * CLIENTS_PER_ROUND} local trainings, "
            f"found {len(selection_df)}."
        )

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
        raise RuntimeError("Non-finite evaluation metrics.")

    if not np.all(
        selection_df.groupby("round")["global_client_id"].nunique().to_numpy()
        == CLIENTS_PER_ROUND
    ):
        raise RuntimeError("Duplicate client selection within a round.")

    final_metrics_row = learning_df[
        learning_df["round"].astype(int) == TOTAL_ROUNDS
    ]
    if len(final_metrics_row) != 1:
        raise RuntimeError("Missing round-30 evaluation.")

    final_metrics = final_metrics_row.iloc[0]

    torch.save(
        {
            "model_state_dict": latest["global_state"],
            "completed_round": TOTAL_ROUNDS,
            "outer_fold": OUTER_FOLD,
            "alpha": ALPHA,
            "scenario": SCENARIO,
            "method": METHOD,
            "fl_seed": FL_SEED,
            "final_metrics": final_metrics.to_dict(),
        },
        output_root / "final_global_model_round30.pt",
    )

    report_lines = [
        "PAMAP2 FEDAVG CLEAN 30-ROUND TECHNICAL VALIDATION",
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
        f"Method: {METHOD}",
        f"FL seed: {FL_SEED}",
        "",
        "EXECUTION",
        "-" * 78,
        f"Rounds completed: {TOTAL_ROUNDS}",
        f"Clients per fold: {CLIENTS_PER_FOLD}",
        f"Clients selected per round: {CLIENTS_PER_ROUND}",
        f"Total local trainings: {TOTAL_ROUNDS * CLIENTS_PER_ROUND}",
        "Checkpoint written after every completed round.",
        "Execution is resumable from latest_checkpoint.pt.",
        "",
        "DATA AND LEAKAGE CONTROLS",
        "-" * 78,
        f"Retained training windows: {len(retained_rows)}",
        f"Outer-test windows: {len(y_test)}",
        "Z-score fitted only on retained outer-training client windows.",
        "Outer test did not contribute to training, selection, or normalization.",
        "",
        "LEARNING",
        "-" * 78,
    ]

    for row in learning_df.itertuples(index=False):
        report_lines.append(
            f"Round {int(row.round):02d}: "
            f"Macro-F1={row.test_macro_f1:.4f}; "
            f"BalAcc={row.test_balanced_accuracy:.4f}; "
            f"Acc={row.test_accuracy:.4f}; "
            f"active={int(row.active_clients)}; "
            f"Emean={row.mean_residual_energy:.4f}; "
            f"Emin={row.min_residual_energy:.4f}"
        )

    report_lines.extend(
        [
            "",
            "FINAL STATE",
            "-" * 78,
            f"Round 30 Macro-F1: {float(final_metrics['test_macro_f1']):.4f}",
            (
                "Round 30 balanced accuracy: "
                f"{float(final_metrics['test_balanced_accuracy']):.4f}"
            ),
            f"Round 30 accuracy: {float(final_metrics['test_accuracy']):.4f}",
            (
                "Final active clients: "
                f"{int(progress_df.iloc[-1]['active_clients'])}/"
                f"{CLIENTS_PER_FOLD}"
            ),
            (
                "Final mean residual energy: "
                f"{float(progress_df.iloc[-1]['mean_residual_energy']):.4f}"
            ),
            (
                "Final minimum residual energy: "
                f"{float(progress_df.iloc[-1]['min_residual_energy']):.4f}"
            ),
            "",
            "FILES",
            "-" * 78,
            "VALIDATION_CONFIG.json",
            "VALIDATION_REPORT.txt",
            "learning_curve.csv",
            "round_progress.csv",
            "selected_client_log.csv",
            "latest_checkpoint.pt",
            "final_global_model_round30.pt",
            "normalization_mean.npy",
            "normalization_std.npy",
        ]
    )

    report_path = output_root / "VALIDATION_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== 30-round FedAvg validation completed successfully ===")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nValidation interrupted. Completed rounds are preserved; "
            "rerun the same command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
