from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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


# =============================================================================
# Frozen screening design
# =============================================================================

NUM_CLASSES = 12
INPUT_CHANNELS = 27
WINDOW_SAMPLES = 256

SEEDS = (123, 456, 789)

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

LOSS_VARIANTS = (
    "cross_entropy",
    "sqrt_inverse_frequency",
)

DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MIN_DELTA = 1e-4
LIGHTWEIGHT_TOLERANCE = 0.005

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


# =============================================================================
# Reproducibility utilities
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


# =============================================================================
# Data loading
# =============================================================================

@dataclass(frozen=True)
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    class_counts: list[int]
    core27_indices: list[int]


def load_core27_data(processed_root: Path) -> DataBundle:
    split_dir = processed_root / "splits"
    stats_dir = processed_root / "statistics"

    required = [
        split_dir / "train_X_full36.npy",
        split_dir / "train_y.npy",
        split_dir / "validation_X_full36.npy",
        split_dir / "validation_y.npy",
        stats_dir / "core27_indices.npy",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    core27_indices = np.load(stats_dir / "core27_indices.npy").astype(np.int64)
    if core27_indices.shape != (27,):
        raise RuntimeError(f"Unexpected Core27 mask shape: {core27_indices.shape}")

    print("Loading train and validation Core27 arrays into RAM...")

    train_full = np.load(split_dir / "train_X_full36.npy", mmap_mode="r")
    val_full = np.load(split_dir / "validation_X_full36.npy", mmap_mode="r")

    # Create contiguous N,C,L arrays for Conv1d.
    x_train_np = np.ascontiguousarray(
        train_full[:, :, core27_indices].transpose(0, 2, 1),
        dtype=np.float32,
    )
    x_val_np = np.ascontiguousarray(
        val_full[:, :, core27_indices].transpose(0, 2, 1),
        dtype=np.float32,
    )

    y_train_np = np.asarray(
        np.load(split_dir / "train_y.npy", mmap_mode="r"),
        dtype=np.int64,
    ).copy()
    y_val_np = np.asarray(
        np.load(split_dir / "validation_y.npy", mmap_mode="r"),
        dtype=np.int64,
    ).copy()

    if x_train_np.shape != (11014, 27, 256):
        raise RuntimeError(f"Unexpected train shape: {x_train_np.shape}")
    if x_val_np.shape != (1932, 27, 256):
        raise RuntimeError(f"Unexpected validation shape: {x_val_np.shape}")
    if set(np.unique(y_train_np).tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError("Training set does not contain all 12 classes.")
    if set(np.unique(y_val_np).tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError("Validation set does not contain all 12 classes.")
    if not np.isfinite(x_train_np).all() or not np.isfinite(x_val_np).all():
        raise RuntimeError("Non-finite features found in training or validation arrays.")

    class_counts = np.bincount(y_train_np, minlength=NUM_CLASSES).astype(int).tolist()

    print(f"[OK] Train Core27:      {x_train_np.shape}")
    print(f"[OK] Validation Core27: {x_val_np.shape}")
    print(f"[OK] Train class counts: {class_counts}")
    print()

    return DataBundle(
        x_train=torch.from_numpy(x_train_np),
        y_train=torch.from_numpy(y_train_np),
        x_validation=torch.from_numpy(x_val_np),
        y_validation=torch.from_numpy(y_val_np),
        class_counts=class_counts,
        core27_indices=core27_indices.astype(int).tolist(),
    )


def make_loaders(
    data: DataBundle,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(data.x_validation, data.y_validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, validation_loader


# =============================================================================
# Model
# =============================================================================

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
    if variant not in MODEL_VARIANTS:
        raise KeyError(variant)

    config = MODEL_VARIANTS[variant]
    return LightweightCNN1D(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        channels=tuple(config["channels"]),
        dropout=float(config["dropout"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimated_fp32_size_mib(model: nn.Module) -> float:
    return count_trainable_parameters(model) * 4 / (1024 ** 2)


# =============================================================================
# Loss and metrics
# =============================================================================

def build_criterion(
    loss_variant: str,
    class_counts: list[int],
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    if loss_variant == "cross_entropy":
        return nn.CrossEntropyLoss(), None

    if loss_variant == "sqrt_inverse_frequency":
        counts = np.asarray(class_counts, dtype=np.float64)
        weights = np.sqrt(counts.max() / counts)
        weights = weights / weights.mean()
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights_tensor), weights.tolist()

    raise KeyError(loss_variant)


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
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


# =============================================================================
# Training/evaluation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()

    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

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

        predictions = logits.detach().argmax(dim=1)
        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        loss=total_loss / max(total_examples, 1),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Metrics, np.ndarray, np.ndarray]:
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

        predictions = logits.argmax(dim=1)
        all_true.append(y_batch.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        loss=total_loss / max(total_examples, 1),
    )
    return metrics, y_true, y_pred


# =============================================================================
# Experiment bookkeeping
# =============================================================================

@dataclass
class RunResult:
    model_variant: str
    loss_variant: str
    seed: int
    parameter_count: int
    fp32_model_size_mib: float
    best_epoch: int
    epochs_ran: int
    best_validation_macro_f1: float
    best_validation_accuracy: float
    best_validation_balanced_accuracy: float
    best_validation_loss: float
    wall_seconds: float
    device: str
    checkpoint_path: str


def save_epoch_history(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    table = pd.DataFrame(
        matrix,
        index=[f"true_{index}_{ACTIVITY_NAMES[index]}" for index in range(NUM_CLASSES)],
        columns=[f"pred_{index}_{ACTIVITY_NAMES[index]}" for index in range(NUM_CLASSES)],
    )
    table.to_csv(path)


def train_run(
    *,
    data: DataBundle,
    model_variant: str,
    loss_variant: str,
    seed: int,
    device: torch.device,
    run_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    min_delta: float,
) -> RunResult:
    set_seed(seed)

    model = build_model(model_variant).to(device)
    parameter_count = count_trainable_parameters(model)
    size_mib = estimated_fp32_size_mib(model)

    criterion, class_weights = build_criterion(
        loss_variant,
        data.class_counts,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader, validation_loader = make_loaders(
        data=data,
        batch_size=batch_size,
        seed=seed,
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best_model.pt"

    run_config = {
        "model_variant": model_variant,
        "model_config": MODEL_VARIANTS[model_variant],
        "loss_variant": loss_variant,
        "class_weights": class_weights,
        "seed": seed,
        "device": str(device),
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "min_delta": min_delta,
        "parameter_count": parameter_count,
        "fp32_model_size_mib": size_mib,
        "test_set_loaded": False,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    best_macro_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_metrics: Metrics | None = None
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None
    history: list[dict[str, object]] = []

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics, y_true, y_pred = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_balanced_accuracy": train_metrics.balanced_accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_balanced_accuracy": validation_metrics.balanced_accuracy,
                "validation_macro_f1": validation_metrics.macro_f1,
            }
        )

        improved = (
            validation_metrics.macro_f1
            > best_macro_f1 + min_delta
        )

        if improved:
            best_macro_f1 = validation_metrics.macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metrics = validation_metrics
            best_y_true = y_true.copy()
            best_y_pred = y_pred.copy()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_variant": model_variant,
                    "loss_variant": loss_variant,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_metrics": asdict(best_metrics),
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"    epoch={epoch:02d} "
            f"train_f1={train_metrics.macro_f1:.4f} "
            f"val_f1={validation_metrics.macro_f1:.4f} "
            f"val_bal_acc={validation_metrics.balanced_accuracy:.4f} "
            f"best={best_macro_f1:.4f} "
            f"no_improve={epochs_without_improvement}/{patience}"
        )

        save_epoch_history(run_dir / "epoch_history.csv", history)

        if epochs_without_improvement >= patience:
            print(f"    early stopping at epoch {epoch}")
            break

    wall_seconds = time.perf_counter() - start_time

    if best_metrics is None or best_y_true is None or best_y_pred is None:
        raise RuntimeError("No best validation state was recorded.")

    save_confusion_matrix(
        run_dir / "best_validation_confusion_matrix.csv",
        best_y_true,
        best_y_pred,
    )

    per_class_f1 = f1_score(
        best_y_true,
        best_y_pred,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    pd.DataFrame(
        {
            "class_index": list(range(NUM_CLASSES)),
            "activity_name": ACTIVITY_NAMES,
            "validation_f1": per_class_f1,
        }
    ).to_csv(run_dir / "best_validation_per_class_f1.csv", index=False)

    result = RunResult(
        model_variant=model_variant,
        loss_variant=loss_variant,
        seed=seed,
        parameter_count=parameter_count,
        fp32_model_size_mib=size_mib,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_macro_f1=best_metrics.macro_f1,
        best_validation_accuracy=best_metrics.accuracy,
        best_validation_balanced_accuracy=best_metrics.balanced_accuracy,
        best_validation_loss=best_metrics.loss,
        wall_seconds=wall_seconds,
        device=str(device),
        checkpoint_path=str(checkpoint_path),
    )

    (run_dir / "run_result.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )

    return result


# =============================================================================
# Screening summary and selection
# =============================================================================

def summarize_results(
    results: list[RunResult],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    runs_df = pd.DataFrame([asdict(result) for result in results])
    runs_df.to_csv(output_root / "screening_runs.csv", index=False)

    grouped = (
        runs_df.groupby(
            ["model_variant", "loss_variant", "parameter_count", "fp32_model_size_mib"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "count"),
            mean_validation_macro_f1=("best_validation_macro_f1", "mean"),
            std_validation_macro_f1=("best_validation_macro_f1", "std"),
            min_validation_macro_f1=("best_validation_macro_f1", "min"),
            max_validation_macro_f1=("best_validation_macro_f1", "max"),
            mean_validation_balanced_accuracy=(
                "best_validation_balanced_accuracy", "mean"
            ),
            mean_validation_accuracy=("best_validation_accuracy", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_wall_seconds=("wall_seconds", "mean"),
        )
    )

    grouped["std_validation_macro_f1"] = grouped["std_validation_macro_f1"].fillna(0.0)
    grouped = grouped.sort_values(
        [
            "mean_validation_macro_f1",
            "parameter_count",
            "std_validation_macro_f1",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    grouped["rank_by_mean_f1"] = np.arange(1, len(grouped) + 1)

    grouped.to_csv(output_root / "screening_config_summary.csv", index=False)

    best_mean = float(grouped["mean_validation_macro_f1"].max())
    eligible = grouped[
        grouped["mean_validation_macro_f1"]
        >= best_mean - LIGHTWEIGHT_TOLERANCE
    ].copy()

    selected = eligible.sort_values(
        [
            "parameter_count",
            "std_validation_macro_f1",
            "mean_validation_macro_f1",
        ],
        ascending=[True, True, False],
    ).iloc[0]

    selection = {
        "selection_rule": (
            "Primary: mean validation Macro-F1 across seeds. "
            f"Among configurations within {LIGHTWEIGHT_TOLERANCE:.3f} absolute "
            "Macro-F1 of the best mean, select the model with the fewest "
            "trainable parameters; then lower across-seed standard deviation."
        ),
        "best_mean_validation_macro_f1": best_mean,
        "lightweight_tolerance": LIGHTWEIGHT_TOLERANCE,
        "selected_model_variant": str(selected["model_variant"]),
        "selected_loss_variant": str(selected["loss_variant"]),
        "selected_parameter_count": int(selected["parameter_count"]),
        "selected_fp32_model_size_mib": float(selected["fp32_model_size_mib"]),
        "selected_mean_validation_macro_f1": float(
            selected["mean_validation_macro_f1"]
        ),
        "selected_std_validation_macro_f1": float(
            selected["std_validation_macro_f1"]
        ),
        "test_set_used_for_selection": False,
    }

    (output_root / "selected_configuration.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    return grouped, selection


def write_report(
    *,
    output_root: Path,
    mode: str,
    device: torch.device,
    processed_root: Path,
    grouped: pd.DataFrame | None,
    selection: dict[str, object] | None,
    results: list[RunResult],
) -> Path:
    report_path = output_root / "SCREENING_REPORT.txt"

    lines = [
        "PAMAP2 CENTRALIZED LIGHTWEIGHT 1D-CNN SCREENING REPORT",
        "=" * 78,
        f"Mode: {mode}",
        f"Processed dataset: {processed_root}",
        f"Device: {device}",
        f"PyTorch: {torch.__version__}",
        "",
        "DATA USAGE",
        "-" * 78,
        "Training set: subject102-subject107",
        "Validation set: subject101",
        "Test set: NOT LOADED and NOT USED",
        "",
        "INPUT",
        "-" * 78,
        "Core27 channels",
        "Window length: 256 samples",
        "Number of classes: 12",
        "",
        "CANDIDATES",
        "-" * 78,
        f"Models: {', '.join(MODEL_VARIANTS)}",
        f"Losses: {', '.join(LOSS_VARIANTS)}",
        f"Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        "",
        "OPTIMIZATION",
        "-" * 78,
        f"Optimizer: AdamW",
        f"Learning rate: {DEFAULT_LR}",
        f"Weight decay: {DEFAULT_WEIGHT_DECAY}",
        f"Batch size: {DEFAULT_BATCH_SIZE}",
        f"Maximum epochs: {DEFAULT_MAX_EPOCHS}",
        f"Early-stopping patience: {DEFAULT_PATIENCE}",
        "Selection metric: validation Macro-F1",
        "",
        "RUNS COMPLETED",
        "-" * 78,
        f"{len(results)}",
    ]

    if grouped is not None:
        lines.extend(
            [
                "",
                "CONFIGURATION SUMMARY",
                "-" * 78,
            ]
        )
        for row in grouped.itertuples(index=False):
            lines.append(
                f"{row.model_variant} + {row.loss_variant}: "
                f"mean val Macro-F1={row.mean_validation_macro_f1:.4f}, "
                f"std={row.std_validation_macro_f1:.4f}, "
                f"params={int(row.parameter_count):,}"
            )

    if selection is not None:
        lines.extend(
            [
                "",
                "SELECTED CONFIGURATION",
                "-" * 78,
                f"Model: {selection['selected_model_variant']}",
                f"Loss: {selection['selected_loss_variant']}",
                f"Parameters: {selection['selected_parameter_count']:,}",
                f"Mean validation Macro-F1: "
                f"{selection['selected_mean_validation_macro_f1']:.4f}",
                f"Std validation Macro-F1: "
                f"{selection['selected_std_validation_macro_f1']:.4f}",
                "",
                "IMPORTANT",
                "-" * 78,
                "The subject108 test set was not loaded or used for architecture,",
                "loss, hyperparameter, or seed selection.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only centralized screening of lightweight 1D-CNN "
            "configurations on the frozen PAMAP2 Core27 dataset."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path.cwd()
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128",
        help="Path to the frozen processed PAMAP2 dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default depends on --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "screen"),
        required=True,
        help="smoke: 1 config x 1 seed x 2 epochs; screen: full 18-run campaign.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser().resolve()
    if not processed_root.is_dir():
        raise SystemExit(f"ERROR: processed dataset not found: {processed_root}")

    # Expected layout:
    # <project>/data/processed/pamap2/protocol_v1_w256_s128
    try:
        project_root = processed_root.parents[3]
    except IndexError:
        project_root = Path.cwd()

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / ("smoke_v1" if args.mode == "smoke" else "screening_v1")
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}\n"
            "Remove it explicitly before rerunning to prevent accidental overwrite."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device()

    print("=== PAMAP2 centralized lightweight 1D-CNN screening ===")
    print(f"Mode:              {args.mode}")
    print(f"Processed dataset: {processed_root}")
    print(f"Output:            {output_root}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")
    print()

    if device.type == "xpu":
        print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    else:
        print("WARNING: XPU is not available; CPU will be used.")
    print()

    # Record provenance before any model training.
    provenance = {
        "mode": args.mode,
        "processed_root": str(processed_root),
        "preprocessing_configuration_sha256": sha256_file(
            processed_root / "preprocessing_configuration.json"
        ),
        "processed_output_inventory_sha256": sha256_file(
            processed_root / "output_inventory_sha256.json"
        ),
        "torch_version": torch.__version__,
        "device": str(device),
        "xpu_available": bool(
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ),
        "model_variants": MODEL_VARIANTS,
        "loss_variants": list(LOSS_VARIANTS),
        "full_screening_seeds": list(SEEDS),
        "test_set_loaded": False,
    }
    (output_root / "screening_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    data = load_core27_data(processed_root)

    if args.mode == "smoke":
        experiments = [("tiny", "cross_entropy", SEEDS[0])]
        max_epochs = 2
        patience = 2
    else:
        experiments = [
            (model_variant, loss_variant, seed)
            for model_variant in MODEL_VARIANTS
            for loss_variant in LOSS_VARIANTS
            for seed in SEEDS
        ]
        max_epochs = args.max_epochs
        patience = args.patience

    print(f"Experiments to run: {len(experiments)}")
    print()

    results: list[RunResult] = []

    for index, (model_variant, loss_variant, seed) in enumerate(experiments, start=1):
        run_name = f"{model_variant}__{loss_variant}__seed{seed}"
        run_dir = output_root / "runs" / run_name

        print("=" * 78)
        print(f"[{index}/{len(experiments)}] {run_name}")
        print("=" * 78)

        result = train_run(
            data=data,
            model_variant=model_variant,
            loss_variant=loss_variant,
            seed=seed,
            device=device,
            run_dir=run_dir,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            min_delta=args.min_delta,
        )
        results.append(result)

        print(
            f"[OK] best_epoch={result.best_epoch}; "
            f"best_val_macro_f1={result.best_validation_macro_f1:.4f}; "
            f"params={result.parameter_count:,}; "
            f"wall={result.wall_seconds:.1f}s"
        )
        print()

    grouped: pd.DataFrame | None = None
    selection: dict[str, object] | None = None

    if args.mode == "screen":
        grouped, selection = summarize_results(results, output_root)

    report_path = write_report(
        output_root=output_root,
        mode=args.mode,
        device=device,
        processed_root=processed_root,
        grouped=grouped,
        selection=selection,
        results=results,
    )

    print("=== Screening workflow completed successfully ===")
    print(f"Report: {report_path}")

    if selection is not None:
        print()
        print("Selected configuration:")
        print(
            f"  model={selection['selected_model_variant']}; "
            f"loss={selection['selected_loss_variant']}; "
            f"mean_val_macro_f1={selection['selected_mean_validation_macro_f1']:.4f}"
        )

    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nScreening interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
