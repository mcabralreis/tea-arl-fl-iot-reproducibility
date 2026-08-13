from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
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


# =============================================================================
# Frozen final-evaluation design
# =============================================================================

NUM_CLASSES = 12
INPUT_CHANNELS = 27
WINDOW_SAMPLES = 256

FINAL_SEEDS = (123, 456, 789, 1011, 1213)

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

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

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
        raise KeyError(f"Unknown model variant: {variant}")

    config = MODEL_VARIANTS[variant]
    return LightweightCNN1D(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        channels=tuple(config["channels"]),
        dropout=float(config["dropout"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =============================================================================
# Metrics
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
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    )


# =============================================================================
# Data
# =============================================================================

@dataclass(frozen=True)
class DataBundle:
    x_development: torch.Tensor
    y_development: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    core27_indices: list[int]


def load_core27_split(
    processed_root: Path,
    split: str,
    core27_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_dir = processed_root / "splits"

    x_path = split_dir / f"{split}_X_full36.npy"
    y_path = split_dir / f"{split}_y.npy"

    if not x_path.is_file():
        raise FileNotFoundError(x_path)
    if not y_path.is_file():
        raise FileNotFoundError(y_path)

    full = np.load(x_path, mmap_mode="r")
    x_np = np.ascontiguousarray(
        full[:, :, core27_indices].transpose(0, 2, 1),
        dtype=np.float32,
    )
    y_np = np.asarray(
        np.load(y_path, mmap_mode="r"),
        dtype=np.int64,
    ).copy()

    if x_np.ndim != 3 or x_np.shape[1:] != (27, 256):
        raise RuntimeError(f"Unexpected {split} X shape: {x_np.shape}")
    if y_np.shape != (x_np.shape[0],):
        raise RuntimeError(f"Unexpected {split} y shape: {y_np.shape}")
    if not np.isfinite(x_np).all():
        raise RuntimeError(f"Non-finite values found in {split} features.")
    if set(np.unique(y_np).tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError(f"{split} does not contain all 12 classes.")

    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def load_final_data(
    processed_root: Path,
    protocol_path: Path,
) -> DataBundle:
    stats_dir = processed_root / "statistics"
    core27_indices = np.load(
        stats_dir / "core27_indices.npy"
    ).astype(np.int64)

    if core27_indices.shape != (27,):
        raise RuntimeError(
            f"Unexpected Core27 mask shape: {core27_indices.shape}"
        )

    print("Loading development data (train + validation)...")
    x_train, y_train = load_core27_split(
        processed_root, "train", core27_indices
    )
    x_validation, y_validation = load_core27_split(
        processed_root, "validation", core27_indices
    )

    x_development = torch.cat((x_train, x_validation), dim=0)
    y_development = torch.cat((y_train, y_validation), dim=0)

    print(f"[OK] Development Core27: {tuple(x_development.shape)}")
    print(
        "[OK] Development class counts: "
        f"{torch.bincount(y_development, minlength=NUM_CLASSES).tolist()}"
    )

    # The protocol file must already exist before the test set is loaded.
    if not protocol_path.is_file():
        raise RuntimeError(
            "Final protocol file does not exist before test loading."
        )

    print()
    print("Final protocol is frozen on disk.")
    print(f"Protocol: {protocol_path}")
    print("Loading the untouched test set for final reporting only...")

    x_test, y_test = load_core27_split(
        processed_root, "test", core27_indices
    )

    print(f"[OK] Test Core27:        {tuple(x_test.shape)}")
    print(
        "[OK] Test class counts: "
        f"{torch.bincount(y_test, minlength=NUM_CLASSES).tolist()}"
    )
    print()

    return DataBundle(
        x_development=x_development,
        y_development=y_development,
        x_test=x_test,
        y_test=y_test,
        core27_indices=core27_indices.astype(int).tolist(),
    )


def make_loaders(
    data: DataBundle,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    development_loader = DataLoader(
        TensorDataset(data.x_development, data.y_development),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )

    test_loader = DataLoader(
        TensorDataset(data.x_test, data.y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return development_loader, test_loader


# =============================================================================
# Training and evaluation
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
        y_true,
        y_pred,
        total_loss / max(total_examples, 1),
    )


@torch.inference_mode()
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Metrics, np.ndarray, np.ndarray, float]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    synchronize(device)
    start = time.perf_counter()

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

    synchronize(device)
    inference_seconds = time.perf_counter() - start

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = compute_metrics(
        y_true,
        y_pred,
        total_loss / max(total_examples, 1),
    )
    return metrics, y_true, y_pred, inference_seconds


# =============================================================================
# Reporting helpers
# =============================================================================

@dataclass
class FinalRunResult:
    seed: int
    fixed_epochs: int
    parameter_count: int
    development_windows: int
    test_windows: int
    test_loss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_macro_f1: float
    training_wall_seconds: float
    test_inference_seconds: float
    test_inference_ms_per_window: float
    checkpoint_path: str


def save_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    table = pd.DataFrame(
        matrix,
        index=[
            f"true_{index}_{ACTIVITY_NAMES[index]}"
            for index in range(NUM_CLASSES)
        ],
        columns=[
            f"pred_{index}_{ACTIVITY_NAMES[index]}"
            for index in range(NUM_CLASSES)
        ],
    )
    table.to_csv(path)
    return matrix


def aggregate_results(
    results: list[FinalRunResult],
    per_class_rows: list[dict[str, object]],
    confusion_matrices: list[np.ndarray],
    output_root: Path,
) -> dict[str, object]:
    runs_df = pd.DataFrame([asdict(result) for result in results])
    runs_df.to_csv(output_root / "final_test_runs.csv", index=False)

    summary: dict[str, object] = {
        "runs": len(results),
        "seeds": [result.seed for result in results],
        "fixed_epochs": results[0].fixed_epochs,
        "parameter_count": results[0].parameter_count,
    }

    for metric in (
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "training_wall_seconds",
        "test_inference_seconds",
        "test_inference_ms_per_window",
    ):
        values = runs_df[metric].to_numpy(dtype=float)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=1))

    (output_root / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(
        output_root / "final_test_per_class_f1_runs.csv",
        index=False,
    )

    per_class_summary = (
        per_class_df.groupby(
            ["class_index", "activity_name"],
            as_index=False,
        )
        .agg(
            mean_test_f1=("test_f1", "mean"),
            std_test_f1=("test_f1", "std"),
            min_test_f1=("test_f1", "min"),
            max_test_f1=("test_f1", "max"),
        )
        .sort_values("class_index")
    )
    per_class_summary.to_csv(
        output_root / "final_test_per_class_f1_summary.csv",
        index=False,
    )

    aggregate_confusion = np.sum(
        np.stack(confusion_matrices, axis=0),
        axis=0,
    )
    pd.DataFrame(
        aggregate_confusion,
        index=[
            f"true_{index}_{ACTIVITY_NAMES[index]}"
            for index in range(NUM_CLASSES)
        ],
        columns=[
            f"pred_{index}_{ACTIVITY_NAMES[index]}"
            for index in range(NUM_CLASSES)
        ],
    ).to_csv(output_root / "aggregate_test_confusion_matrix.csv")

    return summary


def write_report(
    output_root: Path,
    processed_root: Path,
    screening_root: Path,
    selected_model: str,
    selected_loss: str,
    screening_best_epochs: list[int],
    fixed_epochs: int,
    summary: dict[str, object],
    development_windows: int,
    test_windows: int,
    device: torch.device,
) -> Path:
    report_path = output_root / "FINAL_TEST_REPORT.txt"

    lines = [
        "PAMAP2 CENTRALIZED FINAL TEST REPORT",
        "=" * 78,
        f"Processed dataset: {processed_root}",
        f"Screening source: {screening_root}",
        f"Device: {device}",
        f"PyTorch: {torch.__version__}",
        "",
        "FROZEN CONFIGURATION",
        "-" * 78,
        f"Model: {selected_model}",
        f"Loss: {selected_loss}",
        f"Parameters: {summary['parameter_count']:,}",
        "Input: Core27",
        "Window: 256 samples",
        "Optimizer: AdamW",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Batch size: {BATCH_SIZE}",
        "",
        "FIXED TRAINING DURATION",
        "-" * 78,
        f"Screening best epochs: {screening_best_epochs}",
        f"Rule: median of the three selected-configuration best epochs",
        f"Final fixed epochs: {fixed_epochs}",
        "",
        "FINAL DATA USAGE",
        "-" * 78,
        f"Development training set: original train + validation ({development_windows:,} windows)",
        f"Final test set: subject108 only ({test_windows:,} windows)",
        "The frozen preprocessing normalization was retained unchanged.",
        "The final test set was loaded only after the evaluation protocol was",
        "written to disk. No test result was used for model, loss, epoch,",
        "hyperparameter, or seed selection.",
        "",
        "FINAL SEEDS",
        "-" * 78,
        ", ".join(str(seed) for seed in FINAL_SEEDS),
        "",
        "FINAL TEST RESULTS (MEAN +/- SAMPLE STD)",
        "-" * 78,
        f"Macro-F1: "
        f"{summary['mean_test_macro_f1']:.4f} +/- "
        f"{summary['std_test_macro_f1']:.4f}",
        f"Balanced accuracy: "
        f"{summary['mean_test_balanced_accuracy']:.4f} +/- "
        f"{summary['std_test_balanced_accuracy']:.4f}",
        f"Accuracy: "
        f"{summary['mean_test_accuracy']:.4f} +/- "
        f"{summary['std_test_accuracy']:.4f}",
        f"Test loss: "
        f"{summary['mean_test_loss']:.4f} +/- "
        f"{summary['std_test_loss']:.4f}",
        "",
        "COMPUTATIONAL DIAGNOSTICS",
        "-" * 78,
        f"Mean training wall time per seed: "
        f"{summary['mean_training_wall_seconds']:.2f} s",
        f"Mean test inference time: "
        f"{summary['mean_test_inference_seconds']:.4f} s",
        f"Mean inference time per window: "
        f"{summary['mean_test_inference_ms_per_window']:.4f} ms",
        "",
        "FILES",
        "-" * 78,
        "FINAL_EVALUATION_PROTOCOL.json",
        "final_test_runs.csv",
        "final_test_summary.json",
        "final_test_per_class_f1_runs.csv",
        "final_test_per_class_f1_summary.csv",
        "aggregate_test_confusion_matrix.csv",
        "runs/seed*/...",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final, test-only reporting protocol for the frozen PAMAP2 "
            "centralized lightweight 1D-CNN."
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
        "--screening-root",
        type=Path,
        default=None,
        help="Default: <project>/outputs/centralized/pamap2/screening_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/outputs/centralized/pamap2/final_test_v1",
    )
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser().resolve()
    if not processed_root.is_dir():
        raise SystemExit(
            f"ERROR: processed dataset not found: {processed_root}"
        )

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

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / "final_test_v1"
    )

    if not screening_root.is_dir():
        raise SystemExit(
            f"ERROR: screening directory not found: {screening_root}"
        )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}\n"
            "Remove it explicitly before rerunning to prevent accidental overwrite."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_path = screening_root / "selected_configuration.json"
    runs_path = screening_root / "screening_runs.csv"

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    screening_runs = pd.read_csv(runs_path)

    selected_model = str(selected["selected_model_variant"])
    selected_loss = str(selected["selected_loss_variant"])

    if selected_loss != "cross_entropy":
        raise RuntimeError(
            "This final protocol currently expects the selected loss to be "
            f"cross_entropy, found: {selected_loss}"
        )

    selected_rows = screening_runs[
        (screening_runs["model_variant"] == selected_model)
        & (screening_runs["loss_variant"] == selected_loss)
    ].copy()

    if len(selected_rows) != 3:
        raise RuntimeError(
            f"Expected 3 screening runs for selected configuration, found {len(selected_rows)}."
        )

    screening_best_epochs = sorted(
        int(value)
        for value in selected_rows["best_epoch"].tolist()
    )
    fixed_epochs = int(np.median(screening_best_epochs))
    if fixed_epochs < 1:
        raise RuntimeError(f"Invalid fixed epoch count: {fixed_epochs}")

    device = select_device()

    print("=== PAMAP2 centralized final evaluation ===")
    print(f"Processed dataset: {processed_root}")
    print(f"Screening source:  {screening_root}")
    print(f"Output:            {output_root}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")
    if device.type == "xpu":
        print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    print()
    print("Frozen selected configuration:")
    print(f"  Model:            {selected_model}")
    print(f"  Loss:             {selected_loss}")
    print(f"  Screening epochs: {screening_best_epochs}")
    print(f"  Fixed final epochs (median): {fixed_epochs}")
    print(f"  Final seeds:      {list(FINAL_SEEDS)}")
    print()

    # ------------------------------------------------------------------
    # Freeze and write the complete final protocol BEFORE test loading.
    # ------------------------------------------------------------------
    protocol = {
        "status": "FROZEN_BEFORE_TEST_LOADING",
        "selected_model_variant": selected_model,
        "selected_loss_variant": selected_loss,
        "selection_source": str(selected_path),
        "selection_source_sha256": sha256_file(selected_path),
        "screening_runs_source": str(runs_path),
        "screening_runs_source_sha256": sha256_file(runs_path),
        "screening_best_epochs_for_selected_configuration": screening_best_epochs,
        "final_epoch_rule": (
            "median of best epochs across the three validation-screening seeds"
        ),
        "fixed_final_epochs": fixed_epochs,
        "final_seeds": list(FINAL_SEEDS),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "final_training_data": "original train + original validation",
        "final_test_data": "subject108 only",
        "normalization": (
            "frozen preprocessing normalization retained unchanged"
        ),
        "test_based_selection_allowed": False,
        "processed_preprocessing_configuration_sha256": sha256_file(
            processed_root / "preprocessing_configuration.json"
        ),
        "processed_output_inventory_sha256": sha256_file(
            processed_root / "output_inventory_sha256.json"
        ),
        "torch_version": torch.__version__,
        "device": str(device),
    }

    protocol_path = output_root / "FINAL_EVALUATION_PROTOCOL.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print("[OK] Final protocol frozen on disk before test loading.")
    print(f"     {protocol_path}")
    print()

    data = load_final_data(processed_root, protocol_path)

    results: list[FinalRunResult] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_matrices: list[np.ndarray] = []

    for run_index, seed in enumerate(FINAL_SEEDS, start=1):
        print("=" * 78)
        print(f"[{run_index}/{len(FINAL_SEEDS)}] Final seed {seed}")
        print("=" * 78)

        set_seed(seed)

        run_dir = output_root / "runs" / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)

        model = build_model(selected_model).to(device)
        parameter_count = count_trainable_parameters(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        development_loader, test_loader = make_loaders(data, seed)

        history: list[dict[str, object]] = []
        train_start = time.perf_counter()

        for epoch in range(1, fixed_epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics = train_one_epoch(
                model,
                development_loader,
                criterion,
                optimizer,
                device,
            )

            synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start

            history.append(
                {
                    "epoch": epoch,
                    "epoch_seconds": epoch_seconds,
                    "development_loss": train_metrics.loss,
                    "development_accuracy": train_metrics.accuracy,
                    "development_balanced_accuracy": train_metrics.balanced_accuracy,
                    "development_macro_f1": train_metrics.macro_f1,
                }
            )

            print(
                f"    epoch={epoch:02d}/{fixed_epochs:02d} "
                f"dev_loss={train_metrics.loss:.4f} "
                f"dev_f1={train_metrics.macro_f1:.4f} "
                f"dev_bal_acc={train_metrics.balanced_accuracy:.4f}"
            )

        synchronize(device)
        training_wall_seconds = time.perf_counter() - train_start

        pd.DataFrame(history).to_csv(
            run_dir / "development_epoch_history.csv",
            index=False,
        )

        # Exactly one test evaluation for this trained seed.
        test_metrics, y_true, y_pred, inference_seconds = evaluate_test(
            model,
            test_loader,
            criterion,
            device,
        )

        matrix = save_confusion_matrix(
            run_dir / "test_confusion_matrix.csv",
            y_true,
            y_pred,
        )
        confusion_matrices.append(matrix)

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )

        pd.DataFrame(
            {
                "class_index": list(range(NUM_CLASSES)),
                "activity_name": ACTIVITY_NAMES,
                "test_f1": per_class_f1,
            }
        ).to_csv(
            run_dir / "test_per_class_f1.csv",
            index=False,
        )

        for class_index, value in enumerate(per_class_f1):
            per_class_rows.append(
                {
                    "seed": seed,
                    "class_index": class_index,
                    "activity_name": ACTIVITY_NAMES[class_index],
                    "test_f1": float(value),
                }
            )

        checkpoint_path = run_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_variant": selected_model,
                "loss_variant": selected_loss,
                "seed": seed,
                "fixed_epochs": fixed_epochs,
                "test_metrics": asdict(test_metrics),
                "parameter_count": parameter_count,
            },
            checkpoint_path,
        )

        result = FinalRunResult(
            seed=seed,
            fixed_epochs=fixed_epochs,
            parameter_count=parameter_count,
            development_windows=int(data.y_development.shape[0]),
            test_windows=int(data.y_test.shape[0]),
            test_loss=test_metrics.loss,
            test_accuracy=test_metrics.accuracy,
            test_balanced_accuracy=test_metrics.balanced_accuracy,
            test_macro_f1=test_metrics.macro_f1,
            training_wall_seconds=training_wall_seconds,
            test_inference_seconds=inference_seconds,
            test_inference_ms_per_window=(
                1000.0 * inference_seconds / int(data.y_test.shape[0])
            ),
            checkpoint_path=str(checkpoint_path),
        )
        results.append(result)

        (run_dir / "run_result.json").write_text(
            json.dumps(asdict(result), indent=2),
            encoding="utf-8",
        )

        print(
            f"[TEST] macro_f1={test_metrics.macro_f1:.4f}; "
            f"balanced_acc={test_metrics.balanced_accuracy:.4f}; "
            f"accuracy={test_metrics.accuracy:.4f}"
        )
        print()

    summary = aggregate_results(
        results,
        per_class_rows,
        confusion_matrices,
        output_root,
    )

    report_path = write_report(
        output_root=output_root,
        processed_root=processed_root,
        screening_root=screening_root,
        selected_model=selected_model,
        selected_loss=selected_loss,
        screening_best_epochs=screening_best_epochs,
        fixed_epochs=fixed_epochs,
        summary=summary,
        development_windows=int(data.y_development.shape[0]),
        test_windows=int(data.y_test.shape[0]),
        device=device,
    )

    print("=== Final PAMAP2 evaluation completed successfully ===")
    print(
        f"Test Macro-F1: "
        f"{summary['mean_test_macro_f1']:.4f} +/- "
        f"{summary['std_test_macro_f1']:.4f}"
    )
    print(
        f"Test balanced accuracy: "
        f"{summary['mean_test_balanced_accuracy']:.4f} +/- "
        f"{summary['std_test_balanced_accuracy']:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{summary['mean_test_accuracy']:.4f} +/- "
        f"{summary['std_test_accuracy']:.4f}"
    )
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nFinal evaluation interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
