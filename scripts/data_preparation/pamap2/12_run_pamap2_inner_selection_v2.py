from __future__ import annotations

import argparse, gc, hashlib, json, math, random, shutil, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 12
MODEL_CHANNELS = (32, 64, 96)
DROPOUT = 0.20
GROUPS = 8
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 8
MIN_DELTA = 1e-4
TOLERANCE = 0.005
INNER_SEEDS = (123, 456)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try: torch.xpu.manual_seed_all(seed)
        except Exception: pass


def device_now() -> torch.device:
    return torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")


def sync(device: torch.device) -> None:
    if device.type == "xpu": torch.xpu.synchronize()


def clear_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "xpu":
        try: torch.xpu.empty_cache()
        except Exception: pass


def subjects(text: str) -> tuple[int, ...]:
    return tuple(int(t.strip().replace("subject", "")) for t in str(text).split(",") if t.strip())


@dataclass(frozen=True)
class Dataset:
    x_raw: np.ndarray
    y: np.ndarray
    subject: np.ndarray


def load_all(processed_root: Path) -> Dataset:
    sd = processed_root / "splits"; st = processed_root / "statistics"
    mean = np.load(st / "training_mean_full36.npy").astype(np.float32)
    std = np.load(st / "training_std_full36.npy").astype(np.float32)
    xs, ys, ss = [], [], []
    for split in ("train", "validation", "test"):
        x = np.load(sd / f"{split}_X_full36.npy", mmap_mode="r")
        xs.append((np.asarray(x, dtype=np.float32) * std[None, None, :] + mean[None, None, :]).astype(np.float32))
        ys.append(np.asarray(np.load(sd / f"{split}_y.npy", mmap_mode="r"), dtype=np.int64).copy())
        ss.append(np.asarray(np.load(sd / f"{split}_subject_id.npy", mmap_mode="r"), dtype=np.int64).copy())
    x = np.concatenate(xs); y = np.concatenate(ys); s = np.concatenate(ss)
    if x.shape != (14972, 256, 36) or not np.isfinite(x).all():
        raise RuntimeError(f"Unexpected reconstructed data: {x.shape}")
    return Dataset(x, y, s)


def mag3(x: np.ndarray, start: int) -> np.ndarray:
    out = []
    for pos in range(3):
        a = x[:, :, start + 3 * pos:start + 3 * pos + 3]
        out.append(np.sqrt(np.sum(np.square(a, dtype=np.float32), axis=2))[:, :, None])
    return np.concatenate(out, axis=2)


def representation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "core27_axes":
        idx = np.r_[0:9, 18:36]
        return np.ascontiguousarray(x[:, :, idx], dtype=np.float32)
    a, g = mag3(x, 0), mag3(x, 18)
    if name == "magnitude6_acc16_gyro":
        return np.ascontiguousarray(np.concatenate((a, g), axis=2), dtype=np.float32)
    if name == "magnitude9_all":
        return np.ascontiguousarray(np.concatenate((a, g, mag3(x, 27)), axis=2), dtype=np.float32)
    raise KeyError(name)


def fold_normalize(x: np.ndarray, train_mask: np.ndarray, use_mask: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    train = x[train_mask]
    mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    if np.any(std <= 1e-8): raise RuntimeError("Near-zero fold-specific std")
    z = ((x[use_mask] - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    z = np.ascontiguousarray(z.transpose(0, 2, 1))
    if not np.isfinite(z).all(): raise RuntimeError("Non-finite normalized values")
    return torch.from_numpy(z), mean, std


def norm_layer(kind: str, channels: int) -> nn.Module:
    if kind == "batchnorm": return nn.BatchNorm1d(channels)
    if kind == "groupnorm": return nn.GroupNorm(GROUPS, channels)
    raise KeyError(kind)


class Block(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int, stride: int, norm: str):
        super().__init__(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            norm_layer(norm, cout), nn.ReLU(inplace=True)
        )


class Net(nn.Module):
    def __init__(self, cin: int, norm: str):
        super().__init__(); c1, c2, c3 = MODEL_CHANNELS
        self.f = nn.Sequential(
            Block(cin, c1, 7, 2, norm), Block(c1, c1, 5, 1, norm),
            Block(c1, c2, 5, 2, norm), Block(c2, c2, 3, 1, norm),
            Block(c2, c3, 3, 2, norm), Block(c3, c3, 3, 1, norm),
            nn.AdaptiveAvgPool1d(1)
        )
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(c3, NUM_CLASSES))
    def forward(self, x): return self.h(self.f(x))


def metrics(y: np.ndarray, p: np.ndarray, loss: float) -> dict[str, float]:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


def loaders(xtr, ytr, xv, yv, seed: int):
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
    va = DataLoader(TensorDataset(xv, yv), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr, va


def epoch_train(model, loader, criterion, optimizer, device):
    model.train(); loss_sum = 0.0; n = 0; ys = []; ps = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True); logits = model(xb); loss = criterion(logits, yb); loss.backward(); optimizer.step()
        b = len(yb); loss_sum += float(loss.detach().cpu()) * b; n += b
        ys.append(yb.detach().cpu().numpy()); ps.append(logits.detach().argmax(1).cpu().numpy())
    return metrics(np.concatenate(ys), np.concatenate(ps), loss_sum / n)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum = 0.0; n = 0; ys = []; ps = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device); logits = model(xb); loss = criterion(logits, yb)
        b = len(yb); loss_sum += float(loss.detach().cpu()) * b; n += b
        ys.append(yb.cpu().numpy()); ps.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return metrics(y, p, loss_sum / n)


@dataclass
class Result:
    outer_fold: int
    outer_test_subject: int
    inner_fold: int
    inner_validation_subject: int
    inner_training_subjects: str
    candidate_index: int
    representation: str
    input_channels: int
    normalization_layer: str
    seed: int
    parameter_count: int
    best_epoch: int
    epochs_ran: int
    best_validation_macro_f1: float
    best_validation_balanced_accuracy: float
    best_validation_accuracy: float
    best_validation_loss: float
    wall_seconds: float
    device: str
    run_directory: str


def train_run(xtr, ytr, xv, yv, meta: dict, device, run_dir: Path, max_epochs: int, patience: int, verbose: bool) -> Result:
    set_seed(meta["seed"]); model = Net(xtr.shape[1], meta["normalization_layer"]).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr, va = loaders(xtr, ytr, xv, yv, meta["seed"])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_config.json").write_text(json.dumps(meta | {"outer_test_subject_used": False}, indent=2), encoding="utf-8")
    best, best_epoch, noimp, bestm = -math.inf, 0, 0, None; history = []; start = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        t0 = time.perf_counter(); tm = epoch_train(model, tr, criterion, opt, device); vm = evaluate(model, va, criterion, device); sync(device)
        history.append({"epoch": ep, "epoch_seconds": time.perf_counter() - t0,
                        "train_loss": tm["loss"], "train_macro_f1": tm["macro_f1"],
                        "validation_loss": vm["loss"], "validation_accuracy": vm["accuracy"],
                        "validation_balanced_accuracy": vm["balanced_accuracy"], "validation_macro_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best + MIN_DELTA:
            best, best_epoch, noimp, bestm = vm["macro_f1"], ep, 0, vm
            torch.save({"model_state_dict": model.state_dict(), "meta": meta, "best_epoch": ep, "validation_metrics": vm}, run_dir / "best_model.pt")
        else: noimp += 1
        pd.DataFrame(history).to_csv(run_dir / "epoch_history.csv", index=False)
        if verbose: print(f"      epoch={ep:02d} train_f1={tm['macro_f1']:.4f} val_f1={vm['macro_f1']:.4f} best={best:.4f} no_improve={noimp}/{patience}")
        if noimp >= patience: break
    if bestm is None: raise RuntimeError("No best checkpoint")
    r = Result(meta["outer_fold"], meta["outer_test_subject"], meta["inner_fold"], meta["inner_validation_subject"],
               ",".join(map(str, meta["inner_training_subjects"])), meta["candidate_index"], meta["representation"],
               int(xtr.shape[1]), meta["normalization_layer"], meta["seed"], params, best_epoch, len(history),
               bestm["macro_f1"], bestm["balanced_accuracy"], bestm["accuracy"], bestm["loss"],
               time.perf_counter() - start, str(device), str(run_dir))
    (run_dir / "run_result.json").write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")
    return r


def load_results(runs_root: Path) -> list[Result]:
    out = []
    if runs_root.is_dir():
        for p in sorted(runs_root.glob("*/run_result.json")):
            out.append(Result(**json.loads(p.read_text(encoding="utf-8"))))
    return out


def summarize(results: list[Result], output_root: Path):
    runs = pd.DataFrame([asdict(r) for r in results]).sort_values(["outer_fold", "candidate_index", "inner_fold", "seed"])
    runs.to_csv(output_root / "inner_runs.csv", index=False)
    s = runs.groupby(["outer_fold", "outer_test_subject", "candidate_index", "representation", "input_channels", "normalization_layer", "parameter_count"], as_index=False).agg(
        runs=("seed", "count"), inner_validation_subjects=("inner_validation_subject", "nunique"), seeds=("seed", "nunique"),
        mean_validation_macro_f1=("best_validation_macro_f1", "mean"), std_validation_macro_f1=("best_validation_macro_f1", "std"),
        min_validation_macro_f1=("best_validation_macro_f1", "min"), max_validation_macro_f1=("best_validation_macro_f1", "max"),
        mean_validation_balanced_accuracy=("best_validation_balanced_accuracy", "mean"), median_best_epoch=("best_epoch", "median"))
    s["std_validation_macro_f1"] = s["std_validation_macro_f1"].fillna(0.0)
    s = s.sort_values(["outer_fold", "mean_validation_macro_f1", "input_channels", "std_validation_macro_f1"], ascending=[True, False, True, True])
    s.to_csv(output_root / "inner_candidate_summary.csv", index=False)
    selected = []
    for outer in sorted(s.outer_fold.unique()):
        fs = s[s.outer_fold == outer].copy(); best = float(fs.mean_validation_macro_f1.max())
        eligible = fs[fs.mean_validation_macro_f1 >= best - TOLERANCE]
        row = eligible.sort_values(["input_channels", "std_validation_macro_f1", "mean_validation_macro_f1"], ascending=[True, True, False]).iloc[0]
        rr = runs[(runs.outer_fold == outer) & (runs.candidate_index == int(row.candidate_index))]
        epochs = sorted(map(int, rr.best_epoch.tolist()))
        if len(epochs) != 8: raise RuntimeError(f"Outer {outer}: expected 8 selected-candidate runs, found {len(epochs)}")
        selected.append({"outer_fold": int(outer), "outer_test_subject": int(row.outer_test_subject),
                         "best_mean_validation_macro_f1": best, "lightweight_tolerance": TOLERANCE,
                         "selected_candidate_index": int(row.candidate_index), "selected_representation": str(row.representation),
                         "selected_input_channels": int(row.input_channels), "selected_normalization_layer": str(row.normalization_layer),
                         "selected_parameter_count": int(row.parameter_count), "selected_mean_validation_macro_f1": float(row.mean_validation_macro_f1),
                         "selected_std_validation_macro_f1": float(row.std_validation_macro_f1), "selected_best_epochs": json.dumps(epochs),
                         "fixed_outer_epochs": int(np.median(epochs)), "outer_test_used_for_selection": False})
    sel = pd.DataFrame(selected); sel.to_csv(output_root / "selected_candidate_by_outer_fold.csv", index=False)
    (output_root / "selected_candidate_by_outer_fold.json").write_text(json.dumps({"selection_rule": "highest mean inner Macro-F1; within 0.005 choose fewer input channels, then lower std", "outer_fold_selections": sel.to_dict(orient="records"), "outer_test_used_for_selection": False}, indent=2), encoding="utf-8")
    return s, sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--mode", choices=("smoke", "screen"), required=True)
    ap.add_argument("--verbose-epochs", action="store_true")
    a = ap.parse_args(); pr = a.processed_root.resolve(); project = pr.parents[3]
    protocol_root = a.protocol_root.resolve() if a.protocol_root else project / "outputs/protocols/pamap2_evaluation_v2"
    output_root = a.output_root.resolve() if a.output_root else project / "outputs/centralized/pamap2" / ("inner_smoke_v2" if a.mode == "smoke" else "inner_selection_v2")
    protocol_path = protocol_root / "EVALUATION_PROTOCOL_V2.json"; cand_path = protocol_root / "candidate_grid.csv"; manifest_path = protocol_root / "inner_fold_manifest.csv"
    for p in (protocol_path, cand_path, manifest_path):
        if not p.is_file(): raise FileNotFoundError(p)
    if json.loads(protocol_path.read_text(encoding="utf-8")).get("status") != "FROZEN_BEFORE_V2_MODEL_TRAINING": raise RuntimeError("Unexpected protocol status")
    cand = pd.read_csv(cand_path); manifest = pd.read_csv(manifest_path); device = device_now()
    output_root.mkdir(parents=True, exist_ok=True); runs_root = output_root / "runs"; runs_root.mkdir(exist_ok=True)
    cfg = {"mode": a.mode, "protocol_sha256": sha256(protocol_path), "candidate_grid_sha256": sha256(cand_path), "inner_manifest_sha256": sha256(manifest_path), "processed_root": str(pr), "torch_version": torch.__version__, "device": str(device)}
    cfg_path = output_root / "campaign_config.json"
    if cfg_path.is_file() and json.loads(cfg_path.read_text(encoding="utf-8")) != cfg: raise RuntimeError("Existing campaign config differs")
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("=== PAMAP2 v2 nested inner selection ==="); print(f"Mode:              {a.mode}\nProcessed dataset: {pr}\nProtocol:          {protocol_root}\nOutput:            {output_root}\nPyTorch:           {torch.__version__}\nDevice:            {device}")
    if device.type == "xpu": print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    print("\nLoading all windows and reconstructing raw scale..."); data = load_all(pr); print(f"[OK] Full36: {data.x_raw.shape}; subjects={sorted(np.unique(data.subject).tolist())}\n")
    if a.mode == "smoke":
        manifest = manifest[(manifest.outer_fold == 1) & (manifest.inner_fold == 1)]
        cand = cand[(cand.representation == "magnitude6_acc16_gyro") & (cand.normalization_layer == "groupnorm")]
        seeds, max_epochs, patience = (123,), 2, 2
    else: seeds, max_epochs, patience = INNER_SEEDS, MAX_EPOCHS, PATIENCE
    expected = len(manifest) * len(cand) * len(seeds); done = load_results(runs_root)
    keys = {(r.outer_fold, r.inner_fold, r.candidate_index, r.seed) for r in done}
    print(f"Expected runs: {expected}; reusable completed runs: {len(done)}\n")
    counter = 0
    for m in manifest.itertuples(index=False):
        outer, inner = int(m.outer_fold), int(m.inner_fold); test_s = int(str(m.outer_test_subject).replace("subject", "")); val_s = int(str(m.inner_validation_subject).replace("subject", "")); train_s = subjects(m.inner_training_subjects)
        if test_s in train_s or val_s in train_s or test_s == val_s: raise RuntimeError("Fold leakage")
        train_mask = np.isin(data.subject, np.array(train_s)); val_mask = data.subject == val_s; test_mask = data.subject == test_s
        if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask): raise RuntimeError("Overlapping masks")
        ytr = torch.from_numpy(data.y[train_mask].astype(np.int64, copy=True)); yv = torch.from_numpy(data.y[val_mask].astype(np.int64, copy=True))
        print("=" * 78); print(f"Outer {outer}/5 (test subject{test_s}) | Inner {inner}/4 (validation subject{val_s})"); print(f"Train subjects: {list(train_s)} | windows: {train_mask.sum()} train, {val_mask.sum()} validation"); print("=" * 78)
        for rep in cand.representation.drop_duplicates().tolist():
            print(f"  Preparing {rep}..."); xr = representation(data.x_raw, rep); xtr, mean, std = fold_normalize(xr, train_mask, train_mask); xv, _, _ = fold_normalize(xr, train_mask, val_mask); del xr
            nd = output_root / "fold_normalization" / f"outer{outer:02d}" / f"inner{inner:02d}" / rep; nd.mkdir(parents=True, exist_ok=True); np.save(nd / "mean.npy", mean); np.save(nd / "std.npy", std)
            (nd / "normalization.json").write_text(json.dumps({"outer_fold": outer, "outer_test_subject": test_s, "inner_fold": inner, "inner_validation_subject": val_s, "inner_training_subjects": list(train_s), "representation": rep, "train_windows": int(train_mask.sum()), "validation_windows": int(val_mask.sum()), "outer_test_statistics_used": False, "inner_validation_statistics_used": False}, indent=2), encoding="utf-8")
            for c in cand[cand.representation == rep].itertuples(index=False):
                ci, norm = int(c.candidate_index), str(c.normalization_layer)
                for seed in seeds:
                    counter += 1; key = (outer, inner, ci, int(seed)); name = f"outer{outer:02d}__inner{inner:02d}__cand{ci:02d}__{rep}__{norm}__seed{seed}"; rd = runs_root / name
                    if key in keys: print(f"  [{counter}/{expected}] SKIP {name}"); continue
                    if rd.exists(): shutil.rmtree(rd)
                    print(f"  [{counter}/{expected}] RUN  {name}")
                    meta = {"outer_fold": outer, "outer_test_subject": test_s, "inner_fold": inner, "inner_validation_subject": val_s, "inner_training_subjects": list(train_s), "candidate_index": ci, "representation": rep, "normalization_layer": norm, "seed": int(seed), "max_epochs": max_epochs, "patience": patience, "batch_size": BATCH_SIZE, "learning_rate": LR, "weight_decay": WEIGHT_DECAY}
                    r = train_run(xtr, ytr, xv, yv, meta, device, rd, max_epochs, patience, a.verbose_epochs); keys.add(key)
                    print(f"      [OK] best_epoch={r.best_epoch}; val_macro_f1={r.best_validation_macro_f1:.4f}; epochs={r.epochs_ran}; wall={r.wall_seconds:.1f}s"); clear_cache(device)
            del xtr, xv; clear_cache(device)
        del ytr, yv; clear_cache(device); print()
    done = load_results(runs_root)
    lines = ["PAMAP2 V2 NESTED INNER-SELECTION CAMPAIGN REPORT", "=" * 78, f"Mode: {a.mode}", f"Protocol: {protocol_root}", f"Processed dataset: {pr}", f"Device: {device}", f"PyTorch: {torch.__version__}", "", "EXECUTION", "-" * 78, f"Expected runs in this mode: {expected}", f"Completed runs: {len(done)}", "Execution is resumable from per-run run_result.json files.", "", "LEAKAGE CONTROLS", "-" * 78, "Outer test subject excluded from inner training and validation.", "Inner validation subject excluded from normalization fitting.", "Fold-specific z-score fitted on inner-training subjects only."]
    if a.mode == "screen" and len(done) == expected:
        _, sel = summarize(done, output_root); lines += ["", "STATUS", "-" * 78, "COMPLETE", "", "SELECTED CANDIDATE BY OUTER FOLD", "-" * 78]
        for r in sel.itertuples(index=False): lines.append(f"Outer {r.outer_fold} (test subject{r.outer_test_subject}): {r.selected_representation} + {r.selected_normalization_layer}; mean inner Macro-F1={r.selected_mean_validation_macro_f1:.4f}; fixed outer epochs={r.fixed_outer_epochs}")
    elif a.mode == "screen": lines += ["", "STATUS", "-" * 78, "INCOMPLETE", "Re-run the same command to resume."]
    else: lines += ["", "STATUS", "-" * 78, "PASS" if len(done) == expected else "INCOMPLETE", "Smoke mode does not perform candidate selection."]
    report = output_root / "INNER_SELECTION_REPORT.txt"; report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=== Inner-selection workflow finished ==="); print(f"Completed runs: {len(done)}/{expected}\nReport: {report}\n")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCampaign interrupted. Completed runs are preserved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
