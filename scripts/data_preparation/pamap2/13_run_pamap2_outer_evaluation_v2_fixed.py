from __future__ import annotations

import argparse, gc, hashlib, json, random, shutil, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 12
MODEL_CHANNELS = (32, 64, 96)
DROPOUT = 0.20
GROUPS = 8
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUTER_SEEDS = (123, 456, 789)
PILOT_EXPOSED = {101, 108}

ACTIVITY_NAMES = [
    "lying","sitting","standing","walking","running","cycling",
    "Nordic walking","ascending stairs","descending stairs",
    "vacuum cleaning","ironing","rope jumping",
]

def sha256_file(path: Path) -> str:
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

def select_device() -> torch.device:
    return torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")

def sync(device: torch.device) -> None:
    if device.type == "xpu": torch.xpu.synchronize()

def clear_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "xpu":
        try: torch.xpu.empty_cache()
        except Exception: pass

def parse_subjects(text: str) -> tuple[int, ...]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token.lower().startswith("subject"): token = token[7:]
        if token: out.append(int(token))
    return tuple(out)

@dataclass(frozen=True)
class FullDataset:
    x_raw: np.ndarray
    y: np.ndarray
    subject: np.ndarray

def load_all_raw(processed_root: Path) -> FullDataset:
    sdir = processed_root / "splits"
    tdir = processed_root / "statistics"
    mean = np.load(tdir / "training_mean_full36.npy").astype(np.float32)
    std = np.load(tdir / "training_std_full36.npy").astype(np.float32)
    xs, ys, ss = [], [], []
    for split in ("train","validation","test"):
        xnorm = np.load(sdir / f"{split}_X_full36.npy", mmap_mode="r")
        y = np.asarray(np.load(sdir / f"{split}_y.npy", mmap_mode="r"), dtype=np.int64).copy()
        sub = np.asarray(np.load(sdir / f"{split}_subject_id.npy", mmap_mode="r"), dtype=np.int64).copy()
        xraw = (np.asarray(xnorm, dtype=np.float32) * std[None,None,:] + mean[None,None,:]).astype(np.float32, copy=False)
        xs.append(xraw); ys.append(y); ss.append(sub)
    x = np.concatenate(xs); y = np.concatenate(ys); sub = np.concatenate(ss)
    if x.shape != (14972,256,36): raise RuntimeError(f"Unexpected X shape: {x.shape}")
    if not np.isfinite(x).all(): raise RuntimeError("Non-finite raw-scale values.")
    return FullDataset(x, y, sub)

def magnitude_block(x: np.ndarray, start: int) -> np.ndarray:
    parts = []
    for pos in range(3):
        a = x[:,:,start+3*pos:start+3*pos+3]
        parts.append(np.sqrt(np.sum(np.square(a, dtype=np.float32), axis=2))[:,:,None])
    return np.concatenate(parts, axis=2)

def build_representation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "core27_axes":
        idx = np.concatenate((np.arange(0,9), np.arange(18,36)))
        return np.ascontiguousarray(x[:,:,idx], dtype=np.float32)
    if name == "magnitude6_acc16_gyro":
        return np.ascontiguousarray(np.concatenate((magnitude_block(x,0), magnitude_block(x,18)), axis=2), dtype=np.float32)
    if name == "magnitude9_all":
        return np.ascontiguousarray(np.concatenate((magnitude_block(x,0), magnitude_block(x,18), magnitude_block(x,27)), axis=2), dtype=np.float32)
    raise KeyError(name)

def fit_zscore(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    train = x[mask]
    mean = train.mean(axis=(0,1), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0,1), dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all(): raise RuntimeError("Non-finite z-score stats.")
    if np.any(std <= 1e-8): raise RuntimeError("Near-zero z-score std.")
    return mean, std

def tensor_split(x: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    z = ((x[mask] - mean[None,None,:]) / std[None,None,:]).astype(np.float32, copy=False)
    z = np.ascontiguousarray(z.transpose(0,2,1), dtype=np.float32)
    if not np.isfinite(z).all(): raise RuntimeError("Non-finite normalized values.")
    return torch.from_numpy(z)

def norm_layer(name: str, channels: int) -> nn.Module:
    if name == "batchnorm": return nn.BatchNorm1d(channels)
    if name == "groupnorm":
        if channels % GROUPS: raise RuntimeError("GroupNorm divisibility error.")
        return nn.GroupNorm(GROUPS, channels)
    raise KeyError(name)

class Block(nn.Sequential):
    def __init__(self, ci:int, co:int, k:int, s:int, norm:str):
        super().__init__(
            nn.Conv1d(ci,co,kernel_size=k,stride=s,padding=k//2,bias=False),
            norm_layer(norm,co), nn.ReLU(inplace=True)
        )

class LightweightCNN1D(nn.Module):
    def __init__(self, input_channels:int, norm:str):
        super().__init__()
        c1,c2,c3 = MODEL_CHANNELS
        self.features = nn.Sequential(
            Block(input_channels,c1,7,2,norm), Block(c1,c1,5,1,norm),
            Block(c1,c2,5,2,norm), Block(c2,c2,3,1,norm),
            Block(c2,c3,3,2,norm), Block(c3,c3,3,1,norm),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(c3,NUM_CLASSES))
    def forward(self,x): return self.classifier(self.features(x))

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float

def metrics(y_true, y_pred, loss) -> Metrics:
    return Metrics(
        float(loss),
        float(accuracy_score(y_true,y_pred)),
        float(balanced_accuracy_score(y_true,y_pred)),
        float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
    )

def loaders(xtr,ytr,xte,yte,seed):
    g = torch.Generator(); g.manual_seed(seed)
    return (
        DataLoader(TensorDataset(xtr,ytr),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,generator=g),
        DataLoader(TensorDataset(xte,yte),batch_size=BATCH_SIZE,shuffle=False,num_workers=0),
    )

def train_epoch(model, loader, criterion, optimizer, device) -> Metrics:
    model.train(); total=0.0; n=0; yt=[]; yp=[]
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(xb); loss=criterion(logits,yb); loss.backward(); optimizer.step()
        bs=int(yb.shape[0]); total += float(loss.detach().cpu().item())*bs; n += bs
        yt.append(yb.detach().cpu().numpy()); yp.append(logits.detach().argmax(1).cpu().numpy())
    return metrics(np.concatenate(yt), np.concatenate(yp), total/max(n,1))

@torch.inference_mode()
def test_once(model, loader, criterion, device):
    model.eval(); total=0.0; n=0; yt=[]; yp=[]
    sync(device); start=time.perf_counter()
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        logits=model(xb); loss=criterion(logits,yb)
        bs=int(yb.shape[0]); total += float(loss.detach().cpu().item())*bs; n += bs
        yt.append(yb.cpu().numpy()); yp.append(logits.argmax(1).cpu().numpy())
    sync(device); elapsed=time.perf_counter()-start
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return metrics(yt,yp,total/max(n,1)), yt, yp, elapsed

def save_confusion(path: Path, yt, yp) -> None:
    m = confusion_matrix(yt,yp,labels=list(range(NUM_CLASSES)))
    pd.DataFrame(
        m,
        index=[f"true_{i}_{ACTIVITY_NAMES[i]}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}_{ACTIVITY_NAMES[i]}" for i in range(NUM_CLASSES)],
    ).to_csv(path)

def save_per_class(path: Path, yt, yp) -> None:
    vals=f1_score(yt,yp,labels=list(range(NUM_CLASSES)),average=None,zero_division=0)
    pd.DataFrame({"class_index":range(NUM_CLASSES),"activity_name":ACTIVITY_NAMES,"test_f1":vals}).to_csv(path,index=False)

@dataclass
class OuterRunResult:
    outer_fold:int
    outer_test_subject:int
    pilot_heldout_previously_observed:bool
    outer_training_subjects:str
    representation:str
    input_channels:int
    normalization_layer:str
    fixed_epochs:int
    seed:int
    parameter_count:int
    training_windows:int
    test_windows:int
    test_loss:float
    test_accuracy:float
    test_balanced_accuracy:float
    test_macro_f1:float
    training_wall_seconds:float
    test_inference_seconds:float
    test_inference_ms_per_window:float
    run_directory:str

def load_completed(runs_root: Path) -> list[OuterRunResult]:
    out=[]
    if runs_root.is_dir():
        for p in sorted(runs_root.glob("*/run_result.json")):
            out.append(OuterRunResult(**json.loads(p.read_text(encoding="utf-8"))))
    return out

def aggregate(results: list[OuterRunResult], output_root: Path):
    runs=pd.DataFrame([asdict(r) for r in results]).sort_values(["outer_fold","seed"])
    runs.to_csv(output_root/"outer_test_runs.csv",index=False)

    fold=(runs.groupby([
        "outer_fold","outer_test_subject","pilot_heldout_previously_observed",
        "representation","input_channels","normalization_layer","fixed_epochs",
        "parameter_count","training_windows","test_windows"
    ],as_index=False).agg(
        seeds=("seed","count"),
        mean_test_macro_f1=("test_macro_f1","mean"),
        std_test_macro_f1=("test_macro_f1","std"),
        mean_test_balanced_accuracy=("test_balanced_accuracy","mean"),
        std_test_balanced_accuracy=("test_balanced_accuracy","std"),
        mean_test_accuracy=("test_accuracy","mean"),
        std_test_accuracy=("test_accuracy","std"),
        mean_test_loss=("test_loss","mean"),
        std_test_loss=("test_loss","std"),
        mean_training_wall_seconds=("training_wall_seconds","mean"),
        mean_test_inference_ms_per_window=("test_inference_ms_per_window","mean"),
    ).sort_values("outer_fold").reset_index(drop=True))
    fold.to_csv(output_root/"outer_fold_summary.csv",index=False)

    primary={"primary_unit":"outer_subject_fold_mean","outer_folds":int(len(fold))}
    for col,short in [
        ("mean_test_macro_f1","macro_f1"),
        ("mean_test_balanced_accuracy","balanced_accuracy"),
        ("mean_test_accuracy","accuracy"),
    ]:
        v=fold[col].to_numpy(float)
        primary[f"mean_across_outer_subjects_{short}"]=float(v.mean())
        primary[f"std_across_outer_subjects_{short}"]=float(v.std(ddof=1))
        primary[f"median_across_outer_subjects_{short}"]=float(np.median(v))
        primary[f"min_outer_subject_{short}"]=float(v.min())
        primary[f"max_outer_subject_{short}"]=float(v.max())

    secondary={"secondary_unit":"all_seed_runs","runs":int(len(runs))}
    for col in ("test_macro_f1","test_balanced_accuracy","test_accuracy"):
        v=runs[col].to_numpy(float)
        secondary[f"mean_{col}"]=float(v.mean())
        secondary[f"std_{col}"]=float(v.std(ddof=1))

    exp=[]
    for exposed,g in fold.groupby("pilot_heldout_previously_observed"):
        exp.append({
            "stratum":"previously_observed_in_pilot" if bool(exposed) else "not_previously_held_out_in_pilot",
            "outer_folds":int(len(g)),
            "outer_subjects":",".join(str(int(x)) for x in g.outer_test_subject.tolist()),
            "mean_fold_macro_f1":float(g.mean_test_macro_f1.mean()),
            "std_fold_macro_f1":float(g.mean_test_macro_f1.std(ddof=1)) if len(g)>1 else float("nan"),
            "mean_fold_balanced_accuracy":float(g.mean_test_balanced_accuracy.mean()),
            "mean_fold_accuracy":float(g.mean_test_accuracy.mean()),
        })
    pd.DataFrame(exp).to_csv(output_root/"pilot_exposure_stratification.csv",index=False)

    pc=[]
    matrices=[]
    for r in results:
        rd=Path(r.run_directory)
        frame=pd.read_csv(rd/"test_per_class_f1.csv")
        for row in frame.itertuples(index=False):
            pc.append({
                "outer_fold":r.outer_fold,"outer_test_subject":r.outer_test_subject,"seed":r.seed,
                "class_index":int(row.class_index),"activity_name":str(row.activity_name),"test_f1":float(row.test_f1)
            })
        matrices.append(pd.read_csv(rd/"test_confusion_matrix.csv",index_col=0).to_numpy(np.int64))
    pc_runs=pd.DataFrame(pc); pc_runs.to_csv(output_root/"outer_test_per_class_f1_runs.csv",index=False)
    pc_fold=(pc_runs.groupby(["outer_fold","outer_test_subject","class_index","activity_name"],as_index=False)
             .agg(mean_seed_test_f1=("test_f1","mean"),std_seed_test_f1=("test_f1","std")))
    pc_fold.to_csv(output_root/"outer_test_per_class_f1_by_fold.csv",index=False)
    pc_summary=(pc_fold.groupby(["class_index","activity_name"],as_index=False)
                .agg(mean_outer_subject_f1=("mean_seed_test_f1","mean"),
                     std_outer_subject_f1=("mean_seed_test_f1","std"),
                     min_outer_subject_f1=("mean_seed_test_f1","min"),
                     max_outer_subject_f1=("mean_seed_test_f1","max"))
                .sort_values("class_index"))
    pc_summary.to_csv(output_root/"outer_test_per_class_f1_summary.csv",index=False)

    agg=np.sum(np.stack(matrices),axis=0)
    pd.DataFrame(
        agg,
        index=[f"true_{i}_{ACTIVITY_NAMES[i]}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}_{ACTIVITY_NAMES[i]}" for i in range(NUM_CLASSES)],
    ).to_csv(output_root/"aggregate_outer_test_confusion_matrix.csv")

    overall={
        "primary_subject_level_summary":primary,
        "secondary_all_run_summary":secondary,
        "important_note":"Primary reporting treats the five outer subjects as the independent evaluation units. The 15 seed runs are not treated as 15 independent subjects."
    }
    (output_root/"overall_outer_summary.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    return fold, pc_summary, overall

def write_report(output_root, processed_root, protocol_root, inner_root, device, results, fold=None, pc=None, overall=None):
    lines=[
        "PAMAP2 V2 NESTED OUTER-EVALUATION REPORT","="*78,
        f"Processed dataset: {processed_root}",f"Protocol: {protocol_root}",
        f"Inner selection: {inner_root}",f"Device: {device}",f"PyTorch: {torch.__version__}",
        "","EXECUTION","-"*78,f"Expected outer runs: 15",f"Completed outer runs: {len(results)}",
        "Execution is resumable from per-run run_result.json files.",
        "","LEAKAGE CONTROLS","-"*78,
        "Candidate and epoch count fixed by nested inner selection.",
        "Outer test subject excluded from all training.",
        "Outer-fold z-score fitted only on outer-training subjects.",
        "No outer-test statistic used for normalization or model selection.",
        "Each trained model evaluated on its outer test subject once.",
    ]
    if fold is None:
        lines += ["","STATUS","-"*78,"INCOMPLETE","Re-run the same command to resume."]
    else:
        primary=overall["primary_subject_level_summary"]
        lines += ["","STATUS","-"*78,"COMPLETE","","OUTER-FOLD RESULTS (SEED MEAN +/- SAMPLE STD)","-"*78]
        for r in fold.itertuples(index=False):
            lines.append(
                f"Outer {r.outer_fold} (subject{r.outer_test_subject}): {r.representation} + {r.normalization_layer}; "
                f"epochs={r.fixed_epochs}; Macro-F1={r.mean_test_macro_f1:.4f} +/- {r.std_test_macro_f1:.4f}; "
                f"BalAcc={r.mean_test_balanced_accuracy:.4f}; Acc={r.mean_test_accuracy:.4f}"
            )
        lines += [
            "","PRIMARY RESULT: ACROSS OUTER SUBJECTS","-"*78,
            f"Macro-F1: {primary['mean_across_outer_subjects_macro_f1']:.4f} +/- {primary['std_across_outer_subjects_macro_f1']:.4f}",
            f"Balanced accuracy: {primary['mean_across_outer_subjects_balanced_accuracy']:.4f} +/- {primary['std_across_outer_subjects_balanced_accuracy']:.4f}",
            f"Accuracy: {primary['mean_across_outer_subjects_accuracy']:.4f} +/- {primary['std_across_outer_subjects_accuracy']:.4f}",
            "","Primary unit: one mean result per outer subject.","The 15 seed runs are not treated as 15 independent subjects.",
            "","PER-CLASS F1 ACROSS OUTER SUBJECTS","-"*78,
        ]
        for r in pc.itertuples(index=False):
            lines.append(f"{r.activity_name}: {r.mean_outer_subject_f1:.4f} +/- {r.std_outer_subject_f1:.4f}")
        lines += [
            "","TRANSPARENCY","-"*78,
            "subject101 and subject108 had previously been observed as held-out subjects during the pilot phase.",
            "The v2 protocol was frozen before v2 model training and is reported as post-pilot nested cross-validation.",
        ]
    path=output_root/"OUTER_EVALUATION_REPORT.txt"
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed-root",type=Path,required=True)
    ap.add_argument("--protocol-root",type=Path,default=None)
    ap.add_argument("--inner-selection-root",type=Path,default=None)
    ap.add_argument("--output-root",type=Path,default=None)
    ap.add_argument("--verbose-epochs",action="store_true")
    args=ap.parse_args()

    processed=args.processed_root.expanduser().resolve()
    project=processed.parents[3]
    protocol_root=(args.protocol_root.expanduser().resolve() if args.protocol_root else project/"outputs"/"protocols"/"pamap2_evaluation_v2")
    inner_root=(args.inner_selection_root.expanduser().resolve() if args.inner_selection_root else project/"outputs"/"centralized"/"pamap2"/"inner_selection_v2")
    output=(args.output_root.expanduser().resolve() if args.output_root else project/"outputs"/"centralized"/"pamap2"/"outer_evaluation_v2")

    protocol_path=protocol_root/"EVALUATION_PROTOCOL_V2.json"
    manifest_path=protocol_root/"outer_fold_manifest.csv"
    selection_path=inner_root/"selected_candidate_by_outer_fold.csv"
    for path in (protocol_path,manifest_path,selection_path):
        if not path.is_file(): raise FileNotFoundError(path)

    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status")!="FROZEN_BEFORE_V2_MODEL_TRAINING": raise RuntimeError("Unexpected protocol status.")

    manifest=pd.read_csv(manifest_path)
    selection=pd.read_csv(selection_path)

    # Normalize merge keys explicitly. The protocol manifest stores subjects
    # as strings such as "subject101", whereas the inner-selection output
    # stores the same subject IDs as integers such as 101.
    manifest["outer_fold"] = pd.to_numeric(
        manifest["outer_fold"], errors="raise"
    ).astype(int)
    selection["outer_fold"] = pd.to_numeric(
        selection["outer_fold"], errors="raise"
    ).astype(int)

    manifest["outer_test_subject"] = (
        manifest["outer_test_subject"]
        .astype(str)
        .str.strip()
        .str.replace(r"^subject", "", regex=True)
        .pipe(pd.to_numeric, errors="raise")
        .astype(int)
    )
    selection["outer_test_subject"] = (
        selection["outer_test_subject"]
        .astype(str)
        .str.strip()
        .str.replace(r"^subject", "", regex=True)
        .pipe(pd.to_numeric, errors="raise")
        .astype(int)
    )

    merged=manifest.merge(
        selection,
        on=["outer_fold","outer_test_subject"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged)!=5: raise RuntimeError("Expected exactly 5 merged outer folds.")

    device=select_device()
    output.mkdir(parents=True,exist_ok=True)
    runs_root=output/"runs"; runs_root.mkdir(parents=True,exist_ok=True)

    config={
        "protocol_sha256":sha256_file(protocol_path),
        "outer_manifest_sha256":sha256_file(manifest_path),
        "selection_sha256":sha256_file(selection_path),
        "processed_root":str(processed),
        "torch_version":torch.__version__,
        "device":str(device),
        "outer_seeds":list(OUTER_SEEDS),
        "expected_runs":15,
    }
    config_path=output/"campaign_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8"))!=config:
            raise RuntimeError("Existing campaign config does not match.")
    else:
        config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")

    print("=== PAMAP2 v2 nested outer evaluation ===")
    print(f"Processed dataset: {processed}")
    print(f"Protocol:          {protocol_root}")
    print(f"Inner selection:   {inner_root}")
    print(f"Output:            {output}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")
    if device.type=="xpu": print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    print()

    print("Loading all windows and reconstructing raw scale...")
    data=load_all_raw(processed)
    print(f"[OK] Full36: {data.x_raw.shape}; subjects={sorted(np.unique(data.subject).tolist())}\n")

    completed=load_completed(runs_root)
    done={(r.outer_fold,r.seed) for r in completed}
    print(f"Expected runs: 15; reusable completed runs: {len(completed)}\n")

    counter=0
    for row in merged.sort_values("outer_fold").itertuples(index=False):
        outer_fold=int(row.outer_fold)
        test_subject=int(row.outer_test_subject)
        train_subjects=parse_subjects(row.outer_training_subjects)
        rep=str(row.selected_representation)
        norm=str(row.selected_normalization_layer)
        epochs=int(row.fixed_outer_epochs)

        if test_subject in train_subjects: raise RuntimeError("Outer test leakage.")
        train_mask=np.isin(data.subject,np.asarray(train_subjects,dtype=np.int64))
        test_mask=data.subject==test_subject
        if np.any(train_mask & test_mask): raise RuntimeError("Train/test overlap.")

        print("="*78)
        print(f"Outer {outer_fold}/5 | test subject{test_subject}")
        print(f"Selected: {rep} + {norm}")
        print(f"Fixed epochs: {epochs}")
        print(f"Training subjects: {list(train_subjects)}")
        print(f"Windows: train={int(train_mask.sum())}; test={int(test_mask.sum())}")
        print("="*78)

        xrep=build_representation(data.x_raw,rep)
        mean,std=fit_zscore(xrep,train_mask)
        xtr=tensor_split(xrep,train_mask,mean,std)
        xte=tensor_split(xrep,test_mask,mean,std)
        ytr=torch.from_numpy(data.y[train_mask].astype(np.int64,copy=True))
        yte=torch.from_numpy(data.y[test_mask].astype(np.int64,copy=True))

        ndir=output/"fold_normalization"/f"outer{outer_fold:02d}"
        ndir.mkdir(parents=True,exist_ok=True)
        np.save(ndir/"mean.npy",mean); np.save(ndir/"std.npy",std)
        (ndir/"normalization.json").write_text(json.dumps({
            "outer_fold":outer_fold,"outer_test_subject":test_subject,
            "outer_training_subjects":list(train_subjects),"representation":rep,
            "input_channels":int(xtr.shape[1]),"normalization_layer":norm,
            "fixed_epochs":epochs,"training_windows":int(xtr.shape[0]),
            "test_windows":int(xte.shape[0]),"outer_test_statistics_used":False
        },indent=2),encoding="utf-8")
        del xrep

        for seed in OUTER_SEEDS:
            counter+=1
            key=(outer_fold,int(seed))
            name=f"outer{outer_fold:02d}__subject{test_subject}__{rep}__{norm}__seed{seed}"
            rdir=runs_root/name
            if key in done:
                print(f"  [{counter}/15] SKIP completed: {name}")
                continue
            if rdir.exists(): shutil.rmtree(rdir)
            print(f"  [{counter}/15] RUN {name}")

            set_seed(seed)
            model=LightweightCNN1D(int(xtr.shape[1]),norm).to(device)
            criterion=nn.CrossEntropyLoss()
            optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
            train_loader,test_loader=loaders(xtr,ytr,xte,yte,seed)
            rdir.mkdir(parents=True,exist_ok=False)
            (rdir/"run_config.json").write_text(json.dumps({
                "outer_fold":outer_fold,"outer_test_subject":test_subject,
                "outer_training_subjects":list(train_subjects),"representation":rep,
                "normalization_layer":norm,"fixed_epochs":epochs,"seed":seed,
                "outer_test_used_for_selection":False,"outer_test_statistics_used_for_normalization":False
            },indent=2),encoding="utf-8")

            hist=[]; start=time.perf_counter()
            for epoch in range(1,epochs+1):
                estart=time.perf_counter()
                m=train_epoch(model,train_loader,criterion,optimizer,device)
                sync(device)
                hist.append({
                    "epoch":epoch,"epoch_seconds":time.perf_counter()-estart,
                    "train_loss":m.loss,"train_accuracy":m.accuracy,
                    "train_balanced_accuracy":m.balanced_accuracy,"train_macro_f1":m.macro_f1
                })
                if args.verbose_epochs:
                    print(f"      epoch={epoch:02d}/{epochs:02d} train_loss={m.loss:.4f} train_f1={m.macro_f1:.4f}")
            sync(device); train_wall=time.perf_counter()-start
            pd.DataFrame(hist).to_csv(rdir/"training_history.csv",index=False)

            tm,yt,yp,infer=test_once(model,test_loader,criterion,device)
            save_confusion(rdir/"test_confusion_matrix.csv",yt,yp)
            save_per_class(rdir/"test_per_class_f1.csv",yt,yp)

            result=OuterRunResult(
                outer_fold=outer_fold,outer_test_subject=test_subject,
                pilot_heldout_previously_observed=test_subject in PILOT_EXPOSED,
                outer_training_subjects=",".join(str(s) for s in train_subjects),
                representation=rep,input_channels=int(xtr.shape[1]),normalization_layer=norm,
                fixed_epochs=epochs,seed=seed,parameter_count=count_params(model),
                training_windows=int(xtr.shape[0]),test_windows=int(xte.shape[0]),
                test_loss=tm.loss,test_accuracy=tm.accuracy,
                test_balanced_accuracy=tm.balanced_accuracy,test_macro_f1=tm.macro_f1,
                training_wall_seconds=train_wall,test_inference_seconds=infer,
                test_inference_ms_per_window=1000.0*infer/int(xte.shape[0]),
                run_directory=str(rdir),
            )
            (rdir/"run_result.json").write_text(json.dumps(asdict(result),indent=2),encoding="utf-8")
            torch.save({
                "model_state_dict":model.state_dict(),"outer_fold":outer_fold,
                "outer_test_subject":test_subject,"representation":rep,
                "normalization_layer":norm,"fixed_epochs":epochs,"seed":seed,
                "test_metrics":asdict(tm),"parameter_count":count_params(model),
            },rdir/"final_model.pt")

            completed.append(result); done.add(key)
            print(f"      [TEST] Macro-F1={tm.macro_f1:.4f}; BalAcc={tm.balanced_accuracy:.4f}; Acc={tm.accuracy:.4f}; train_wall={train_wall:.1f}s")
            del model, optimizer, train_loader, test_loader
            clear_cache(device)

        del xtr,xte,ytr,yte
        clear_cache(device)
        print()

    completed=load_completed(runs_root)
    fold=pc=overall=None
    if len(completed)==15:
        fold,pc,overall=aggregate(completed,output)
    report=write_report(output,processed,protocol_root,inner_root,device,completed,fold,pc,overall)

    print("=== Outer-evaluation workflow finished ===")
    print(f"Completed runs: {len(completed)}/15")
    print(f"Report: {report}")
    if overall is not None:
        p=overall["primary_subject_level_summary"]
        print("\nPRIMARY ACROSS-SUBJECT RESULT")
        print(f"Macro-F1: {p['mean_across_outer_subjects_macro_f1']:.4f} +/- {p['std_across_outer_subjects_macro_f1']:.4f}")
        print(f"Balanced accuracy: {p['mean_across_outer_subjects_balanced_accuracy']:.4f} +/- {p['std_across_outer_subjects_balanced_accuracy']:.4f}")
        print(f"Accuracy: {p['mean_across_outer_subjects_accuracy']:.4f} +/- {p['std_across_outer_subjects_accuracy']:.4f}")
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nOuter evaluation interrupted. Completed runs are preserved; rerun the same command to resume.",file=sys.stderr)
        raise SystemExit(130)
