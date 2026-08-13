from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

RUNNER_ID = "CICIoT2023_FROZEN_180_RUN_SCIENTIFIC_CAMPAIGN_RUNNER_V1"
AUTHORIZATION_ID = "CICIoT2023_GATE107_ONE_RUN_CANARY_AUTHORIZATION_V1"
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)
ORDER_KEY_2_CONSTANT = np.uint64(0xD1B54A32D192ED03)
CLASS_WEIGHTS = [
    2.3619476556261541,
    21.618424210059288,
    0.61793142035948634,
    1.5759929623559756,
    2.989431643370104,
    3.6669224591635454,
    15.7394759911818,
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def splitmix64_vector(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = x + SPLITMIX_GAMMA
        x = (x ^ (x >> np.uint64(30))) * SPLITMIX_M1
        x = (x ^ (x >> np.uint64(27))) * SPLITMIX_M2
        x = x ^ (x >> np.uint64(31))
    return x.astype(np.uint64, copy=False)


def rotl64_scalar(value: int, shift: int) -> int:
    value &= MASK64
    shift &= 63
    return (((value << shift) & MASK64) | (value >> (64 - shift))) & MASK64


def exact_local_order_indices(hash1: np.ndarray, hash2: np.ndarray, seed_uint64: int) -> np.ndarray:
    h1 = np.asarray(hash1, dtype=np.uint64)
    h2 = np.asarray(hash2, dtype=np.uint64)
    seed = np.uint64(seed_uint64)
    rotated_seed = np.uint64(rotl64_scalar(seed_uint64, 32))
    key1 = splitmix64_vector(h1 ^ seed)
    key2 = splitmix64_vector(h2 ^ rotated_seed ^ ORDER_KEY_2_CONSTANT)
    return np.lexsort((h2, h1, key2, key1)).astype(np.int64, copy=False)


class ScientificMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(39, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_is_finite(state: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(v).all().item() for v in state.values())


def import_module_exact(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def normalize_state_result(value: Any, reference_keys: set[str]) -> tuple[dict[str, torch.Tensor], Any]:
    def is_state_dict(x: Any) -> bool:
        return isinstance(x, dict) and set(x.keys()) == reference_keys and all(torch.is_tensor(v) for v in x.values())

    if is_state_dict(value):
        return clone_state(value), None
    if isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            if is_state_dict(item):
                aux = [v for j, v in enumerate(value) if j != idx]
                return clone_state(item), aux
    if isinstance(value, dict):
        for key in ("state_dict", "global_state", "aggregated_state", "new_state", "model_state"):
            if key in value and is_state_dict(value[key]):
                return clone_state(value[key]), {k: v for k, v in value.items() if k != key}
    raise RuntimeError(f"Method kernel returned no recognizable state_dict; type={type(value)!r}")


def invoke_with_context(fn, context: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in context:
            kwargs[name] = context[name]
        elif param.default is inspect._empty:
            missing.append(name)
    require(not missing, f"Missing required context for {fn.__name__}{sig}: {missing}")
    return fn(**kwargs)


def build_common_context(
    *,
    global_state: dict[str, torch.Tensor],
    local_states: list[dict[str, torch.Tensor]],
    local_losses: list[float],
    selected_clients: list[int],
    sample_counts: list[int],
    round_number: int,
    method_state: Any,
) -> dict[str, Any]:
    deltas = [
        {k: local[k] - global_state[k] for k in global_state}
        for local in local_states
    ]
    equal_weights = np.full(len(local_states), 1.0 / max(1, len(local_states)), dtype=np.float64)
    round_reference = fedavg_state(local_states, sample_counts)
    return {
        "global_state": global_state,
        "base_state": global_state,
        "reference_state": global_state,
        "round_reference": round_reference,
        "local_states": local_states,
        "client_states": local_states,
        "states": local_states,
        "updates": deltas,
        "client_updates": deltas,
        "deltas": deltas,
        "client_deltas": deltas,
        "local_losses": local_losses,
        "client_losses": local_losses,
        "losses": local_losses,
        "selected_clients": selected_clients,
        "selected_client_ids": selected_clients,
        "client_ids": selected_clients,
        "sample_counts": sample_counts,
        "client_sizes": sample_counts,
        "n_samples": sample_counts,
        "weights": equal_weights,
        "client_weights": equal_weights,
        "selection_probabilities": equal_weights,
        "probabilities": equal_weights,
        "round_idx": round_number - 1,
        "round_index": round_number - 1,
        "round_number": round_number,
        "round_id": round_number,
        "current_round": round_number,
        "state": method_state,
        "method_state": method_state,
        "trust_state": method_state,
        "risk_state": method_state,
        "device": torch.device("cpu"),
    }


def fedavg_state(states: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    require(len(states) > 0 and len(states) == len(sample_counts), "Invalid FedAvg input")
    total = float(sum(sample_counts))
    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float64)
        for state, count in zip(states, sample_counts):
            acc += state[key].to(torch.float64) * (float(count) / total)
        out[key] = acc.to(states[0][key].dtype)
    return out


def common_local_train(
    global_state: dict[str, torch.Tensor],
    x_block: np.ndarray,
    y_block: np.ndarray,
    class_weights: torch.Tensor,
    local_seed: int,
) -> dict[str, Any]:
    require(x_block.shape == (51200, 39), f"Unexpected x_block shape {x_block.shape}")
    require(y_block.shape == (51200,), f"Unexpected y_block shape {y_block.shape}")
    torch.manual_seed(int(local_seed))
    model = ScientificMLP()
    model.load_state_dict(global_state, strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    x_tensor = torch.from_numpy(np.ascontiguousarray(x_block, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_block, dtype=np.int64))
    for step in range(200):
        start = step * 256
        end = start + 256
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor[start:end])
        loss = criterion(logits, y_tensor[start:end])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    state = cpu_state_dict(model)
    require(state_is_finite(state), "Non-finite common local state")
    return {
        "state_dict": state,
        "train_loss": total_loss / 200.0,
        "batch_count": 200,
        "total_examples": 51200,
        "optimizer_class": "Adam",
    }


class FrozenDataSurface:
    def __init__(self, binding: dict[str, Any], config_id: str):
        p = binding["paths"]
        self.effective_root = Path(p["effective_root"])
        self.feature_root = Path(p["feature_root"])
        self.split_root = Path(p["split_root"])
        self.assignment_root = Path(p["assignment_root_base"]) / config_id
        self._client_candidates: dict[int, dict[str, np.ndarray]] | None = None

    @staticmethod
    def bucket_file(root: Path, bucket: int, name: str) -> Path:
        path = root / f"bucket_{bucket:03d}" / name
        require(path.is_file(), f"Missing bucket file: {path}")
        return path

    def gather_all_clients(self) -> dict[int, dict[str, np.ndarray]]:
        if self._client_candidates is not None:
            return self._client_candidates
        parts: dict[int, dict[str, list[np.ndarray]]] = {
            c: {k: [] for k in ("label", "source_bucket", "source_row", "hash1", "hash2")}
            for c in range(30)
        }
        for bucket in range(256):
            effective = np.load(self.bucket_file(self.effective_root, bucket, "effective_observation_index.npy"), mmap_mode="r", allow_pickle=False)
            assignment = np.load(self.bucket_file(self.assignment_root, bucket, "client_id_u8.npy"), mmap_mode="r", allow_pickle=False)
            for client_id in range(30):
                rows = np.flatnonzero(np.asarray(assignment) == client_id)
                if len(rows) == 0:
                    continue
                parts[client_id]["label"].append(np.asarray(effective["Task7LabelID"][rows], dtype=np.uint8))
                parts[client_id]["source_bucket"].append(np.asarray(effective["RepresentativeSourceBucket"][rows], dtype=np.uint8))
                parts[client_id]["source_row"].append(np.asarray(effective["RepresentativeSourceRowIndex"][rows], dtype=np.uint32))
                parts[client_id]["hash1"].append(np.asarray(effective["TransformedHash1"][rows], dtype=np.uint64))
                parts[client_id]["hash2"].append(np.asarray(effective["TransformedHash2"][rows], dtype=np.uint64))
            del effective, assignment
        result: dict[int, dict[str, np.ndarray]] = {}
        for client_id, fields in parts.items():
            result[client_id] = {k: np.concatenate(v) for k, v in fields.items()}
        self._client_candidates = result
        return result

    def materialize_client_block(self, client_id: int, local_order_seed: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        candidates = self.gather_all_clients()[client_id]
        order = exact_local_order_indices(candidates["hash1"], candidates["hash2"], local_order_seed)
        selected = order[start:end]
        require(len(selected) == 51200, f"Client block length {len(selected)} != 51200")
        labels = np.asarray(candidates["label"][selected], dtype=np.uint8)
        source_buckets = np.asarray(candidates["source_bucket"][selected], dtype=np.uint8)
        source_rows = np.asarray(candidates["source_row"][selected], dtype=np.uint32)
        x = np.empty((51200, 39), dtype=np.float32)
        for source_bucket in np.unique(source_buckets).tolist():
            pos = np.flatnonzero(source_buckets == source_bucket)
            rows = np.asarray(source_rows[pos], dtype=np.int64)
            features = np.load(self.bucket_file(self.feature_root, int(source_bucket), "features_f32.npy"), mmap_mode="r", allow_pickle=False)
            x[pos, :] = np.asarray(features[rows, :], dtype=np.float32)
            del features
        require(np.isfinite(x).all(), "Non-finite client block")
        return x, labels

    def iter_split_batches(self, split_id: int, batch_size: int = 8192) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for bucket in range(256):
            effective = np.load(self.bucket_file(self.effective_root, bucket, "effective_observation_index.npy"), mmap_mode="r", allow_pickle=False)
            split = np.load(self.bucket_file(self.split_root, bucket, "split_id_u8.npy"), mmap_mode="r", allow_pickle=False)
            rows_all = np.flatnonzero(np.asarray(split) == split_id)
            for offset in range(0, len(rows_all), batch_size):
                rows = rows_all[offset:offset + batch_size]
                labels = np.asarray(effective["Task7LabelID"][rows], dtype=np.int64)
                source_buckets = np.asarray(effective["RepresentativeSourceBucket"][rows], dtype=np.uint8)
                source_rows = np.asarray(effective["RepresentativeSourceRowIndex"][rows], dtype=np.uint32)
                x = np.empty((len(rows), 39), dtype=np.float32)
                for source_bucket in np.unique(source_buckets).tolist():
                    pos = np.flatnonzero(source_buckets == source_bucket)
                    src_rows = np.asarray(source_rows[pos], dtype=np.int64)
                    features = np.load(self.bucket_file(self.feature_root, int(source_bucket), "features_f32.npy"), mmap_mode="r", allow_pickle=False)
                    x[pos, :] = np.asarray(features[src_rows, :], dtype=np.float32)
                    del features
                yield x, labels
            del effective, split


def evaluate_state(state: dict[str, torch.Tensor], data: FrozenDataSurface, split_id: int) -> dict[str, float]:
    model = ScientificMLP()
    model.load_state_dict(state, strict=True)
    model.eval()
    confusion = np.zeros((7, 7), dtype=np.int64)
    total_loss = 0.0
    total_count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in data.iter_split_batches(split_id):
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
            logits = model(xt)
            total_loss += float(criterion(logits, yt).cpu())
            pred = logits.argmax(dim=1).cpu().numpy()
            np.add.at(confusion, (y, pred), 1)
            total_count += len(y)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    accuracy = float(tp.sum() / max(1, total_count))
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
    }


def better_checkpoint(candidate: dict[str, float], best: dict[str, Any] | None, round_number: int) -> bool:
    if best is None:
        return True
    if candidate["macro_f1"] > best["macro_f1"]:
        return True
    if candidate["macro_f1"] < best["macro_f1"]:
        return False
    if candidate["balanced_accuracy"] > best["balanced_accuracy"]:
        return True
    if candidate["balanced_accuracy"] < best["balanced_accuracy"]:
        return False
    return round_number < int(best["round"])


def contract_check(binding: dict[str, Any]) -> dict[str, Any]:
    dispatch = binding["method_dispatch"]
    report: dict[str, Any] = {"runner_id": RUNNER_ID, "methods": {}}
    for method_id, spec in dispatch.items():
        method_report: dict[str, Any] = {}
        for role in ("state_update", "aggregation"):
            entry = spec.get(role)
            if not entry:
                method_report[role] = None
                continue
            path = Path(entry["path"])
            require(sha256_file(path) == entry["sha256"], f"Source SHA mismatch: {path}")
            module = import_module_exact(path, f"contract_{method_id}_{role}")
            fn = getattr(module, entry["callable"])
            method_report[role] = {
                "callable": entry["callable"],
                "signature": str(inspect.signature(fn)),
                "source_sha256": entry["sha256"],
            }
        report["methods"][method_id] = method_report
    fedprox = Path(binding["paths"]["fedprox_rebound_adapter"])
    require(sha256_file(fedprox) == binding["hashes"]["fedprox_rebound_adapter_sha256"], "FedProx rebound adapter SHA mismatch")
    module = import_module_exact(fedprox, "contract_fedprox_rebound")
    fn = getattr(module, "train_one_client_fedprox_rebound")
    report["fedprox_rebound_signature"] = str(inspect.signature(fn))
    report["scientific_optimizer_steps_executed"] = 0
    report["scientific_training_started"] = False
    return report


def load_method_kernels(binding: dict[str, Any], method_id: str):
    spec = binding["method_dispatch"][method_id]
    out = {}
    for role in ("state_update", "aggregation"):
        entry = spec.get(role)
        if not entry:
            out[role] = None
            continue
        path = Path(entry["path"])
        require(sha256_file(path) == entry["sha256"], f"Method source SHA mismatch: {path}")
        module = import_module_exact(path, f"scientific_{method_id}_{role}")
        out[role] = getattr(module, entry["callable"])
    return out


def apply_attack(global_state: dict[str, torch.Tensor], local_state: dict[str, torch.Tensor], malicious: bool, active: bool) -> dict[str, torch.Tensor]:
    if not (malicious and active):
        return clone_state(local_state)
    return {k: global_state[k] - 5.0 * (local_state[k] - global_state[k]) for k in global_state}


def parse_round_set(text: str) -> set[int]:
    raw = str(text).strip()
    if not raw or raw.upper() == "NONE":
        return set()
    tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    require(all(token.upper() != "NONE" for token in tokens), "AttackActiveRounds NONE marker must be standalone")
    values: set[int] = set()
    for token in tokens:
        upper = token.upper()
        if "_TO_" in upper:
            a, b = upper.split("_TO_", 1)
            start, end = int(a), int(b)
            require(start <= end, f"Invalid attack-round range: {token}")
            values.update(range(start, end + 1))
        elif "-" in token:
            a, b = token.split("-", 1)
            start, end = int(a), int(b)
            require(start <= end, f"Invalid attack-round range: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    require(all(1 <= value <= 20 for value in values), f"AttackActiveRounds outside frozen 1..20 range: {sorted(values)}")
    return values


def resolve_malicious_clients(binding: dict[str, Any], seed: int, scenario_row: dict[str, str]) -> set[int]:
    prefix = int(float(scenario_row["MaliciousPrefixSize"]))
    if prefix <= 0:
        return set()
    rows = read_csv(Path(binding["paths"]["malicious_rankings"]))
    seed_cols = [c for c in rows[0] if "seed" in c.lower()]
    client_cols = [c for c in rows[0] if "client" in c.lower()]
    rank_cols = [c for c in rows[0] if "rank" in c.lower()]
    require(seed_cols and client_cols and rank_cols, "Could not resolve malicious-ranking columns")
    seed_col, client_col, rank_col = seed_cols[0], client_cols[0], rank_cols[0]
    selected = [r for r in rows if int(r[seed_col]) == seed]
    selected.sort(key=lambda r: int(r[rank_col]))
    require(len(selected) >= prefix, "Malicious ranking shorter than prefix")
    return {int(r[client_col]) for r in selected[:prefix]}


def run_one(binding: dict[str, Any], run_index: int, resume: bool, authorization: Path) -> None:
    auth = read_json(authorization)
    require(auth.get("authorization_id") == AUTHORIZATION_ID, "Invalid Gate-107 authorization ID")
    require(auth.get("runner_sha256") == binding["runner_sha256"], "Authorization runner SHA mismatch")
    require(auth.get("run_index") == run_index, "Authorization run index mismatch")
    require(auth.get("one_time") is True, "Authorization is not one-time")

    matrix = read_csv(Path(binding["paths"]["campaign_matrix"]))
    require(1 <= run_index <= len(matrix), f"run-index out of range 1..{len(matrix)}")
    row = matrix[run_index - 1]
    require(str(row["MainCampaign"]).strip().lower() in {"1", "true", "yes"}, "Selected row is not MainCampaign")
    config_id = row["ConfigID"]
    seed = int(row["ExperimentalSeed"])
    method_id = row["MethodID"]
    scenario_id = row["ScenarioID"]

    run_name = f"run_{run_index:03d}__{config_id}__{method_id}__{scenario_id}"
    run_root = Path(binding["paths"]["scientific_output_root"]) / run_name
    completed = run_root / "RUN_COMPLETE.json"
    require(not completed.exists(), f"Completed run already exists: {completed}")
    if run_root.exists() and not resume:
        raise RuntimeError(f"Partial run exists; use --resume only after Gate-107 review: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "runner_id": RUNNER_ID,
        "runner_sha256": binding["runner_sha256"],
        "build_binding_sha256": binding["build_binding_sha256"],
        "run_index": run_index,
        "config_id": config_id,
        "experimental_seed": seed,
        "method_id": method_id,
        "scenario_id": scenario_id,
        "scientific_run": True,
    }
    atomic_json(run_root / "RUN_CONTRACT.json", run_contract)

    scenario_rows = read_csv(Path(binding["paths"]["scenario_plan"]))
    scenario = next(r for r in scenario_rows if r["ScenarioID"] == scenario_id)
    malicious_clients = resolve_malicious_clients(binding, seed, scenario)
    attack_active_rounds = parse_round_set(scenario["AttackActiveRounds"])

    schedule_rows = [r for r in read_csv(Path(binding["paths"]["canonical_schedule"])) if int(r["ExperimentalSeed"]) == seed]
    schedule_by_round = defaultdict(list)
    for r in schedule_rows:
        schedule_by_round[int(r["Round"])].append(r)
    for r in schedule_by_round.values():
        r.sort(key=lambda x: int(x["Slot"]))

    local_order_rows = read_csv(Path(binding["paths"]["local_order_seeds"]))
    local_order_seed = {(r["ConfigID"], int(r["ClientID"])): int(r["SeedUInt64"]) for r in local_order_rows}
    boundary_rows = read_csv(Path(binding["paths"]["block_boundaries"]))
    block_boundary = {(r["ConfigID"], int(r["ClientID"]), int(r["ParticipationOrdinal"])): (int(r["StartInclusive"]), int(r["EndExclusive"])) for r in boundary_rows}
    rng_rows = read_csv(Path(binding["paths"]["local_rng_application"]))
    event_rng = {(int(r["ExperimentalSeed"]), int(r["Round"]), int(r["ClientID"])): (int(r["ParticipationOrdinal"]), int(r["TorchManualSeedInt64"])) for r in rng_rows}
    global_seed_rows = read_csv(Path(binding["paths"]["global_init_seeds"]))
    global_seed_u64 = int(next(r["SeedUInt64"] for r in global_seed_rows if int(r["ExperimentalSeed"]) == seed))

    data = FrozenDataSurface(binding, config_id)
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    kernels = load_method_kernels(binding, method_id)
    fedprox_module = import_module_exact(Path(binding["paths"]["fedprox_rebound_adapter"]), "scientific_fedprox_rebound")
    fedprox_fn = getattr(fedprox_module, "train_one_client_fedprox_rebound")

    checkpoint = run_root / "ROUND_CHECKPOINT.pt"
    metrics_csv = run_root / "ROUND_VALIDATION_METRICS.csv"
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        global_state = payload["global_state"]
        method_state = payload.get("method_state")
        best = payload.get("best")
        start_round = int(payload["completed_round"]) + 1
    else:
        torch.manual_seed(global_seed_u64 % ((1 << 63) - 1))
        global_state = cpu_state_dict(ScientificMLP())
        method_state = None
        best = None
        start_round = 1

    for round_number in range(start_round, 21):
        selected = [int(r["ClientID"]) for r in schedule_by_round[round_number]]
        require(len(selected) == 9 and len(set(selected)) == 9, f"Invalid selected clients round {round_number}")
        local_states: list[dict[str, torch.Tensor]] = []
        local_losses: list[float] = []
        sample_counts: list[int] = []
        for client_id in selected:
            ordinal, torch_seed = event_rng[(seed, round_number, client_id)]
            start, end = block_boundary[(config_id, client_id, ordinal)]
            x_block, y_block_u8 = data.materialize_client_block(client_id, local_order_seed[(config_id, client_id)], start, end)
            if method_id == "FEDPROX":
                result = fedprox_fn(
                    model=ScientificMLP(),
                    global_state=copy.deepcopy(global_state),
                    x_block=torch.from_numpy(x_block),
                    y_block=torch.from_numpy(y_block_u8.astype(np.int64, copy=False)),
                    class_weights=class_weights,
                    local_seed=int(torch_seed),
                    technical_audit=None,
                )
            else:
                result = common_local_train(global_state, x_block, y_block_u8.astype(np.int64, copy=False), class_weights, int(torch_seed))
            local_state = clone_state(result["state_dict"])
            attacked = apply_attack(global_state, local_state, client_id in malicious_clients, round_number in attack_active_rounds)
            local_states.append(attacked)
            local_losses.append(float(result["train_loss"]))
            sample_counts.append(int(result.get("total_examples", 51200)))

        if method_id in {"FEDAVG", "FEDPROX"}:
            new_global = fedavg_state(local_states, sample_counts)
            aux = None
        else:
            context = build_common_context(
                global_state=global_state,
                local_states=local_states,
                local_losses=local_losses,
                selected_clients=selected,
                sample_counts=sample_counts,
                round_number=round_number,
                method_state=method_state,
            )
            if kernels["state_update"] is not None:
                method_state = invoke_with_context(kernels["state_update"], context)
                context.update({"state": method_state, "method_state": method_state, "trust_state": method_state, "risk_state": method_state})
            value = invoke_with_context(kernels["aggregation"], context)
            new_global, aux = normalize_state_result(value, set(global_state.keys()))
            if aux is not None:
                method_state = {"state": method_state, "aggregation_aux": aux}

        require(state_is_finite(new_global), f"Non-finite global state after round {round_number}")
        global_state = new_global
        val = evaluate_state(global_state, data, int(binding["split_roles"]["VALIDATION"]))
        row_metrics = {"Round": round_number, **val}
        append_csv(metrics_csv, ["Round", "loss", "accuracy", "macro_f1", "balanced_accuracy"], row_metrics)
        if better_checkpoint(val, best, round_number):
            best = {**val, "round": round_number}
            torch.save(clone_state(global_state), run_root / "BEST_STATE.pt")
        torch.save({
            "global_state": clone_state(global_state),
            "method_state": method_state,
            "best": best,
            "completed_round": round_number,
        }, checkpoint)
        atomic_json(run_root / "RUN_STATE.json", {"status": "IN_PROGRESS", "completed_round": round_number, "best": best})

    best_state = torch.load(run_root / "BEST_STATE.pt", map_location="cpu")
    test_metrics = evaluate_state(best_state, data, int(binding["split_roles"]["TEST"]))
    complete = {
        "status": "SCIENTIFIC_RUN_COMPLETE",
        "run_contract": run_contract,
        "best_validation": best,
        "test_metrics": test_metrics,
        "scientific_optimizer_steps_expected": 20 * 9 * 200,
        "scientific_training_started": True,
        "scientific_metrics_computed": True,
    }
    complete["run_result_binding_sha256"] = canonical_sha(complete)
    atomic_json(completed, complete)
    print(json.dumps(complete, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-json", default=str(Path(__file__).with_name("SCIENTIFIC_CAMPAIGN_RUNNER_BINDING.json")))
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--authorization")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    binding_path = Path(args.binding_json).resolve()
    binding = read_json(binding_path)
    runner_path = Path(__file__).resolve()
    require(sha256_file(runner_path) == binding["runner_sha256"], "Runner self-hash mismatch")
    require(binding["runner_id"] == RUNNER_ID, "Runner ID mismatch")

    if args.contract_check:
        require(args.run_index is None and args.authorization is None, "Contract check must not include run authorization")
        report = contract_check(binding)
        print(json.dumps(report, indent=2))
        return 0

    require(args.run_index is not None, "Scientific execution requires --run-index")
    require(args.authorization is not None, "Scientific execution requires Gate-107 --authorization")
    run_one(binding, args.run_index, args.resume, Path(args.authorization).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
