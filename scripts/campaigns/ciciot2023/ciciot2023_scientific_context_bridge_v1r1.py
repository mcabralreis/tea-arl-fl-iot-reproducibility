from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SOURCE_MAP_PATH = Path(__file__).with_name("BRIDGE_SOURCE_MAP.json")
SOURCE_MAP = json.loads(SOURCE_MAP_PATH.read_text(encoding="utf-8-sig"))
_CACHE = {}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def _load(tag: str):
    item = SOURCE_MAP[tag]
    path = Path(item["path"])
    observed = _sha(path)
    if observed != item["sha256"]:
        raise RuntimeError(f"Frozen source SHA mismatch for {tag}: {observed}")
    key = (str(path), observed)
    module = _CACHE.get(key)
    if module is None:
        name = "ciciot_bridge_" + tag.lower() + "_" + observed[:12].lower()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _CACHE[key] = module
    return module, getattr(module, item["callable"])


def _cid(value) -> str:
    text = str(value)
    if text.startswith("client_"):
        return text
    try:
        return f"client_{int(text):02d}"
    except Exception:
        return text


def _as_list(value, n: int, name: str):
    result = list(value.tolist()) if hasattr(value, "tolist") else list(value)
    if len(result) != n:
        raise RuntimeError(
            f"{name} cardinality mismatch: expected {n}, observed {len(result)}"
        )
    return result


def _results(local_states, selected_client_ids, sample_counts, local_losses):
    states = list(local_states)
    ids = _as_list(selected_client_ids, len(states), "selected_client_ids")
    counts = _as_list(sample_counts, len(states), "sample_counts")
    losses = _as_list(local_losses, len(states), "local_losses")
    return [
        SimpleNamespace(
            client_id=_cid(client_id),
            windows=int(windows),
            train_loss=float(loss),
            train_macro_f1=float("nan"),
            wall_seconds=0.0,
            state_dict=state,
        )
        for client_id, windows, loss, state
        in zip(ids, counts, losses, states)
    ]


def _clients(client_ids, selected_client_ids):
    source = client_ids if client_ids is not None else selected_client_ids
    values = list(source.tolist()) if hasattr(source, "tolist") else list(source)
    return [_cid(value) for value in values]


def _state(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise RuntimeError(f"method_state must be dict or None, got {type(value)}")


def _extract(value):
    if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
        return value
    if isinstance(value, (tuple, list)):
        candidates = [
            item for item in value
            if isinstance(item, dict)
            and item
            and all(torch.is_tensor(v) for v in item.values())
        ]
        if len(candidates) == 1:
            return candidates[0]
    raise RuntimeError("Could not extract one tensor state_dict")


def bridge_random_trimmed_aggregation(
    *, global_state, local_states, selected_client_ids, sample_counts, local_losses
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    module, function = _load("RANDOM_TRIMMED")
    original = getattr(module, "CLIENTS_PER_ROUND")
    setattr(module, "CLIENTS_PER_ROUND", len(results))
    try:
        output = function(global_state=global_state, local_results=results)
    finally:
        setattr(module, "CLIENTS_PER_ROUND", original)
    return _extract(output)


def bridge_fedle_aggregation(
    *, global_state, local_states, selected_client_ids, sample_counts, local_losses
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    _module, function = _load("FEDLE")
    return _extract(function(global_state=global_state, local_results=results))


def bridge_tea_state_update(
    *, round_number, client_ids, selected_client_ids, local_states,
    sample_counts, local_losses, method_state
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    selected = [result.client_id for result in results]
    all_clients = _clients(client_ids, selected_client_ids)
    state = _state(method_state)
    trust = state.get("trust")
    if trust is None:
        trust = {client_id: 0.5 for client_id in all_clients}
    else:
        trust = {_cid(k): float(v) for k, v in trust.items()}
        for client_id in all_clients:
            trust.setdefault(client_id, 0.5)
    _module, function = _load("TEA_UPDATE")
    output = function(
        round_index=int(round_number),
        selected=selected,
        local_results=results,
        trust=trust,
    )
    candidates = [
        item for item in output
        if isinstance(item, dict)
        and item
        and all(isinstance(v, (int, float, np.floating)) for v in item.values())
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"TEA updated-trust candidate count: {len(candidates)}")
    state["trust"] = {_cid(k): float(v) for k, v in candidates[0].items()}
    state["last_state_update_round"] = int(round_number)
    return state


def bridge_tea_aggregation(
    *, global_state, client_ids, selected_client_ids, local_states,
    sample_counts, local_losses, method_state
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    state = _state(method_state)
    trust = state.get("trust")
    if not isinstance(trust, dict):
        raise RuntimeError("TEA aggregation missing updated trust")
    residual_energy = {
        client_id: 1.0 for client_id in _clients(client_ids, selected_client_ids)
    }
    _module, function = _load("TEA_AGG")
    return _extract(function(
        global_state=global_state,
        local_results=results,
        trust=trust,
        residual_energy=residual_energy,
    ))


def bridge_arl_state_update(
    *, round_number, client_ids, selected_client_ids, global_state,
    local_states, sample_counts, local_losses, method_state
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    selected = [result.client_id for result in results]
    all_clients = _clients(client_ids, selected_client_ids)
    state = _state(method_state)
    risk_state = state.get("risk_state")
    if risk_state is None:
        risk_state = {client_id: 0.0 for client_id in all_clients}
    else:
        risk_state = {_cid(k): float(v) for k, v in risk_state.items()}
        for client_id in all_clients:
            risk_state.setdefault(client_id, 0.0)
    previous_update = state.get("previous_update", {})
    _module, function = _load("ARL_UPDATE")
    output = function(
        round_index=int(round_number),
        selected=selected,
        global_state=global_state,
        local_results=results,
        previous_update=previous_update,
        risk_state=risk_state,
    )
    if not isinstance(output, tuple) or len(output) < 5:
        raise RuntimeError("ARL state update did not return five outputs")
    state["risk_state"] = {_cid(k): float(v) for k, v in output[1].items()}
    state["previous_update"] = output[2]
    state["global_pressure"] = float(output[3])
    state["current_updates"] = output[4]
    state["last_state_update_round"] = int(round_number)
    return state


def bridge_arl_aggregation(
    *, round_number, global_state, selected_client_ids, local_states,
    sample_counts, local_losses, method_state
):
    results = _results(local_states, selected_client_ids, sample_counts, local_losses)
    state = _state(method_state)
    current_updates = state.get("current_updates")
    global_pressure = state.get("global_pressure")
    if not isinstance(current_updates, dict):
        raise RuntimeError("ARL aggregation missing current_updates")
    if not isinstance(global_pressure, (int, float, np.floating)):
        raise RuntimeError("ARL aggregation missing global_pressure")
    _module, function = _load("ARL_AGG")
    return _extract(function(
        round_index=int(round_number),
        global_state=global_state,
        local_results=results,
        current_updates=current_updates,
        global_pressure=float(global_pressure),
    ))
