from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG = PROJECT_ROOT / "config" / "active_experiment.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else ACTIVE_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_agent_cfg(cfg: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    agents = cfg.get("agents", {})
    return agents.get(str(agent_id), agents.get(agent_id, {}))


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_run_dir(cfg: Dict[str, Any], agent_id: int | None = None) -> Path:
    raw = cfg.get("experiment", {}).get("run_dir", "results/run")
    root = project_path(raw) or (PROJECT_ROOT / "results" / "run")
    if agent_id is None:
        out = root
    else:
        out = root / f"agent_{agent_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def send_json(emitter, payload: Dict[str, Any]) -> None:
    emitter.send(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def recv_json(receiver):
    if receiver.getQueueLength() <= 0:
        return None
    text = receiver.getString()
    receiver.nextPacket()
    try:
        return json.loads(text)
    except Exception:
        return {"type": "legacy_text", "value": text}


def drain_json(receiver):
    messages = []
    while receiver.getQueueLength() > 0:
        msg = recv_json(receiver)
        if msg is not None:
            messages.append(msg)
    return messages


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def runtime_manifest(cfg: Dict[str, Any], agent_id: int, algorithm: str | None = None) -> Dict[str, Any]:
    versions = {}
    for name in ["numpy", "gymnasium", "stable_baselines3", "sb3_contrib", "torch", "pandas", "scipy"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return {
        "created_unix": time.time(),
        "agent_id": agent_id,
        "algorithm": algorithm,
        "experiment": cfg.get("experiment", {}),
        "arena": cfg.get("arena", {}),
        "robot_design": cfg.get("robot_design", {}),
        "reward_design": cfg.get("reward_design", {}),
        "drive": cfg.get("drive", {}),
        "action_mapping_v112": {
            "0": "forward_low", "1": "forward_medium", "2": "forward_high",
            "3": "backward_low", "4": "backward_medium", "5": "backward_high",
            "6": "left_low", "7": "left_medium", "8": "left_high",
            "9": "right_low", "10": "right_medium", "11": "right_high",
            "12": "spin_left_low", "13": "spin_left_medium", "14": "spin_left_high",
            "15": "spin_right_low", "16": "spin_right_medium", "17": "spin_right_high",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "model_files_sha256": {
            "worlds/sumo_research.wbt": _sha256(PROJECT_ROOT / "worlds" / "sumo_research.wbt"),
            "protos/SumoRobot2.proto": _sha256(PROJECT_ROOT / "protos" / "SumoRobot2.proto"),
            "config/base_experiment.yaml": _sha256(PROJECT_ROOT / "config" / "base_experiment.yaml"),
        },
    }
