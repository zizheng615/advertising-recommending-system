"""落盘与日志小工具。

⚠️ 本机 **没有 pyarrow**，所以不能存 parquet —— 统一用 pickle。
   DataFrame 从生成那一刻就用 float32：192 列 × 300 万行，
   float64 是 5.2GB，float32 只要 2.6GB。
⚠️ C 盘只剩不到 1GB，所有产物必须落在 D 盘的 artifacts/。
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config import ARTIFACTS

_LOG_CONFIGURED = False


def get_logger(name: str = "adrec", level: int = logging.INFO) -> logging.Logger:
    global _LOG_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
        root = logging.getLogger("adrec")
        root.addHandler(handler)
        root.setLevel(level)
        _LOG_CONFIGURED = True
    return logger


def save_pickle(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: str | Path) -> Any:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"不可序列化类型 {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)
    return path


def load_json(path: str | Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def artifacts_path(*parts: str) -> Path:
    p = ARTIFACTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
