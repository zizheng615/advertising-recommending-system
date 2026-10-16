"""脚本共用的骨架：参数解析、产物加载、日志、指标落盘。

**为什么脚本之间用文件而不是 import 传递产物**

每个 ``scripts/NN_*.py`` 都能**单独重跑**。这意味着调试某一环时
不必从头把整条链路（生成 → 特征 → 训练三个模型 → 出价 → 实验）
再跑一遍 —— 在 CPU 上那是十几分钟的事。

代价是「必须按顺序跑」，所以 ``load_*`` 在文件缺失时会抛出
**指明该先跑哪个脚本**的错误，而不是一个 FileNotFoundError。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import ARTIFACTS, get_config
from src.common.io import (get_logger, load_json, load_pickle, save_json,
                           save_pickle)

# --------------------------------------------------------------------------
# 控制台编码 —— 一次修掉三类故障，其中一类是**静默**的
# --------------------------------------------------------------------------
#
# ⚠️ Windows 上 Python 的 stdout 默认是 GBK（cp936），于是：
#
#   * ``print("σ²/2")``  → 直接抛 UnicodeEncodeError，而报错位置在
#     ``show_table`` 里，看着像输出层的毛病，其实是编码；
#   * ``log.info("σ²/2")`` → **不抛**。``logging.StreamHandler.emit`` 捕获
#     了 UnicodeEncodeError，往 stderr 打一段 "--- Logging error ---"，
#     然后把**整行日志丢掉**。脚本照常跑到「体检通过」，只是输出里少了
#     几行 —— 没有人会注意到。实测 ``scripts/05_ziln.py`` 就因为
#     ``²`` 丢过日志行；
#   * 终端里的中文变成乱码。
#
# 实测 GBK 编不出来的字符：``² Ê ̂ −(U+2212) ⇒ ⚠ ✅ ❌`` 以及变体选择符
# U+FE0F —— 而这些都是本项目注释和输出里天天用的符号。
#
# 换成 UTF-8 一次解决三件事。pytest 会把 ``sys.stdout`` 换成捕获对象
# （没有 ``reconfigure``），所以必须做存在性判断。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):       # pragma: no cover - 被包装过的流
            pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:               # 允许直接 python scripts/xx.py
    sys.path.insert(0, str(ROOT))

DATA_DIR = ARTIFACTS / "data"
MODEL_DIR = ARTIFACTS / "models"
METRIC_DIR = ARTIFACTS / "metrics"
FIG_DIR = ARTIFACTS / "figures"

log = get_logger("adrec")


# --------------------------------------------------------------------------
# 参数与产物
# --------------------------------------------------------------------------


def parse_scale(description: str) -> tuple[str, str]:
    """解析 ``--scale``，返回 ``(scale, description)``。

    规模档只允许三个值，且**没有默认值以外的捷径**：
    写错一个字母就静默退回 small 的话，「我明明跑了 full」
    这类问题会拖到看 README 数字时才发现。
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--scale", default="small",
                    choices=["mini", "small", "full"],
                    help="规模档：mini（秒级）/ small（默认）/ full（10 万用户）")
    args = ap.parse_args()
    return args.scale, description


def cfg_for(scale: str):
    return get_config(scale)


def _require(path: Path, hint: str):
    if not path.exists():
        raise SystemExit(
            f"缺少产物 {path}。\n先跑：python {hint}")
    return path


def load_data(cfg) -> dict:
    """原始模拟数据（users / items / impressions）。"""
    return load_pickle(_require(_scaled(DATA_DIR / "raw.pkl", cfg),
                                "scripts/01_generate.py"))


def load_pca(cfg):
    return load_pickle(_require(_scaled(MODEL_DIR / "pca.pkl", cfg),
                                "scripts/02_features.py"))


def load_kmeans(cfg):
    return load_pickle(_require(_scaled(MODEL_DIR / "kmeans.pkl", cfg),
                                "scripts/02_features.py"))


def load_bundle(cfg):
    return load_pickle(_require(_scaled(DATA_DIR / "bundle.pkl", cfg),
                                "scripts/02_features.py"))


def load_ranker_tensors(cfg) -> dict:
    # ⚠️ 这份张量由 **03** 落盘（02 只产出 bundle / pca / kmeans），
    # 提示里写错脚本号会让人对着一个不存在的产出找半天。
    return load_pickle(_require(_scaled(DATA_DIR / "ranker_tensors.pkl", cfg),
                                "scripts/03_train_ranker.py"))


def load_split(cfg):
    return load_pickle(_require(_scaled(DATA_DIR / "split.pkl", cfg),
                                "scripts/01_generate.py"))


def load_metrics(name: str, cfg=None) -> dict:
    path = METRIC_DIR / (f"{name}.json" if cfg is None
                         else f"{name}_{cfg.scale}.json")
    return load_json(_require(path, "run_all.py"))


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------


def _scaled(path: Path, cfg) -> Path:
    """按规模档给文件名加后缀。

    ⚠️ 不加后缀的话，跑完 ``--scale mini`` 再跑 ``--scale full``
       会把小规模的产物覆盖掉，而后面读到的 pkl 和张量**维度对不上**，
       报出来的错会是「矩阵乘法维度不匹配」这种和真实原因无关的信息。
    """
    return path.with_name(f"{path.stem}_{cfg.scale}{path.suffix}")


def save_artifact(obj, path: Path, cfg) -> Path:
    p = _scaled(path, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    return save_pickle(obj, p)


def save_metrics(name: str, payload: dict, cfg) -> Path:
    p = METRIC_DIR / f"{name}_{cfg.scale}.json"
    save_json(payload, p)
    log.info("指标已落盘 %s", p.name)
    return p


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


class Timer:
    """给每个脚本一个「跑了多久」的交代。

    单纯为了输出好看，但 CPU 上跑全链路时这个数很有用 ——
    改动某一步后能立刻看出它是不是变慢了一个量级。
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        log.info("=== %s ===", self.name)
        return self

    def __exit__(self, *exc):
        log.info("=== %s 用时 %.1fs ===", self.name, time.time() - self.t0)
        return False


def banner(text: str) -> None:
    log.info("-" * 68)
    log.info("%s", text)
    log.info("-" * 68)


def show_table(rows: list[dict], keys: list[str], title: str = "") -> None:
    """把一串字典打成对齐的文本表。

    比 ``print(dict)`` 有用得多：这些表会被直接抄进 README，
    所以列宽必须稳定、数字必须能一眼横着比。
    """
    if title:
        banner(title)
    if not rows:
        return
    widths = {k: max(len(k), *(len(_fmt(r.get(k))) for r in rows))
              for k in keys}
    print("  " + "  ".join(k.rjust(widths[k]) for k in keys))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(k)).rjust(widths[k]) for k in keys))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def dump_json_stdout(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
