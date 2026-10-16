"""按时间切分 train / valid / test。

为什么必须按时间切而不是随机切
------------------------------
随机切分会让同一个用户的曝光同时出现在训练和测试集里，模型记住了
用户 ID 就等于作弊。真实场景永远是「用过去预测未来」，所以切分必须
沿时间轴切。这也让 test 集上的 AUC 天然低于 valid（分布漂移），
正好解释为什么模型需要定期重训。

⚠️ SMOTE / 负采样等任何采样操作都必须在切分**之后**、且只在 train 上做。
   跨切分做近邻插值 = 造出未来样本的近邻 = 时序泄漏。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config


@dataclass
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:
        return (f"Split(train={len(self.train):,}, valid={len(self.valid):,}, "
                f"test={len(self.test):,})")


def time_split(imp: pd.DataFrame, cfg: Config,
               train_frac: float = 0.70, valid_frac: float = 0.15) -> Split:
    """按 day 列切分，返回三组的**行索引**。

    同一天的所有曝光必须整组落在同一侧，否则同一天内部的信息会跨切分泄漏。
    """
    n_days = int(imp["day"].max()) + 1
    n_train = max(1, int(round(n_days * train_frac)))
    n_valid = max(1, int(round(n_days * valid_frac)))
    train_end = n_train
    valid_end = min(n_days, n_train + n_valid)

    day = imp["day"].to_numpy()
    tr = np.flatnonzero(day < train_end)
    va = np.flatnonzero((day >= train_end) & (day < valid_end))
    te = np.flatnonzero(day >= valid_end)

    # 极端小数据集下保证三组都非空
    if len(va) == 0:
        va = tr[-max(1, len(tr) // 10):]
        tr = tr[:len(tr) - len(va)]
    if len(te) == 0:
        te = va
    return Split(tr, va, te)


def label_stats(imp: pd.DataFrame, idx: np.ndarray, col: str = "y_conv") -> dict:
    """切分后的标签统计，用于体检数据是否失衡或退化。"""
    sub = imp.iloc[idx]
    n_click = int(sub["y_click"].sum())
    n_conv = int(sub[col].sum())
    return {
        "n": int(len(sub)),
        "n_click": n_click,
        "n_conv": n_conv,
        "ctr": float(sub["y_click"].mean()) if len(sub) else 0.0,
        "cvr": float(n_conv / n_click) if n_click else 0.0,
        "ctcvr": float(sub[col].mean()) if len(sub) else 0.0,
    }
