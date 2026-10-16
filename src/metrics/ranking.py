"""排序 / 概率校准指标。

统一在这里实现，是为了让「同一件事只算一次」——
训练脚本、评测脚本、测试断言都调这里的函数，
不会出现「脚本里算的 AUC 和测试里算的对不上」这种事。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score as _sk_auc


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC，但单类别时返回 0.5 而不是抛异常。

    测试和边界场景（某个切分里一个转化都没有）很容易踩到，
    让它返回一个中性的 0.5 比让整条流水线崩掉更合理。
    """
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    return float(_sk_auc(y_true, y_score))


def gauc(y_true: np.ndarray, y_score: np.ndarray,
         groups: np.ndarray, min_size: int = 10) -> float:
    """按用户分组的 AUC 加权平均（Group AUC）。

    全局 AUC 会被「高活跃用户曝光多」这件事主导；GAUC 先算每个用户
    自己的排序质量再加权，更贴近线上「给同一个用户排序」的真实场景。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    boundaries = np.flatnonzero(np.diff(g_sorted)) + 1
    splits = np.split(order, boundaries)

    num, den = 0.0, 0.0
    for idx in splits:
        if len(idx) < min_size:
            continue
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        num += safe_auc(yt, y_score[idx]) * len(idx)
        den += len(idx)
    return float(num / den) if den > 0 else 0.5


def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS 统计量 = max|TPR − FPR|，风控/广告里常用的区分度指标。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    grid = np.unique(y_score)
    tpr = np.searchsorted(np.sort(pos), grid, side="right") / len(pos)
    fpr = np.searchsorted(np.sort(neg), grid, side="right") / len(neg)
    return float(np.max(np.abs(tpr - fpr)))


def logloss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE：把预测概率分桶，比较每桶的平均预测与真实频率。

    ⚠️ 这是最能抓「先验校正漏做」的指标 —— 1:5 负下采样后忘了把 logit
    平移回去，AUC 完全不受影响（排序没变），但 ECE 会从 0.02 飙到 0.3+。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return float(abs(p.mean() - y_true.mean()))
    bins = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, p: np.ndarray,
                      n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (每桶平均预测, 每桶真实频率, 每桶样本数)，用于画可靠性图。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(p, edges[1:-1])
    xs, ys, ns = [], [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y_true[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def prior_corrected_prob(p_sample: np.ndarray, neg_keep_ratio: float) -> np.ndarray:
    """把「负下采样后训练出的概率」还原成真实先验下的概率。

    在 logit 上直接加 ``ln β``（β = 负样本保留比例），一行代码。
    漏掉这一步会让 eCPM 高估约 4 倍，而 AUC 完全看不出来。
    """
    p = np.clip(np.asarray(p_sample, dtype=np.float64), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p)) + np.log(neg_keep_ratio)
    return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
