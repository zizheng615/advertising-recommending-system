"""金额 / 分布类指标 —— 专门为 ZILN 的零膨胀场景设计。

为什么不能只看 MAE
------------------
零膨胀下 90% 的目标是 0，「全预测 0」的 MAE 已经接近最优，
任何模型都能刷出好看的数字。所以要换成：

    PIT 均匀性   —— 直接检验「对数正态假设是否成立」
    聚合偏差     —— 直接检验「出价要用的期望金额准不准」
    skill score  —— 相对「预测训练集均值」这个基线，模型到底赢了多少
    条件 log-MAE —— 只在正样本上看金额精度，绕开零主导

⚠️ 期望金额的定义必须是 ``Ê[y] = p̂ · exp(μ̂ + σ̂²/2)``。
   写成 ``p̂ · exp(μ̂)`` 得到的是条件**中位数**，会系统性低估 ——
   聚合偏差这个指标就是专门用来抓这个 bug 的。
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def conditional_log_mae(y_true_log: np.ndarray,
                        y_pred_log: np.ndarray) -> float:
    """正样本上的对数尺度 MAE（越小越好）。"""
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    if len(y_true_log) == 0:
        return float("nan")
    return float(np.abs(y_pred_log - y_true_log).mean())


def log_mae_skill(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """相对「预测训练集均值」基线的 skill score = 1 − MAE/MAE_base。

    > 0 表示比瞎猜均值强。零膨胀下普通 MAE 会被 0 主导，
    所以这个 skill 要算在对数尺度上。
    """
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    if len(y_true_log) == 0:
        return 0.0
    base = np.abs(y_true_log - y_true_log.mean()).mean()
    if base < 1e-12:
        return 0.0
    return float(1.0 - conditional_log_mae(y_true_log, y_pred_log) / base)


def pit_values(y_log: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """PIT = Φ((ln y − μ)/σ)。

    如果「正样本金额服从对数正态」这个假设成立，PIT 应当服从 U(0,1)。
    这一个检验同时验证了**生成器**和**损失函数**两边，
    是性价比最高的一条断言。
    """
    y_log = np.asarray(y_log, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-6, None)
    return stats.norm.cdf((y_log - mu) / sigma)


def pit_uniformity_pvalue(y_log: np.ndarray, mu: np.ndarray,
                          sigma: np.ndarray) -> float:
    """PIT 的 KS 检验 p 值。p > 0.05 表示不能拒绝「PIT 均匀」。"""
    pit = pit_values(y_log, mu, sigma)
    if len(pit) < 5:
        return float("nan")
    return float(stats.kstest(pit, "uniform").pvalue)


def expected_value(p_pay: np.ndarray, mu: np.ndarray,
                   sigma: np.ndarray) -> np.ndarray:
    """Ê[y] = p̂ · exp(μ̂ + σ̂²/2)。

    ⚠️ 这是**条件期望**，含零膨胀。写成 ``p̂ · exp(μ̂)`` 是条件中位数，
    会系统性低估，而且 eCPM 会跟着一起错。
    """
    p_pay = np.asarray(p_pay, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return (p_pay * np.exp(mu + sigma ** 2 / 2.0)).astype(np.float64)


def aggregate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """|Σ预测 − Σ真实| / Σ真实 —— 出价最关心的那个数。

    排序指标全对、但这个数偏了 4 倍的系统，上线就是灾难。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = abs(y_true.sum())
    if denom < 1e-12:
        return 0.0
    return float(abs(y_pred.sum() - y_true.sum()) / denom)


def decile_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                       n_bins: int = 10) -> float:
    """按预测值分桶，返回各桶「相对误差」的最大值。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < n_bins:
        return 0.0
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(y_pred, edges[1:-1])
    worst = 0.0
    for b in np.unique(bins):
        m = bins == b
        if m.sum() == 0:
            continue
        actual = y_true[m].mean()
        pred = y_pred[m].mean()
        denom = max(abs(actual), 1e-6)
        worst = max(worst, abs(pred - actual) / denom)
    return float(worst)


def gini(y: np.ndarray) -> float:
    """基尼系数（排序能力的另一种度量，对长尾比 AUC 更敏感）。"""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0 or y.sum() <= 0:
        return 0.0
    y_sorted = np.sort(y)
    cum = np.cumsum(y_sorted)
    # 洛伦兹曲线下面积 → 基尼
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """模型预测的 Gini / 完美预测的 Gini。

    > 0.30 才算有实用价值。零膨胀场景下它比 AUC 更能反映
    「有没有把高价值用户排上来」。
    """
    g_pred = gini(np.asarray(y_pred, dtype=np.float64))
    g_perfect = gini(np.asarray(y_true, dtype=np.float64))
    if abs(g_perfect) < 1e-12:
        return 0.0
    return float(g_pred / g_perfect)


def lognormal_nll(y_log: np.ndarray, mu: np.ndarray,
                  sigma: np.ndarray) -> float:
    """正样本的对数正态负对数似然（不含 ln y 项，因为它是常数）。"""
    y_log = np.asarray(y_log, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-6, None)
    if len(y_log) == 0:
        return float("nan")
    return float((np.log(sigma) + 0.5 * np.log(2 * np.pi)
                  + (y_log - mu) ** 2 / (2 * sigma ** 2)).mean())
