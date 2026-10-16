"""CUPED：用实验**前**的协变量把方差消掉一块。

原理只有一行。取任意**实验前**就能观测到的协变量 ``X``，令

    Y_adj = Y − θ·(X − X̄),      θ = Cov(Y, X) / Var(X)

因为 ``E[X_adj] = E[X]``（中心化保证），所以 ``E[Y_adj] = E[Y]`` ——
**处理效应的点估计完全不变**，变的只有方差：

    Var(Y_adj) = Var(Y)·(1 − ρ²)

``ρ`` 是 Y 与 X 的相关系数，所以 ρ=0.5 时方差降 25%、
等价于样本量白涨 33%。这就是 CUPED 的全部价值：
**在不改变结论含义的前提下把实验做短**。

**三条纪律，踩错任何一条都会得出错误结论**

1. **协变量必须来自实验之前**。这是唯一不可让步的一条。用实验期间
   或实验后的量做协变量（比如「实验期前 3 天的转化」）
   会把处理效应本身吸收进 ``X``，于是效应被「调整掉」——
   实验明明是正的，调整完变成 0。这个错误不会报错，
   只会让一个有效的功能上线不了。本模块的 ``leakage_warning`` 会检查
   协变量与分组的独立性，但那只是辅助，**根本的保证来自调用方
   把 X 取在分流之前**。
2. **``θ`` 要估得稳**。``θ`` 用样本估计，样本小的时候 θ 自身的噪声
   会让方差**变大**（净效果 = ``1 − ρ² + 估计误差``）。
   所以设了 ``min_rho`` 门槛：协变量太弱（``|ρ| < 0.3``）就干脆不调，
   返回原值。ρ 在 0.3 时理论收益 9%，扣掉估计误差后所剩无几。
3. **回归调整 ≠ 分层**。CUPED 减的是**方差**，
   它不修正「两组本来就不一样」的系统性差异 ——
   那是随机化该保证的事（用 AA 检验和 SRM 去验），
   不是 CUPED 的职责。指望 CUPED 去救一个分流坏掉的实验是错的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CupedResult:
    """调整结果 + 诊断量。``variance_ratio`` 是调整后/调整前的方差比。"""

    y_adj: np.ndarray
    theta: float
    rho: float
    variance_ratio: float
    applied: bool

    @property
    def variance_reduction(self) -> float:
        """方差下降比例（``0.25`` 表示降了 25%）。"""
        return 1.0 - self.variance_ratio

    @property
    def effective_sample_multiplier(self) -> float:
        """等价于样本量放大多少倍（``1/(1−ρ²)``）。"""
        return 1.0 / self.variance_ratio if self.variance_ratio > 0 else float("nan")


def cuped_adjust(y, x, *, min_rho: float = 0.30) -> CupedResult:
    """对 ``y`` 做 CUPED 调整。``x`` 必须是**实验前**的协变量。

    ``y`` 可以是用户级的任意可加量（收入、转化次数、时长）。
    比率指标（CVR、ROI）不能直接这么调 —— 要先把分子做 CUPED 调整，
    再走 ``stats.ratio_delta_test``；本项目的 ``runner`` 就是这么做的。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size != x.size:
        raise ValueError("协变量与指标的用户数必须一致")
    if y.size < 3:
        return CupedResult(y, 0.0, float("nan"), 1.0, applied=False)

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    if var_x <= 0 or var_y <= 0:
        # 协变量是常数（或指标是常数）：θ 无法定义。
        # 这里不能返回 θ=0 就走人 —— 常数协变量往往意味着取错了列。
        return CupedResult(y, 0.0, float("nan"), 1.0, applied=False)

    cov = float(np.cov(y, x, ddof=1)[0, 1])
    rho = cov / np.sqrt(var_x * var_y)
    if abs(rho) < min_rho:
        return CupedResult(y, 0.0, float(rho), 1.0, applied=False)

    theta = cov / var_x
    y_adj = y - theta * (x - float(x.mean()))
    ratio = float(np.var(y_adj, ddof=1) / var_y)
    return CupedResult(y_adj, theta, float(rho), ratio, applied=True)


def standardized_mean_diff(x, variant) -> float:
    """两组协变量的**标准化均值差**（SMD，单位是合并标准差）。

    随机分流下它应该只有随机波动（``|SMD|`` 大多在 1 以内，
    10 万用户时 95% 落在 ±0.01 内）。用 SMD 而不是绝对差，
    是因为绝对差没有可比性 —— 「均值差 3 元」是好是坏取决于方差。
    """
    v = np.asarray(variant).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    groups = [x[v == g] for g in np.unique(v)]
    if len(groups) < 2 or any(g.size < 2 for g in groups):
        return float("nan")
    a, b = groups[0], groups[1]
    pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
    if pooled <= 0:
        return float("nan")
    return abs(float(a.mean()) - float(b.mean())) / pooled


def leakage_warning(x, variant, *, smd_threshold: float = 1.0) -> str | None:
    """检查协变量是否与分组相关 —— 相关就说明它**可能**是实验后的量。

    只是一个**提示**，不是证明：
    * 随机分流下，实验前的协变量与分组只有随机差异；
    * 但如果 X 是实验后的量，它就会带上处理效应，SMD 明显偏大。

    ⚠️ 反过来说，「没报警」不等于没问题：真实效应很小时，
       实验后的协变量也可能看起来与分组无关（SMD 只有 0.05）。
       真正的保证是**在代码里把 X 取在分流之前**，这个函数只是最后一道网。
    """
    xa = np.asarray(x, dtype=np.float64)
    if np.all(xa == xa.reshape(-1)[0]):
        return "协变量是常数，CUPED 无法工作（θ 无定义）"
    smd = standardized_mean_diff(x, variant)
    if np.isfinite(smd) and smd > smd_threshold:
        return (f"协变量在两组间的标准化差异 {smd:.2f}σ，远超随机波动 —— "
                f"它很可能不是实验前的量（或分流本身有问题），"
                f"CUPED 会把处理效应一起调整掉")
    return None
