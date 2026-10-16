"""功效分析与样本量反解。

**为什么这件事必须做在实验之前**：样本量决定了实验的**分辨率**。
样本不够时，一个真实存在的 +5% 提升大概率测不出来（power 不足），
于是实验的结论是「无显著差异」—— 而团队会把它读成「这个功能没用」，
把它下线。这不是统计学的细枝末节，这是**用数据做出错误决策**的典型路径。

反过来说，算出来的样本量经常会**推翻既定的实验设计**：
本项目的例子就是——按 CVR 基线 6%、相对提升 5% 的目标，
朴素公式给出「每臂 10.1 万次点击」（人均 17 次点击 → 约 5800 用户），
看着 10 万活跃用户绰绰有余；但把实测的设计效应
（``deff ≈ 6.4``）代进去，真实需求是每臂约 64.3 万次点击、
约 3.7 万用户 —— 差了 6 倍。这个结论比跑出一个显著的 p 值
有价值得多：它在实验**开始前**就告诉你「按现在的流量，
5% 的提升大概率测不出来，要么扩流量、要么把目标放宽到 14%」。

**公式**（两独立比例，双侧）

    n_per_arm = ( z_{1−α/2}·sqrt(2·p̄·q̄) + z_{1−β}·sqrt(p₁q₁ + p₂q₂) )² / Δ²

其中 ``p̄ = (p₁+p₂)/2`` 是 H0 下的合并比例（第一项用合并方差，
对应「检验」），第二项用各自方差（对应「备择假设下要有多大的
标准误才能把差异认出来」）。漏掉第二项是常见错误 ——
那只算出了「差异刚好达到显著性」的样本量，power 只有 50%。

**``deff``：这个公式最容易出错的地方**

公式假设每个观测单元独立，但 CVR 这类指标的观测单元是**点击**，
而点击嵌套在用户里：同一个人有稳定的转化倾向（点击之间正相关），
且用户的点击量差两个数量级（方差被重度用户撑起来）。
结果是真实方差是朴素二项方差的 ``deff`` 倍，
**有效样本量只有 ``n / deff``**。

所以 ``deff`` 必须从数据里量出来（``stats.design_effect``），
不能拍 1。本项目的模拟里实测 deff ≈ 6.6，
也就是说「17 万次点击」只相当于两万多次独立观测 ——
不乘这个数会把样本量低估 6 倍以上，
而低估的后果不是「跑得久一点」，是**实验做完得出「无提升」的错误结论**。

⚠️ ``deff`` 是**量出来的**，不是常数。它对数据生成过程的细节很敏感
   （用户的点击量分布、转化倾向的离散度、指标的分母口径），
   换一份数据就要重新测。这里传的 ``deff`` 都是同一个量纲：
   **聚类稳健方差 ÷ 朴素二项方差**，与 ``stats.design_effect`` 一致。

**MDE 是反解出来的**，没有闭式解（n 同时出现在两个地方），
用二分法。区间取 ``[1e-6, 0.5]``：相对提升超过 50% 的 MDE
在效果广告里已经不是「最小可检测」而是「出了大事」。
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _z(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("显著性水平必须在 (0, 1) 内")
    if not 0 < power < 1:
        raise ValueError("功效必须在 (0, 1) 内")
    return (float(sps.norm.ppf(1 - alpha / 2)),
            float(sps.norm.ppf(power)))


def sample_size_per_arm(baseline: float, mde_relative: float, *,
                        alpha: float = 0.05, power: float = 0.80,
                        deff: float = 1.0) -> float:
    """每臂所需**观测单元数**。``mde_relative`` 是**相对**提升（0.05 = +5%）。

    ``deff``（设计效应）是聚类带来的方差膨胀倍数，见模块 docstring。
    返回的是**实际要收集的单元数** = 独立单元数 × ``deff``，
    也就是「点击次数」这个口径 —— 想换算成用户数再除以人均点击数。

    ⚠️ 基线为 0、``p₂`` 越界或 ``deff`` 非正时返回 ``inf`` 而不是抛异常：
       「测不出来」是个合法的结论，抛异常会让调用方不得不写 try，
       而写 try 的人往往会顺手 return 一个假数字。
    """
    if not 0 < baseline < 1:
        raise ValueError("基线比例必须在 (0, 1) 内")
    if deff <= 0:
        return float("inf")
    if mde_relative == 0:
        return float("inf")
    p1 = baseline
    p2 = baseline * (1 + mde_relative)
    if not 0 < p2 < 1:
        return float("inf")
    delta = abs(p2 - p1)
    z_a, z_b = _z(alpha, power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return float(np.ceil(num / delta ** 2 * deff))


def achieved_power(baseline: float, n_per_arm: float, mde_relative: float, *,
                   alpha: float = 0.05, deff: float = 1.0) -> float:
    """给定样本量能拿到多少功效（``sample_size_per_arm`` 的正向计算）。

    用来回答「手上只有这么多用户，我能测出多大的提升」。
    ``n_per_arm`` 同样是**观测单元数**（点击），内部按 ``deff`` 折算成
    独立单元数 ``n/deff`` —— 这一步漏掉就会把功效算高 ``deff`` 倍。
    """
    if not 0 < baseline < 1:
        raise ValueError("基线比例必须在 (0, 1) 内")
    if n_per_arm <= 0 or deff <= 0:
        return 0.0
    n_eff = n_per_arm / deff
    p1 = baseline
    p2 = baseline * (1 + mde_relative)
    delta = abs(p2 - p1)
    if delta == 0:
        return float("nan")
    z_a, _ = _z(alpha, 0.5)
    p_bar = (p1 + p2) / 2
    se_h0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_eff)
    se_h1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_eff)
    return float(sps.norm.sf(z_a * se_h0 / se_h1 - delta / se_h1))


def mde(baseline: float, n_per_arm: float, *, alpha: float = 0.05,
        power: float = 0.80, deff: float = 1.0, lo: float = 1e-6,
        hi: float = 0.5, iters: int = 100) -> float:
    """反解 MDE：给定样本量，能测出的**最小相对提升**是多少。

    二分法。``achieved_power`` 对 ``mde`` 单调递增（提升越大越好测），
    所以二分一定收敛。若连 ``hi`` 处都达不到目标功效，
    返回 ``hi`` —— 「样本量连 50% 的提升都测不出来」是必须让调用方
    看到的结论，返回 NaN 会被下游静默吃掉。
    """
    if n_per_arm <= 0:
        return float("inf")
    if achieved_power(baseline, n_per_arm, hi, alpha=alpha,
                      deff=deff) < power:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if achieved_power(baseline, n_per_arm, mid, alpha=alpha,
                          deff=deff) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def plan_experiment(baseline: float, mde_relative: float, *,
                    available_units: float | None = None,
                    units_per_user: float | None = None,
                    deff: float = 1.0, alpha: float = 0.05,
                    power: float = 0.80) -> dict:
    """把「要多少样本」和「有这么多样本时能做到什么」一次算清楚。

    ``available_units`` 是**观测单元数**（点击），``units_per_user``
    是人均单元数（人均点击数）—— 给这两个数就把结论换算成人，
    否则只报单元数。**不要**直接把用户数当 ``available_units`` 传进来，
    那会漏掉 ``units_per_user`` 和 ``deff`` 两个因子，
    样本量能差两个数量级。

    返回字典而不是 dataclass：这是一个**给人看的决策摘要**，
    字段会随项目演进增删，用 dict 更合适。
    """
    need = sample_size_per_arm(baseline, mde_relative, alpha=alpha,
                               power=power, deff=deff)
    out = {
        "baseline": baseline,
        "mde_relative": mde_relative,
        "alpha": alpha,
        "power": power,
        "deff": deff,
        "units_per_arm": need,
        "total_units": 2 * need,
    }
    if units_per_user:
        out["users_per_arm"] = need / units_per_user
        out["total_users"] = 2 * need / units_per_user
    if available_units is not None:
        per_arm = available_units / 2
        out["available_units"] = available_units
        out["achieved_power"] = achieved_power(baseline, per_arm,
                                               mde_relative, alpha=alpha,
                                               deff=deff)
        out["mde_at_available"] = mde(baseline, per_arm, alpha=alpha,
                                      power=power, deff=deff)
        out["feasible"] = bool(available_units >= 2 * need)
        if units_per_user:
            out["available_users"] = available_units / units_per_user
    return out
