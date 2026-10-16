"""广告拍卖市场：给定出价，算赢率与期望扣费。

**为什么用 LogNormal 而不是「出价高就一定赢」**

真实的广告交易市场是**多个广告主同时竞价**：你出 3 块，对手出 3.2 块，
你就输了。把赢率建模成 0/1 会让整条出价链路失去意义 ——
预算控制、水位调节全都不需要了。

市场价（第二高价，也就是你实际要付的钱）在广告业里是出了名的右偏长尾，
**LogNormal 是行业里最常用的拟合分布**。选它还有一个非常实际的理由：
赢率和条件期望扣费都有**解析解**，不需要蒙特卡洛。

    竞争价 X ~ LogNormal(μ_m, σ_m)          （单位：元/千次，CPM）

    赢率          P(win | b) = P(X < b) = Φ((ln b − μ_m)/σ_m)
    期望扣费      E[X | X < b] = e^{μ+σ²/2} · Φ((ln b − μ − σ²)/σ) / Φ((ln b − μ)/σ)
                                 └── 截断对数正态的一阶矩，标准结果

⚠️ **第二条公式里的 ``−σ²`` 极易漏掉。** 漏掉它得到的是
   ``E[X] = e^{μ+σ²/2}`` 这个**无条件**均值 —— 也就是「不管赢没赢，
   市场价平均是多少」。它比条件均值**大**（因为我们只在 X 比较小时
   才赢），所以在低价出价区间会**高估成本**，进而把水位压得过低。
   这个错误不会报错，只会让消耗一直跑不满预算。

**第一价 vs 第二价**：本模块默认第二价（``price_rule="second"``），
这是主流交易市场的规则。第一价的扣费就是出价本身，也实现了，
用来做对照 —— 第一价下最优出价要乘一个 shading 因子，
是另一套策略，本项目的控制器不覆盖它。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Φ 的下界截断。P(win) = 0 会让期望成本出现 0/0，也会让 eCPM 排序失去意义。
# 1e-12 对应「出价低到几乎没有赢面」，比任何真实投放的边界都远。
_PHI_FLOOR = 1e-12


@dataclass
class MarketConfig:
    """市场参数。单位统一是**元/千次曝光（CPM）**。

    ⚠️ 单位必须在整个模块里保持一致。``bidder`` 那边算出来的是
       **元/次曝光**，两边相减之前必须有一个显式的换算 ——
       ``Config.cpm_to_impression`` 就是为这件事存在的。
       CPM 和「元/次」差 1000 倍，混淆的话水位会偏 1000 倍，
       而且表现出来的现象只是「消耗跑不满」，很难联想到单位。
    """

    mu_m: float = float(np.log(25.0))   # ln(市场均价 25 元/CPM)
    sigma_m: float = 0.6

    @property
    def mean_cpm(self) -> float:
        """市场价的**无条件**均值 E[X] = e^{μ+σ²/2}。"""
        return float(np.exp(self.mu_m + self.sigma_m ** 2 / 2.0))

    @classmethod
    def from_mean(cls, mean_cpm: float, sigma_m: float) -> "MarketConfig":
        """用「市场均价 + 对数标准差」参数化。

        直接指定 μ 很容易写出一个均值和预期差十倍的市场 ——
        因为 E[X] = e^{μ+σ²/2} 里的 σ²/2 在大 σ 下不可忽略
        （σ=0.6 时是 1.20 倍）。
        """
        if mean_cpm <= 0:
            raise ValueError("市场均价必须为正")
        return cls(mu_m=float(np.log(mean_cpm) - sigma_m ** 2 / 2.0),
                   sigma_m=float(sigma_m))


# --------------------------------------------------------------------------
# 赢率与扣费
# --------------------------------------------------------------------------


def win_prob(bid_cpm: np.ndarray, mkt: MarketConfig) -> np.ndarray:
    """``P(win | b) = Φ((ln b − μ_m)/σ_m)``。

    定义域处理：``b ≤ 0`` 时返回 0（不出价就是弃权），
    而不是让 ``ln(0) = -inf`` 一路传播成 NaN。
    """
    b = np.asarray(bid_cpm, dtype=np.float64)
    out = np.zeros_like(b)
    np.maximum(b, 0.0, out=b)
    ok = b > 0
    if not np.any(ok):
        return out
    z = (np.log(b[ok]) - mkt.mu_m) / mkt.sigma_m
    out[ok] = stats.norm.cdf(z)
    return out


def expected_cost_cpm(bid_cpm: np.ndarray, mkt: MarketConfig,
                      price_rule: str = "second") -> np.ndarray:
    """赢下这次曝光时，**期望要付多少钱**（CPM）。

    - ``second``（默认）：付第二高价，即市场价 X 在 ``X < b`` 下的条件均值。
      这是本模块里唯一需要 ``−σ²`` 修正的地方。
    - ``first``：付自己的出价，成本就是 ``b``。

    ⚠️ 返回值是**条件**期望（已经赢了的条件下）。
       要算「每次请求的期望花费」得再乘 ``win_prob`` ——
       见 ``expected_spend_cpm``。这两个量经常被混为一谈，
       混了会让预算规划差一个赢率因子（本市场下约 0.5），
       表现出来就是「预算永远花不完」。
    """
    b = np.asarray(bid_cpm, dtype=np.float64)
    if price_rule == "first":
        return np.maximum(b, 0.0)
    if price_rule != "second":
        raise ValueError(f"未知的计价规则 {price_rule!r}")

    p = win_prob(b, mkt)
    safe = np.maximum(p, _PHI_FLOOR)
    z = (np.log(np.maximum(b, 1e-12)) - mkt.mu_m - mkt.sigma_m ** 2) / mkt.sigma_m
    # 截断对数正态的一阶矩 / P(X<b)
    return np.exp(mkt.mu_m + mkt.sigma_m ** 2 / 2.0) * stats.norm.cdf(z) / safe


def expected_spend_cpm(bid_cpm: np.ndarray, mkt: MarketConfig,
                       price_rule: str = "second") -> np.ndarray:
    """每次请求的**无条件**期望花费 = 赢率 × 条件期望扣费。"""
    return win_prob(bid_cpm, mkt) * expected_cost_cpm(bid_cpm, mkt, price_rule)


# --------------------------------------------------------------------------
# 向量化的单日拍卖
# --------------------------------------------------------------------------


def budget_gate_scale(bid_cpm: np.ndarray, remaining: float,
                      mkt: MarketConfig,
                      cpm_to_impression: float = 1e-3) -> float:
    """预算硬闸门：返回缩放系数，使这一批的**期望**扣费不超过 ``remaining``。

    ``1.0`` 表示不用缩，``0.0`` 表示预算已尽、必须停投。
    ``remaining`` 由调用方扣掉 reserve 之后给出（各策略的 reserve 口径不同）。

    ⚠️ **用期望扣费而不是「全部赢下且按第一价付款」的最坏情况**，
       这是实测换来的结论，值得写清楚。

       最坏情况读起来更严谨（「红线上不能用概率保证」），但它比期望大
       **5~7 倍** —— 赢率只有约 0.1，而且赢了只付市场价。于是晚高峰一个
       slot 的期望花费约 8.7 元、最坏情况却是 46 元，闸门在还剩 36 元的时候
       就启动了（此时全天只剩 6 个 slot、实际只会再花约 30 元）。
       更糟的是它和 PID 会形成一个平衡点：控制器看到落后 → 抬 α →
       闸门按比例压回来 → 永远落后。实测的代价是**系统性少花 1.7%**，
       而且 α 轨迹平滑、闸门命中次数只有 5，任何单看指标的排查都发现不了。

       用期望口径的风险有多大？闸门把期望扣费压在剩余预算内，而单个 slot
       的实际扣费相对期望的波动只有 4~5%（750 个以上请求的均值，重尾也被
       平均掉了），reserve 比这个尾巴大一个数量级。所以这里保证的是
       「**期望**不超支 + 尾部由 reserve 吸收」，而不是
       「无论怎么抽样都不超」。把后者做实的代价是 3% 的预算花不出去，
       而超支红线真正要防的是「某个 slot 突然抢量把预算打爆」——
       4% 的 slot 级波动离那个场景很远。
    """
    if remaining <= 0.0:
        return 0.0
    b = np.maximum(np.asarray(bid_cpm, dtype=np.float64), 0.0)
    exp = float(np.sum(expected_spend_cpm(b, mkt))) * float(cpm_to_impression)
    if exp <= remaining:
        return 1.0
    return float(remaining / exp)


def run_auction(bid_cpm: np.ndarray, market_bid_cpm: np.ndarray,
                price_rule: str = "second") -> tuple[np.ndarray, np.ndarray]:
    """给定出价和**已抽好的**市场价，返回 ``(是否赢, 实际扣费)``。

    和上面两个解析函数的关系：解析式给的是**期望**，这个函数给的是
    一次抽样的**实现值**。仿真的每一轮都要用实现值（否则预算消耗
    就是确定的，pacing 的方差、积分的抗饱和全都测不出来），
    而控制器的规划要用期望值（否则规划本身带噪）。

    这个分工是刻意的：**用期望做决策，用实现算账。**
    """
    b = np.asarray(bid_cpm, dtype=np.float64)
    x = np.asarray(market_bid_cpm, dtype=np.float64)
    win = (x < b) & (b > 0)
    if price_rule == "first":
        cost = np.where(win, b, 0.0)
    else:
        # 第二价：赢了就付市场价（GSP 下是下一名的出价）
        cost = np.where(win, x, 0.0)
    return win, cost
