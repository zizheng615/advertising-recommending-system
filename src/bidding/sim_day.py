"""单日出价仿真：真实广告环境的数字孪生。

**仿真的是什么**

一天 96 个 15min slot，流量按昼夜曲线起伏（凌晨低谷 / 晚高峰），
每个请求带着「这个人的 pCTR、pCVR、LTV 是多少」进场，
多条策略各自出价，与市场价竞价，赢的扣费、可能转化。

**为什么要有它**：出价链路的正确性**无法从单测看出**。
PID 的积分抗饱和、λ 的收敛速度、硬闸门的边界 ——
这些都是**轨迹性质**，只有在「一天 96 步、预算有限、流量有起伏」
的闭环里才暴露。单测只能验证「一步算得对」。

**两个价值口径（这是理解全部指标的关键）**

| 口径 | 含义 | 用途 |
|---|---|---|
| ``pred_value`` | 模型预测的 q·LTV | **策略只能看到这个**，出价据此产生 |
| ``true_value`` | 生成器的真实 q·LTV | **只用于打分**，绝不进入任何策略 |

真实环境里没有 ``true_value``；这里能拿到，是因为数据是模拟的。
用它打分让策略对比的方差大幅下降（不用等转化真的发生），
但必须严守一条纪律：**``true_value`` 不得出现在 ``DayStream`` 之外**。
``test_bidding.py`` 里有一条专门断言这件事的测试 ——
一旦泄漏，B5「上帝视角」和 B4 的差距会消失，而所有指标看起来都很好。

**主指标是 ``expected_value``**：赢下的曝光按**真实** q·LTV 计的价值和。
这正是优化目标 ``max Σ q_i·P(win|b_i)`` 的实现值。
``realized_value``（按伯努利抽样实际发生的转化计）也报，但方差大得多 ——
一次转化就是几千块，噪声能把策略差距完全盖住。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bidding.auction import MarketConfig, run_auction
from src.bidding.bidder import PacingPolicy
from src.bidding.controller import (cumulative_weights, diurnal_weights,
                                    slot_quality, spend_weights)

# 一个 slot 内的最大批量。只影响内存，不影响结果（拍卖是逐请求独立的）
_SLOT_CHUNK = 200_000


@dataclass
class DayStream:
    """一天的请求流。**唯一允许持有真值的地方**。"""

    slot: np.ndarray            # (n,) int，所属 slot
    pred_value_cpm: np.ndarray  # (n,) 模型预测 q·LTV，单位 CPM
    true_value_cpm: np.ndarray  # (n,) 真实 q·LTV，**只用于打分**
    market_bid_cpm: np.ndarray  # (n,) 该次请求的市场竞争价
    true_conv_prob: np.ndarray  # (n,) 真实转化概率 q，用于抽样实现值
    true_ltv: np.ndarray        # (n,) 转化发生时的期望金额（用于打分口径）
    realized_ltv: np.ndarray    # (n,) 转化真的发生时收到的金额
    traffic: np.ndarray = field(default_factory=lambda: np.zeros(0))
    plan_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(self.slot.shape[0])


@dataclass
class DayResult:
    """一条策略跑完一天的结果。"""

    name: str
    budget: float
    spend: float
    expected_value: float     # 按真实 q·LTV 计的价值（主指标）
    realized_value: float     # 按实际发生的转化计
    requests: int
    wins: int
    win_rate: float
    pacing_rmse: float
    cumulative_spend: np.ndarray
    planned_spend: np.ndarray
    alpha_path: np.ndarray
    gate_hits: int

    @property
    def overspend_frac(self) -> float:
        """超支比例。**负数表示没花完**，所以只看正值。"""
        return (self.spend - self.budget) / self.budget

    @property
    def budget_error(self) -> float:
        """预算偏差的绝对值，目标 < 2%。"""
        return abs(self.overspend_frac)

    @property
    def roi(self) -> float:
        """花一块钱买到多少期望价值。"""
        return self.expected_value / self.spend if self.spend > 0 else 0.0

    def summary(self) -> str:
        return (f"{self.name:16s} 花费 {self.spend:8.1f} "
                f"(误差 {self.overspend_frac * 100:+6.2f}%) "
                f"价值 {self.expected_value:9.1f} "
                f"ROI {self.roi:5.2f} "
                f"赢率 {self.win_rate * 100:5.1f}% "
                f"pacing {self.pacing_rmse:.4f}")


# --------------------------------------------------------------------------
# 构造请求流
# --------------------------------------------------------------------------


def _synthetic_values(n: int, cfg, rng: np.random.Generator, *,
                      noise_sigma: float = 0.35):
    """不依赖任何模型的价值分布，供测试与快速仿真使用。

    返回 ``(true_q, true_ltv, true_realized_ltv, pred_q, pred_ltv)``。

    ``noise_sigma`` 是**模型预测误差**在对数空间的规模。
    0.35 大致对应一个「特征齐、训练充分」的模型（排序相关性 0.9 左右）。

    ⚠️ 不要让预测误差无限大 —— 那样 B4 和 B5 的差距会变成
       「模型有多差」而不是「pacing 有多好」，整个对照就失去意义了。

    **LTV 的口径（这里踩过一个坑，值得写下来）**

    第一版直接把「这个人最终付了多少钱」当成 LTV，于是 90% 的请求
    LTV 恰好是 0（对应 ``zero_inflation=0.90``），``pred_value_cpm``
    整列是 0，出价全为 0。

    错在把**实现值**当成了**期望值**。出价要用的 LTV 是
    ``Ê[y] = P(付费)·E[金额|付费]`` —— **对每个人都严格为正**，
    只是高价值用户和长尾用户差两个数量级。这正是 ZILN 的输出口径：
    它不会对任何人输出 0，只会输出一个很小的数。

    所以这里分成两个量，含义完全不同，绝不能混用：

    * ``true_ltv``：**期望**金额，用于价值打分（``true_value_cpm``）
    * ``true_realized_ltv``：转化**真的发生**时收到的金额，
      ``E[realized] = true_ltv``，只用于 ``realized_value``
    """
    # q = pCTR × pCVR：用 Beta 造两个有右偏长尾的概率，再相乘。
    # 直接对 q 取 lognormal 会让 pCTR、pCVR 各自超过 1，
    # 而 q ≤ 1 是硬约束（虽然实际值远小于 1，不会撞到上界）。
    a_ctr, b_ctr = 2.0, 38.0            # 均值 0.05
    a_cvr, b_cvr = 2.0, 31.3            # 均值 0.06
    true_q = rng.beta(a_ctr, b_ctr, n) * rng.beta(a_cvr, b_cvr, n)

    # 期望 LTV：人人都为正，但高价值用户（(1-zero_inflation) 那一小撮）
    # 比长尾用户高出约两个数量级。分两簇而不是一条光滑分布，
    # 是因为真实的用户价值分布本来就是双峰的 —— 有付费习惯的人和没有的人。
    heavy = rng.random(n) < (1.0 - cfg.zero_inflation)
    heavy_ltv = np.exp(rng.normal(cfg.ltv_mu_log, cfg.ltv_sigma, n))
    light_ltv = np.exp(rng.normal(cfg.ltv_mu_log - 2.5, 0.7, n))
    true_ltv = np.where(heavy, heavy_ltv, light_ltv)

    # 实现值：只影响 realized_value，均值为 true_ltv（对数正态无偏还原）
    true_realized_ltv = true_ltv * np.exp(
        rng.normal(0.0, cfg.ltv_sigma, n) - 0.5 * cfg.ltv_sigma ** 2)

    # 模型视角：真值 + 对数正态噪声（均值无偏）
    def noisy(x):
        return x * np.exp(rng.normal(0.0, noise_sigma, n)
                          - 0.5 * noise_sigma ** 2)

    return true_q, true_ltv, true_realized_ltv, noisy(true_q), noisy(true_ltv)


def make_stream(cfg, market: MarketConfig, *, n_requests: int, seed: int,
                traffic: np.ndarray | None = None,
                value_source=None,
                market_shock: tuple[int, int, float] | None = None) -> DayStream:
    """造一天的请求流。

    ``value_source(n, rng) -> (true_q, true_ltv, realized_ltv, pred_q, pred_ltv)``
    留作注入点：``scripts/07_bidding_sim.py`` 会传一个用**真实模型输出**
    构造的 source，把 DeepFM / ZILN 的预测接进拍卖。

    ``market_shock=(slot_lo, slot_hi, factor)``：在 ``[slot_lo, slot_hi)``
    这段把市场价整体乘 ``factor``，用来模拟「竞品突然加价 / 大促开始」。
    ⚠️ 这是**非平稳市场**的人工构造，PID 的价值只能在这里测出来：
       平稳流下前馈 λ 一个环就把 pacing 做到 0.5% 以内，快环无事可做。
       扰动只持续两三个小时，而 λ 每小时才更新一次、单步上限
       ``max_dlog_alpha``，追不上；PID 每个 slot 都在看进度误差。
    """
    if market_shock is not None:
        lo, hi, factor = market_shock
        if not (0 <= lo < hi <= cfg.t_slots):
            raise ValueError(f"扰动窗口 [{lo}, {hi}) 超出一天的范围")
        if factor <= 0:
            raise ValueError("扰动倍数必须为正")
    if n_requests <= 0:
        raise ValueError("请求数必须为正")
    rng = np.random.default_rng([seed, 31337])

    w = diurnal_weights(cfg.t_slots) if traffic is None else np.asarray(traffic)
    per_slot = rng.multinomial(n_requests, w / w.sum())
    slot = np.repeat(np.arange(cfg.t_slots), per_slot)

    src = ((lambda n, r: _synthetic_values(n, cfg, r))
           if value_source is None else value_source)
    true_q, true_ltv, realized_ltv, pred_q, pred_ltv = src(len(slot), rng)

    # --- 时段对**价值密度**的影响（不只是流量多少）---
    #
    # ⚠️ 这一段是「pacing 到底有没有用」的**前提**，不能省。
    #    如果价值在一天里是同分布的，那么「早上把预算花光」和
    #    「均摊到全天」买到的流量价值完全一样，
    #    pacing 唯一的作用就退化成「别超支」，B4 赢 B1 也就无从谈起。
    #    （第一版正是这样，于是 B1 的价值**高于** B4 ——
    #    因为它花得更多。这不是 bug，是仿真缺了一个真实世界的性质。）
    #
    #    真实世界里晚高峰的流量不只是**多**，而且**好** ——
    #    晚上在家的人才有时间逛、才有购买意图。这里用一个
    #    以流量曲线为形状的乘子来表达：峰谷之间约 2 倍。
    #
    #    乘子对 true / pred / realized 三个口径**同步施加**，
    #    所以不会引入任何系统性偏差，只是让「什么时候买」
    #    变成一个真正有对错的决策。
    quality = slot_quality(w)[slot]
    true_ltv = true_ltv * quality
    realized_ltv = realized_ltv * quality
    pred_ltv = pred_ltv * quality

    # 市场价：每次请求独立抽一个 LogNormal。用解析参数化保证
    # 抽样的均值就是 ``market.mean_cpm``（否则仿真和市场模型对不上）。
    market_bid = rng.lognormal(market.mu_m, market.sigma_m, len(slot))
    if market_shock is not None:
        lo, hi, factor = market_shock
        market_bid = np.where((slot >= lo) & (slot < hi),
                              market_bid * factor, market_bid)

    return DayStream(
        slot=slot,
        pred_value_cpm=pred_q * pred_ltv * 1000.0,
        true_value_cpm=true_q * true_ltv * 1000.0,
        market_bid_cpm=market_bid,
        true_conv_prob=true_q,
        true_ltv=true_ltv,
        realized_ltv=realized_ltv,
        traffic=w / w.sum(),
        plan_weights=spend_weights(w),
    )


# --------------------------------------------------------------------------
# 跑一天
# --------------------------------------------------------------------------


def simulate_day(stream: DayStream, policy: PacingPolicy, cfg,
                 market: MarketConfig, *, budget: float | None = None,
                 price_rule: str = "second", seed: int = 0,
                 cpm_to_impression: float | None = None) -> DayResult:
    """让一条策略跑完 ``stream`` 这一天。"""
    B = float(cfg.daily_budget if budget is None else budget)
    to_imp = float(cfg.cpm_to_impression if cpm_to_impression is None
                   else cpm_to_impression)
    policy.reset()
    rng = np.random.default_rng([seed, 9091])

    plan = B * cfg.budget_safety
    # ⚠️ 进度基准必须和控制器用的是**同一条曲线**（spend_weights），
    #    否则 pacing_rmse 测的是「两条不同曲线的差」，不是控制精度。
    F = cumulative_weights(stream.plan_weights if stream.plan_weights.size
                           else spend_weights(diurnal_weights(cfg.t_slots)))

    cum_spend = np.zeros(cfg.t_slots)
    alpha_path = np.zeros(cfg.t_slots)
    spend = wins = 0.0
    exp_value = real_value = 0.0
    gate_hits = 0

    slot_all = stream.slot
    for t in range(cfg.t_slots):
        m = slot_all == t
        if not m.any():
            alpha_path[t] = policy.alpha(t)
            policy.observe(t, 0.0)
            cum_spend[t] = spend
            continue

        # 出价依据：默认是**预测**价值。只有声明了 ``views_truth`` 的
        # 上帝视角策略（B5）才拿真值 —— 开关在策略类上，见 bidder.PacingPolicy。
        vals = (stream.true_value_cpm[m] if policy.views_truth
                else stream.pred_value_cpm[m])
        a, bid = policy.bid(t, vals)
        scale = policy.gate_scale(bid)
        if scale < 1.0:
            gate_hits += 1
            bid = bid * scale

        # 分块拍卖：内存友好，且逐请求独立所以与整块算结果完全一致
        idx = np.flatnonzero(m)
        won_all = np.zeros(idx.size, dtype=bool)
        cost_all = np.zeros(idx.size)
        for s in range(0, idx.size, _SLOT_CHUNK):
            e = min(s + _SLOT_CHUNK, idx.size)
            w_, c_ = run_auction(bid[s:e], stream.market_bid_cpm[idx[s:e]],
                                 price_rule)
            won_all[s:e] = w_
            cost_all[s:e] = c_

        slot_spend = float(cost_all.sum() * to_imp)
        spend += slot_spend
        wins += int(won_all.sum())

        # 打分：只有赢下的曝光才算价值
        exp_value += float(stream.true_value_cpm[idx[won_all]].sum()) * to_imp
        if won_all.any():
            w_idx = idx[won_all]
            conv = rng.random(w_idx.size) < stream.true_conv_prob[w_idx]
            real_value += float(stream.realized_ltv[w_idx][conv].sum())

        policy.observe(t, slot_spend)
        cum_spend[t] = spend
        alpha_path[t] = a

    planned = plan * F
    denom = max(B, 1e-12)
    pacing = float(np.sqrt(np.mean(((cum_spend - planned) / denom) ** 2)))

    return DayResult(
        name=policy.name, budget=B, spend=spend, expected_value=exp_value,
        realized_value=real_value, requests=len(stream), wins=int(wins),
        win_rate=wins / max(1, len(stream)), pacing_rmse=pacing,
        cumulative_spend=cum_spend, planned_spend=planned,
        alpha_path=alpha_path, gate_hits=gate_hits,
    )


def run_all_policies(stream: DayStream, policies: list[PacingPolicy], cfg,
                     market: MarketConfig, *, budget: float | None = None,
                     price_rule: str = "second", seed: int = 0
                     ) -> list[DayResult]:
    """跑完一组策略。

    ⚠️ 所有策略共用**同一条 stream**（同一批请求、同一批市场价）。
       每条策略各抽一天的话，策略间差异里会混进流量抽样噪声，
       而晚高峰早高峰的流量差异远大于策略差异 —— 必须配对比较。
       但 ``simulate_day`` 内部给每条策略**相同的 rng 种子**，
       所以「转化是否发生」这一层的抽样也是配对的。
    """
    return [simulate_day(stream, p, cfg, market, budget=budget,
                         price_rule=price_rule, seed=seed)
            for p in policies]
