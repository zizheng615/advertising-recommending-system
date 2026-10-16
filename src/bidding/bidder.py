"""把模型预测翻译成出价，以及六条对照策略（B0–B5）。

**eCPM 公式与单位**

    eCPM   = pCTR × pCVR × LTV × 1000      （元 / 千次曝光）
    bid    = eCPM × α / 1000               （元 / 次曝光）

``α`` 是出价水位（``controller`` 的输出）。``× 1000`` 是因为
eCPM 按**千次**计价，而拍卖是按**每次**曝光结算的 ——
``Config.cpm_to_impression = 0.001`` 就是这一步。

⚠️ **这个 1000 是本项目最容易出错的地方**，因为它既不报错也不难看：
   漏掉它，出价会偏小 1000 倍，现象是「消耗只有预算的 0.1%」；
   多乘一次，出价偏大 1000 倍，现象是「第一个 slot 就把预算打光」。
   两个现象都很容易被误诊成「控制器参数不对」，然后去调 PID ——
   越调越远。所以单位换算**只在 ``ecpm_to_bid`` 一处发生**，
   其余任何地方都不许再出现 1000。

**LTV 用哪个口径**：必须是 ZILN 的 ``Ê[y] = p̂·exp(μ̂ + σ̂²/2)``，
不是 ``p̂·exp(μ̂)``（漏掉对数正态修正，低估约 ``exp(σ²/2)`` 倍，
本项目 σ≈0.85 下是 1.44 倍）。这个错误会同时压低所有出价，
表现出来是「量买不够」，而不是「出价算错了」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bidding.auction import MarketConfig, budget_gate_scale
from src.bidding.controller import WaterLevelController
from src.metrics.regression import expected_value

# eCPM 按千次计价的换算系数。只在 ``ecpm_to_bid`` 里使用。
CPM = 1000.0


# --------------------------------------------------------------------------
# 从预测到出价
# --------------------------------------------------------------------------


def ltv_from_ziln(p: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """ZILN 的期望金额。**带对数正态修正**，见模块 docstring。"""
    return expected_value(np.asarray(p, dtype=np.float64),
                          np.asarray(mu, dtype=np.float64),
                          np.asarray(sigma, dtype=np.float64))


def ecpm(p_ctr: np.ndarray, p_cvr: np.ndarray, ltv: np.ndarray) -> np.ndarray:
    """元/千次曝光。三个入参都必须已经是**真实先验**下的概率。

    ⚠️ 负下采样训练出来的模型，推理前必须做先验校正
       （``logit += cfg.prior_logit_shift``）。漏掉这一步，
       pCTR 会高估约 5 倍，eCPM 跟着高估 5 倍 —— 而 AUC 完全正常，
       所以模型评估发现不了，只能靠出价阶段的「消耗爆掉」暴露。
    """
    pc = np.asarray(p_ctr, dtype=np.float64)
    pv = np.asarray(p_cvr, dtype=np.float64)
    lt = np.asarray(ltv, dtype=np.float64)
    for name, a in (("pCTR", pc), ("pCVR", pv), ("LTV", lt)):
        if np.any(a < 0):
            raise ValueError(f"{name} 出现负值，概率/金额不可能为负")
    return pc * pv * lt * CPM


def ecpm_to_bid(ecpm_cpm: np.ndarray, alpha: float) -> np.ndarray:
    """``bid = eCPM × α / 1000``（元/次曝光 → 再乘回 CPM）。

    上面那行注释读起来别扭，是因为拍卖那边统一用 CPM。
    展开写：``bid_cpm = (eCPM_cpm × α) × cpm_to_impression × 1000``
    —— 也就是 ``eCPM_cpm × α``。**两次 1000 恰好抵消**，
    所以最终表达式里看不到 1000。这正是它容易被「顺手优化掉」的原因，
    故此处保留显式写法并写明抵消关系。
    """
    e = np.asarray(ecpm_cpm, dtype=np.float64)
    if alpha < 0:
        raise ValueError("水位不能为负")
    return np.maximum(e, 0.0) * float(alpha)


@dataclass
class BidOutcome:
    """一次批量出价的中间量，方便把每一步都画出来、测出来。"""

    ecpm: np.ndarray
    bid: np.ndarray
    alpha: float
    gate_scale: float = 1.0
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------


class PacingPolicy:
    """策略基类。``sim_day`` 只依赖这三个方法。"""

    name = "policy"

    #: 是否允许看到 ``true_value_cpm``。**只有 B5 上帝视角是 True**。
    #: 这个开关放在策略上而不是仿真里，是为了让「谁能看真值」这件事
    #: 一眼可查 —— ``tests`` 里有一条断言「除 B5 之外没有任何策略打开它」。
    views_truth = False

    def reset(self) -> None:            # pragma: no cover - 默认空实现
        pass

    def alpha(self, slot: int) -> float:  # pragma: no cover - 抽象
        raise NotImplementedError

    def observe(self, slot: int, spend: float) -> None:  # pragma: no cover
        pass

    def bid(self, slot: int, pred_value_cpm: np.ndarray) -> tuple[float, np.ndarray]:
        """返回 ``(水位, 出价 CPM)``。

        ``bid`` 和 ``alpha`` 分开，是因为 **B0 固定出价没有水位这个概念** ——
        它的出价是常数，不随价值变化。如果只提供 ``alpha``，
        B0 就只能伪装成「α = 常数/价值」，那已经不是固定出价了。
        """
        a = self.alpha(slot)
        return a, ecpm_to_bid(pred_value_cpm, a)

    def gate_scale(self, bid_cpm: np.ndarray) -> float:
        """默认不设闸门。有预算约束的策略必须覆写。"""
        return 1.0


class GreedyPolicy(PacingPolicy):
    """B1：按价值全额出价（α ≡ 1），**没有任何 pacing**。

    它的作用是证明 pacing 不是可有可无的装饰：α ≡ 1 时出价等于期望价值，
    而期望价值远高于市场价，于是**日初就把预算打光** ——
    等到晚高峰（一天里流量既多又好、最值钱的时段）到来时，
    预算早就没了，一点没买到。

    **硬闸门保留**，理由和 B0 一样：闸门是平台的预算保护，
    真实平台一定会在预算耗尽后停投，它不是这条策略的功劳。
    去掉闸门反而会把对比搞糊 —— 那样 B1 的超支会来自
    「没有预算概念」，而不是这里要证明的「买的时点不对」。
    两条策略的预算合规性一样，差别就只剩「什么时候买」，
    这才是干净的对照。
    """

    name = "B1_greedy"

    def __init__(self, budget: float, market: MarketConfig):
        self.budget = float(budget)
        self.market = market
        self.spent = 0.0

    def reset(self) -> None:
        self.spent = 0.0

    def alpha(self, slot: int) -> float:
        return 1.0

    def observe(self, slot: int, spend: float) -> None:
        self.spent += float(spend)

    def gate_scale(self, bid_cpm: np.ndarray) -> float:
        return budget_gate_scale(bid_cpm, self.budget - self.spent,
                                 self.market)


class FixedBidPolicy(PacingPolicy):
    """B0：全天一个固定出价（不分人群、不看素材、不看时段）。

    这是「人工粗放投放」的量化化身，也是本项目要打败的对手。
    固定值通过 ``calibrate`` 以「期望花完预算」为准反解 ——
    不能让 B0 因为出价恰好定得太低而输，那样赢得不光彩。

    ⚠️ 硬闸门**保留**：固定出价也受平台预算保护，这是真实情况。
       去掉闸门会让 B0 超支，但那超支来自「没有预算概念」，
       和「出价不分人群」是两件事，不该混进同一个对比里。
    """

    name = "B0_fixed"

    def __init__(self, bid_cpm: float, budget: float, market: MarketConfig):
        self.bid_cpm = float(bid_cpm)
        self.budget = float(budget)
        self.market = market
        self.spent = 0.0

    def reset(self) -> None:
        self.spent = 0.0

    def alpha(self, slot: int) -> float:
        return 1.0

    def bid(self, slot: int, pred_value_cpm: np.ndarray) -> tuple[float, np.ndarray]:
        v = np.asarray(pred_value_cpm, dtype=np.float64)
        return 1.0, np.full(v.shape, self.bid_cpm, dtype=np.float64)

    def observe(self, slot: int, spend: float) -> None:
        self.spent += float(spend)

    def gate_scale(self, bid_cpm: np.ndarray) -> float:
        return budget_gate_scale(bid_cpm, self.budget - self.spent,
                                 self.market)


class ControllerPolicy(PacingPolicy):
    """B2/B3/B4：把 ``WaterLevelController`` 包成策略。

    三档只差两个开关：
    - ``B2_lambda_only``：只用前馈 λ（PID 关）
    - ``B3_pid_only``：只用 PID（前馈关，α_ff 停在初值）
    - ``B4_lambda_pid``：全开（**主方案**）
    """

    def __init__(self, ctl: WaterLevelController, name: str = "B4_lambda_pid"):
        self.ctl = ctl
        self.name = name

    def reset(self) -> None:
        self.ctl.reset_day()

    def alpha(self, slot: int) -> float:
        return self.ctl.alpha(slot)

    def observe(self, slot: int, spend: float) -> None:
        self.ctl.observe_slot(slot, spend)

    def gate_scale(self, bid_cpm: np.ndarray) -> float:
        return self.ctl.budget_scale(bid_cpm)


class OraclePolicy(PacingPolicy):
    """B5：上帝视角 —— 用**真实** pCTR/pCVR/LTV，且 λ 离线全局最优。

    两个优势叠加，构成这条链路能达到的**上界**：

    1. **没有模型误差**：用生成器的真值，等于把 DeepFM/ZILN 的误差清零。
    2. **不必在线试错**：λ 用解析拍卖模型在**全天流量上**一次性反解，
       而不是像控制器那样一小时一步地摸。在线控制的代价（滞后、
       超调、日末才追上）在这里全部消失。

    ⚠️ 它是**同一个策略类**（线性于价值、单调于水位）的上界，
       不是「所有可能策略」的上界。写清楚这一点很重要：
       B4 达到 B5 的 85–95% 说明「模型误差 + 在线试错的代价」被控制住了，
       而不是说「B4 已经最优了」。

    **``views_truth`` 是这条策略的全部特权**：它出价用的是真值而不是预测。
    这一点差点写错 —— 第一版只把 ``alpha_star`` 换成真值反解的，
    出价却仍然走基类的 ``bid()``，也就是 ``α* × pred_value``。
    于是 B5 既漏掉了模型误差的对比，又比真正的上帝视角**少花 1.2%**
    （两条出价的赢率差 0.2 个百分点）。这种「一半作弊」最难发现：
    B4/B5 的比值看起来完全正常（0.98），只有把 B5 的花费和解析期望
    对不上账时才露馅。
    """

    name = "B5_oracle"
    views_truth = True

    def __init__(self, ctl: WaterLevelController, alpha_star: float):
        self.ctl = ctl
        self.alpha_star = float(alpha_star)

    def reset(self) -> None:
        self.ctl.reset_day(alpha_init=self.alpha_star)
        # 离线最优值不再需要在线调整
        self.ctl.use_feedforward = False
        self.ctl.use_pid = False

    def alpha(self, slot: int) -> float:
        return self.alpha_star

    def observe(self, slot: int, spend: float) -> None:
        self.ctl.observe_slot(slot, spend)

    def gate_scale(self, bid_cpm: np.ndarray) -> float:
        return self.ctl.budget_scale(bid_cpm)


# --------------------------------------------------------------------------
# 策略工厂
# --------------------------------------------------------------------------


def build_policies(cfg, market: MarketConfig, *, pred_value_cpm: np.ndarray,
                   true_value_cpm: np.ndarray, budget: float,
                   traffic: np.ndarray,
                   plan_weights: np.ndarray | None = None) -> list[PacingPolicy]:
    """构造 B0–B5 六条策略，水位初值都用**同一个**解析反解。

    ⚠️ 六条策略的初值必须一致，否则对比的就不是策略而是初值。
       B2/B3/B4 的 α 初值来自预测价值（它们只看得到预测），
       B5 的来自真实价值（它有权看到真值）—— 这个差别是**刻意保留**的，
       因为它正是「上帝视角」的定义的一部分。
    """
    from src.bidding.controller import solve_alpha

    n_req = int(pred_value_cpm.shape[0])
    plan = budget * cfg.budget_safety

    # ⚠️ 反解出来的水位**必须**落进 [alpha_min, alpha_max]，
    #    否则控制器第一步就把它 clip 掉，热启动等于白做，
    #    而且现象只是「消耗怎么都跑不满」。
    #    这里夹一下保证仿真不崩；**但夹住就说明标定有问题**，
    #    所以 tests 里有一条 test_solved_water_levels_are_inside_the_band
    #    专门断言这件事没发生 —— 夹持只是兜底，不是解决方案。
    def clipped(a: float) -> float:
        return float(min(max(a, cfg.alpha_min), cfg.alpha_max))

    a_init = clipped(solve_alpha(pred_value_cpm, market, plan / n_req))
    a_star = clipped(solve_alpha(true_value_cpm, market, plan / n_req))

    def mk(use_ff: bool, use_pid: bool, name: str) -> ControllerPolicy:
        ctl = WaterLevelController(cfg, market, budget=budget, traffic=traffic,
                                   alpha_init=a_init, plan_weights=plan_weights,
                                   use_feedforward=use_ff, use_pid=use_pid)
        return ControllerPolicy(ctl, name)

    # B0 的固定出价：让**期望**花费正好等于计划预算的那个恒定 CPM。
    # 用 solve_alpha 的同一套机制，只是把 α 乘在「全 1 的价值」上 ——
    # 等价于直接反解一个常数出价。
    fixed = (solve_alpha(np.ones(n_req), market, plan / n_req)
             if n_req else 0.0)

    oracle_ctl = WaterLevelController(cfg, market, budget=budget,
                                      traffic=traffic, alpha_init=a_star,
                                      plan_weights=plan_weights)
    return [
        FixedBidPolicy(fixed, budget, market),
        GreedyPolicy(budget, market),
        mk(True, False, "B2_lambda_only"),
        mk(False, True, "B3_pid_only"),
        mk(True, True, "B4_lambda_pid"),
        OraclePolicy(oracle_ctl, a_star),
    ]
