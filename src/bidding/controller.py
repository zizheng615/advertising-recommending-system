"""预算约束下的水位控制器：前馈 λ + 反馈 PID + 硬闸门。

**先把优化问题写清楚，否则「水位」这个词会一直含糊**

     max  Σ_i q_i · P(win | b_i)      s.t.  Σ_i c_i · P(win | b_i) ≤ B
     b_i

``q_i = pCTR_i · pCVR_i · LTV_i`` 是这次曝光的期望价值，``c_i`` 是扣费。
拉格朗日乘子 λ（**预算的影子价格**，单位是「多花一块钱的机会成本」）给出

    b*_i = v_i / λ        α ≡ 1/λ 就是出价水位

α 的物理含义：**「我愿意为一块钱的期望价值付多少钱」**。
α=1 是「按价值出价」（不赚钱也不亏），α<1 是「只买划算的那部分流量」。
预算越紧，λ 越大，α 越小，买得越少 —— 这就是全部的直觉。

⚠️ **对偶梯度的符号极易写反，而且写反了程序照样跑得欢**
（只是消耗会跑到 0 或者跑到爆）。自检一遍：

    超支 S > B  →  λ 要变大  →  α = 1/λ 变小  →  出价降低  →  消耗下降 ✅

所以 λ 的更新是 ``λ ← λ · exp(η_λ · (S − B)/B)``，
在 **log 空间**做乘法更新有两个好处：λ 恒正（不需要 clip 掉负值导致卡死），
以及 ``(S−B)/B`` 归一化之后 η_λ 与预算的绝对大小无关 ——
预算从 5000 改成 50000 不用重调参。

**三环分工（这是「PID 到底在控制什么」的答案）**

| 环 | 时间尺度 | 控制什么 | 为什么 |
|---|---|---|---|
| 前馈 λ | 每小时 | **买什么** | 慢、平滑、有经济含义（影子价格） |
| 反馈 PID | 每 15min | **花多少** | 快、有界，只修「预测不准」的部分 |
| 硬闸门 | 每请求 | **绝不超支** | 前两环都是软的，红线必须有硬的兜底 |

分工不是洁癖：把「买什么」也交给快环，α 会跟着流量噪声乱跳，
同一份流量早上出一个价、下午出另一个价，投放完全不可解释。
反过来，让慢环去修预算偏差，它会一直慢半拍，到日末才发现花不掉。

**PID 递推（精确定义，测试按这个手算）**

    e_t   = (B_plan · F(t) − A_t) / B_plan        进度偏差，正 = 落后
    ẽ_t   = ρ · ẽ_{t−1} + (1−ρ) · e_t             D 项专用平滑
    I_t   = clip(I_{t−1} + k_i · e_t , ±i_max)    仅在不死区、不饱和时累加
    u_t   = clip(k_p · e_t + I_t + k_d · (ẽ_t − ẽ_{t−1}), ±u_max)
    α_t   = clip(α_ff · exp(u_t), α_min, α_max)

``F(t)`` 是**历史流量曲线归一化后的累计值**，不是 ``t/T``。
这个区别是实质性的：线性进度假设流量均匀，
而真实流量在凌晨 3 点只有晚高峰的 1/4，
用线性进度会让 PID 在凌晨误以为「落后了」，然后疯狂提价去买没人看的流量 ——
钱花掉了，量一点没多。

**D 项的物理意义**：``de/dt = B·f(t) − s_t``，也就是
「此刻应该花的速率」减去「实际花的速率」。
所以 D 项惩罚的是**消耗速率的超额**，天然抑制「突发抢量」。

**抗饱和（三件套全上，缺一个都会出问题）**
1. 积分限幅 ``|I| ≤ i_max``：防止长时间同向偏差把积分撑到无法回头的量级。
2. 输出饱和时冻结积分：否则 α 已经顶到 α_max 还在累加，
   等偏差反向时要花十几个 slot 才能把积分放掉 —— 这段时间消耗是失控的。
3. 每小时衰减 ``I ← I · integral_decay``：给积分一个遗忘的出口，
   让控制器对新信息保持敏感，而不是被早上的一个异常 slot 绑架一整天。

**四层不超支**
1. 硬闸门：这一批的**期望**扣费不许超过剩余预算（``budget_scale``）。
   ⚠️ 这里用的是期望而不是「全部赢下」的最坏情况 —— 最坏情况比期望大
   5~7 倍，会让闸门提前 6 个 slot 就启动，实测代价是系统性少花 1.7%。
   完整的权衡写在 ``auction.budget_gate_scale``。
2. 保守曲线：全天只计划花 ``budget_safety = 99%``，留 1% 给预测误差。
3. reserve：``reserve_frac = 0.5%`` 不参与计划，专门吸收
   「闸门按期望放行、实际却多花了一点」的那点尾巴。
4. 日末收紧：最后 2 小时压 ``u_max``，防止为了凑数去抢高价低质流量。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bidding.auction import (MarketConfig, budget_gate_scale,
                                 expected_spend_cpm)

# 每个小时包含的 slot 数（96 slot / 24h）
_SLOTS_PER_HOUR = 4


# --------------------------------------------------------------------------
# 流量曲线
# --------------------------------------------------------------------------


def diurnal_weights(n_slots: int, *, peak_hour: float = 20.0,
                    peak_width: float = 4.0, floor: float = 0.18) -> np.ndarray:
    """一天 96 个 slot 的流量权重（**已归一化，和为 1**）。

    用「以晚高峰为中心的周期高斯 + 地板」而不是真实的历史报表：
    形状对了就够了（凌晨低谷、晚高峰、白天平台），
    而参数化的曲线可以写进测试里做精确断言。

    ``floor`` 是凌晨谷底相对峰值的比例。0.18 对应
    **峰值约为谷底的 3.4 倍** —— 效果广告的实测昼夜比大多在 3–5 倍。

    ⚠️ 这个值不能随手调到 0.3 以上。昼夜比一旦压到 2.5 倍左右，
       「用线性进度 t/T 代替流量曲线」这个错误的代价就变得很小，
       而它本来是 PID 设计里最容易犯、后果最严重的错误之一
       （凌晨误判成「落后了」→ 疯狂提价 → 钱花了、量没多）。

    环状距离用 ``min(|Δ|, 24−|Δ|)``：不这么写的话，
    23 点和 1 点会被算成「相距 22 小时」，曲线在午夜断掉。
    """
    if n_slots % 24 != 0:
        raise ValueError(f"slot 数 {n_slots} 必须能被 24 整除（否则整点对齐不了）")
    h = (np.arange(n_slots) + 0.5) * (24.0 / n_slots)
    d = np.abs(h - peak_hour)
    d = np.minimum(d, 24.0 - d)
    w = floor + (1.0 - floor) * np.exp(-0.5 * (d / peak_width) ** 2)
    return w / w.sum()


def cumulative_weights(w: np.ndarray) -> np.ndarray:
    """累计流量占比 ``F(t)``，``F(-1) := 0``。用作「此刻应花多少」的进度基准。"""
    c = np.cumsum(np.asarray(w, dtype=np.float64))
    return c / c[-1]


def slot_quality(w: np.ndarray, *, floor: float = 0.25) -> np.ndarray:
    """把流量曲线变成**价值密度**曲线：``0.25 → 1.0``。

    实测峰谷价值密度比约 2.1（峰值时段 39.5 元/CPM，谷底 18.8）。
    真实效果广告的晚高峰不只是**人多**，而且**人好** ——
    晚上在家的人才有时间逛、才有购买意图。

    ⚠️ 这个比值不能随手调到接近 1。它一旦消失，
       「把预算花在早上还是晚上」就不再是个有对错的决策，
       pacing 退化成只剩「别超支」，B4 与 B1 的对比也就没了。
    """
    return floor + (1.0 - floor) * (w / w.max())


def spend_weights(traffic: np.ndarray, *, floor: float = 0.25) -> np.ndarray:
    """**出价进度曲线**：``traffic × quality`` 归一化。

    ⚠️ 这是本项目里一个不显眼但很关键的设计点，值得单独讲。

    PID 的误差是 ``(计划花费 − 实际花费)``，所以「计划花费曲线」用哪条
    直接决定了控制器的行为。第一版用的是**流量曲线** ``F(t)``，
    理由是「按流量进度花钱」听起来天经地义 —— 结果是**系统性地少花 1.6%**，
    而且原因很难看出来。

    真正的原因是：按价值出价的买手，花费密度正比于
    ``流量 × 价值密度``，而价值密度本身在晚高峰更高。
    所以「自然花费曲线」比流量曲线**更靠后**。
    拿流量曲线当基准，控制器从上午开始就一直看到
    「实际 < 计划、我落后了」，于是抬价去追 ——
    而它追的方向恰恰是**把预算提前花在便宜的时段**，
    正好和 pacing 想做的事相反。

    换成 ``流量 × 价值密度`` 之后，基准和「不动手时的自然轨迹」对齐，
    控制器才真正只在**出了意外**时才动作。
    这件事说明：pacing 的基准曲线不是随便挑一条历史曲线就行，
    它必须和出价策略的自然花费分布同形。
    """
    q = slot_quality(traffic, floor=floor)
    w = np.asarray(traffic, dtype=np.float64) * q
    return w / w.sum()


# --------------------------------------------------------------------------
# 水位初值：用解析拍卖模型反解
# --------------------------------------------------------------------------


def solve_alpha(values_cpm: np.ndarray, market: MarketConfig,
                target_spend_per_request: float, *,
                cpm_to_impression: float = 1e-3,
                lo: float = 1e-6, hi: float = 100.0, iters: int = 80) -> float:
    """反解 α，使**每次请求的期望花费（元）**等于给定目标（二分法）。

    为什么需要这个：λ 的迭代是每小时一步、一天只有 24 步，
    而 α 的真实量级和它的自然初值（1.0）可能差一两个数量级。
    从 1.0 起步、每步最多动 ``max_dlog_alpha``，一天下来也追不上，
    表现就是「消耗爆掉 + 控制器一直在往下压但压不住」。

    这条路径解析拍卖模型**本来就是为它准备的** ——
    有了 ``P(win|b)`` 和条件期望扣费的闭式解，反解一个标量只是二分。

    期望花费对 α 单调递增，所以二分一定收敛。返回的是 ``[lo, hi]``
    内的一个点；若真实解在区间外，返回对应的端点（宁可给个保守的初值，
    也不要抛异常把整条仿真打断 —— 控制器随后会自己修正）。

    ⚠️ **单位陷阱（这里踩过，所以参数名写得很啰嗦）**：
       ``expected_spend_cpm`` 返回的是**元/千次曝光**，
       而 ``target_spend_per_request`` 是**元/次**，两者差 1000 倍。
       第一版直接把 ``预算/请求数``（元/次）传进去当 CPM 用，
       于是二分求出的 α 对应的真实花费只有目标的千分之一 ——
       现象是「消耗怎么都跑不满预算」，而 α 的轨迹平滑正常、
       控制器看起来工作得好好的，非常难查。
       现在换算**只在函数内部发生一次**，调用方只跟「元/次」打交道。
    """
    v = np.asarray(values_cpm, dtype=np.float64)
    if v.size == 0 or not np.any(v > 0):
        raise ValueError("没有正的曝光价值，无法反解水位")
    if target_spend_per_request <= 0:
        raise ValueError("目标花费必须为正")
    if cpm_to_impression <= 0:
        raise ValueError("CPM 换算系数必须为正")

    def spend_at(a: float) -> float:
        # 元/千次 → 元/次
        return float(np.mean(expected_spend_cpm(a * v, market))) * cpm_to_impression

    if spend_at(hi) < target_spend_per_request:
        return hi
    if spend_at(lo) > target_spend_per_request:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if spend_at(mid) < target_spend_per_request:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# 控制器
# --------------------------------------------------------------------------


@dataclass
class ControllerState:
    """控制器的可观测内部状态（用于测试与绘图）。"""

    slot: int = -1
    alpha_ff: float = 1.0
    u: float = 0.0
    error: float = 0.0
    integral: float = 0.0
    spent: float = 0.0
    gated: bool = False


class WaterLevelController:
    """一天的预算水位控制器。**每个自然日必须 ``reset_day()``。**

    ⚠️ 日切重置是硬性要求，不是优化：λ 和积分携带的是**当天**的预算
       信息，跨天复用会让第二天的前几个小时按昨天的偏差出价。
       表现是「今天开头总是异常激进或异常保守」，而且很难查。
    """

    def __init__(self, cfg, market: MarketConfig, *,
                 budget: float | None = None,
                 traffic: np.ndarray | None = None,
                 alpha_init: float | None = None,
                 plan_weights: np.ndarray | None = None,
                 use_feedforward: bool = True,
                 use_pid: bool = True):
        self.cfg = cfg
        self.market = market
        # 两个开关只用来做消融（B2 仅 λ / B3 仅 PID）。
        # 关掉前馈时 α_ff 停在初值，前馈环就退化成「一个静态水位 + PID 修偏差」；
        # 关掉 PID 时 u ≡ 0，水位完全由影子价格决定。
        self.use_feedforward = bool(use_feedforward)
        self.use_pid = bool(use_pid)
        self.budget = float(cfg.daily_budget if budget is None else budget)
        if self.budget <= 0:
            raise ValueError("日预算必须为正")
        self.w = (diurnal_weights(cfg.t_slots) if traffic is None
                  else np.asarray(traffic, dtype=np.float64))
        if self.w.shape != (cfg.t_slots,):
            raise ValueError(f"流量曲线长度 {self.w.shape} 与 t_slots={cfg.t_slots} 不符")
        # 进度基准：默认用「流量 × 价值密度」，理由见 ``spend_weights``。
        # 单元测试里直接构造控制器时通常只给 traffic，走默认即可。
        self.pw = (spend_weights(self.w) if plan_weights is None
                   else np.asarray(plan_weights, dtype=np.float64))
        if self.pw.shape != (cfg.t_slots,):
            raise ValueError(f"进度曲线长度 {self.pw.shape} 与 t_slots={cfg.t_slots} 不符")
        self.F = cumulative_weights(self.pw)

        # 全天计划花费：留 1% 给预测误差。硬闸门用的是**真实预算** B，
        # 两者差出来的这 1% 就是闸门在正常情况下不该被触发的原因 ——
        # 闸门一旦触发就说明前两环已经失效了，它只是最后一道保险。
        self.plan = self.budget * cfg.budget_safety
        self.reserve = self.budget * cfg.reserve_frac
        self.spendable = self.budget - self.reserve

        self._alpha_init = float(alpha_init if alpha_init is not None else 1.0)
        self.reset_day(alpha_init=self._alpha_init)

    # ---------------- 日切 ----------------

    def reset_day(self, *, alpha_init: float | None = None) -> None:
        """重置到一天的开始。λ 回到初值、积分清零、已花归零。"""
        a0 = float(self._alpha_init if alpha_init is None else alpha_init)
        if not (self.cfg.alpha_min <= a0 <= self.cfg.alpha_max):
            raise ValueError(
                f"初始水位 {a0} 落在 [alpha_min={self.cfg.alpha_min}, "
                f"alpha_max={self.cfg.alpha_max}] 之外，会被第一步就 clip 掉")
        self.lam = 1.0 / a0          # 影子价格，α = 1/λ
        self.alpha_ff = a0
        self.integral = 0.0
        self.err_smooth = 0.0        # ẽ_{t-1}
        self._alpha = a0
        self.st = ControllerState(slot=-1, alpha_ff=a0)

    # ---------------- 只读

    @property
    def spent(self) -> float:
        return self.st.spent

    @property
    def remaining(self) -> float:
        """还能花多少（扣掉 reserve）。**已经扣过 reserve**，
        所以调用方不要重复扣。"""
        return max(0.0, self.spendable - self.st.spent)

    def planned_spend(self, slot: int) -> float:
        """到 slot 结束为止，按流量曲线应该花掉多少。"""
        return self.plan * float(self.F[slot])

    def end_of_day_factor(self, slot: int) -> float:
        """日末 2 小时把 ``u_max`` 打折，从 1.0 线性收到 0.4。

        最后两小时如果还落后，PID 会想大幅提价把预算花掉 ——
        但此刻剩下的流量质量通常最差，硬买就是「为花钱而花钱」。
        压住上界，接受「今天没花完」比「花完但买了一堆垃圾」好。
        """
        tail = 2 * _SLOTS_PER_HOUR
        if slot < self.cfg.t_slots - tail:
            return 1.0
        k = (self.cfg.t_slots - 1 - slot) / (tail - 1.0)   # 1.0 → 0.0
        return 0.4 + 0.6 * k

    # ---------------- 主循环
    #
    # 状态推进**只发生在 ``_begin_slot`` 里**，而它只在 slot 变化时触发。
    # 这样 ``alpha(slot)`` 在同一个 slot 内被调用多少次都返回同一个值 ——
    # 仿真里每个请求都要问一次 α，如果每次调用都推进一步积分，
    # 那么「一个 slot 有几个请求」就会悄悄改变控制器的行为，
    # 而这件事在任何指标上都看不出来（只会发现调参怎么调都不对）。

    def alpha(self, slot: int) -> float:
        """当前 slot 的出价水位 α。同 slot 内幂等。"""
        if slot != self.st.slot:
            self._begin_slot(slot)
        return self._alpha

    def _begin_slot(self, slot: int) -> None:
        """进入 slot：算误差 → 更新积分 → 算 u → 定 α。"""
        prev = self.st.slot
        if slot < prev:
            raise ValueError(f"slot 必须单调递增：{prev} → {slot}")
        if slot >= self.cfg.t_slots:
            raise ValueError(f"slot {slot} 超出一天的范围 [0, {self.cfg.t_slots})")

        # 跳过的 slot 按「没花钱」处理：进度会自然落后，PID 随后补上。
        # 静默跳过而不是报错，是因为仿真里按小时聚合是常见写法。
        e = (self.planned_spend(slot) - self.st.spent) / self.plan

        # D 项必须在 ẽ 被覆盖**之前**取到上一轮的值。
        # 顺序写反不会报错，只会让 D 项恒等于 0（差分变成自己减自己），
        # 而少一个 D 项的系统仍然能跑、只是抑制突发抢量的能力没了。
        prev_smooth = self.err_smooth
        self.err_smooth = (self.cfg.ema_rho * prev_smooth
                           + (1.0 - self.cfg.ema_rho) * e)
        d_term = self.cfg.k_d * (self.err_smooth - prev_smooth)

        u_max = self.cfg.u_max * self.end_of_day_factor(slot)

        # --- 积分：死区 + 抗饱和 + 限幅 ---
        u_raw = self.cfg.k_p * e + self.integral + d_term
        saturated = abs(u_raw) >= u_max
        pushing_out = (u_raw > 0 and e > 0) or (u_raw < 0 and e < 0)
        if (self.use_pid and abs(e) > self.cfg.pid_deadzone
                and not (saturated and pushing_out)):
            self.integral = float(np.clip(self.integral + self.cfg.k_i * e,
                                          -self.cfg.i_max, self.cfg.i_max))

        # 积分更新后重算 u（否则用的是上一轮的积分，等价于凭空多一拍延迟）
        u_raw = self.cfg.k_p * e + self.integral + d_term
        u = float(np.clip(u_raw, -u_max, u_max)) if self.use_pid else 0.0

        self._alpha = float(np.clip(self.alpha_ff * np.exp(u),
                                    self.cfg.alpha_min, self.cfg.alpha_max))
        self.st.slot, self.st.error, self.st.u = slot, float(e), u
        self.st.integral, self.st.alpha_ff = self.integral, self.alpha_ff
        self.st.gated = self.remaining <= 0.0

    def observe_slot(self, slot: int, spend: float) -> None:
        """一个 slot 结束后汇报实际花费，并推进前馈 λ。

        ⚠️ ``spend`` 是**本 slot 新增**的花费，不是累计值。
           传累计值进去不会报错，只是 λ 会以为第一个小时就花光了预算，
           然后一路把 α 压到 ``alpha_min``。
        """
        if spend < 0:
            raise ValueError("slot 花费不能为负")
        if slot != self.st.slot:
            self._begin_slot(slot)
        self.st.spent += float(spend)

        # --- 小时边界上的两件事 ---
        # ⚠️ 积分衰减**不能**挂在 ``use_feedforward`` 上。它是 PID 的
        #    抗饱和机制，和影子价格没有关系；绑在一起的话，
        #    「只开前馈」的消融档会连带丢掉 PI 的抗饱和，
        #    对比出来的差异就不再只归因于「有没有前馈」。
        if (slot + 1) % _SLOTS_PER_HOUR == 0:
            if self.use_pid:
                self.integral *= self.cfg.integral_decay
            if self.use_feedforward:
                self._update_feedforward()
        self.st.alpha_ff = self.alpha_ff

    def _update_feedforward(self) -> None:
        """对偶梯度一步：``λ ← λ·exp(η_λ·(S−B)/B)``，等价于
        ``ln α_ff ← ln α_ff − η_λ·(S−B)/B``（因为 α = 1/λ）。

        归一化用 ``plan`` 而不是 ``budget``：前馈环的职责是「按计划曲线走」，
        而计划本来就是 97% 的预算。用 budget 归一化会让前馈永远认为
        「还差 3% 没花」，长期偏置在「多花一点」的方向上。
        """
        err = (self.st.spent - self.planned_spend(self.st.slot)) / self.plan
        step = float(np.clip(-self.cfg.eta_lambda * err,
                             -self.cfg.max_dlog_alpha, self.cfg.max_dlog_alpha))
        ln_a = float(np.clip(np.log(self.alpha_ff) + step,
                             np.log(self.cfg.alpha_min),
                             np.log(self.cfg.alpha_max)))
        self.alpha_ff = float(np.exp(ln_a))
        self.lam = 1.0 / self.alpha_ff

    # ---------------- 硬闸门

    def budget_scale(self, bid_cpm: np.ndarray) -> float:
        """预算硬闸门：把这一批的**期望**扣费压在剩余预算内。

        算术在 ``auction.budget_gate_scale`` 里，那里写了「为什么用期望
        而不是最坏情况」的完整推导 —— 一句话是：最坏情况比期望大 5~7 倍，
        会让闸门在还剩 36 元时就把出价压下去，代价是系统性少花 1.7%。

        这里只负责把「剩余多少」按控制器的口径算出来：已经扣掉 reserve。
        """
        return budget_gate_scale(bid_cpm, self.remaining, self.market,
                                 self.cfg.cpm_to_impression)
