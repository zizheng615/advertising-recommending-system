"""职责点 3：拍卖模型 / 水位控制器 / 单日出价仿真。

测试的策略分三层，容差策略也分三层
------------------------------------
* **纯数学**（赢率公式、条件期望、单位换算、PID 递推、积分限幅）
  → 精确断言 ``1e-10``。这些是闭式解或纯递推，没有随机性，
    断言松了就等于没测。
* **抽样一致性**（解析式 vs 蒙特卡洛）
  → 用固定的 ``n`` 和容差，容差由 MC 标准误反推，不随手给 0.01。
* **闭环轨迹**（预算误差、pacing、策略对比）
  → 门槛断言。这里的数**不是**从理论上算出来的，
    而是先跑一遍看实际水平，再把门槛设在「明显差于实测」的位置 ——
    门槛的作用是抓回归，不是证明自己优秀。

⚠️ 本文件里所有涉及金额的断言都依赖 ``daily_budget``。
   迷你档默认的 5000 元配上测试用的请求量**根本花不完**
   （200k 请求 × 50% 赢率 × 0.018 元 ≈ 1800 元），
   预算约束根本不 binding，PID 会退化成「没有约束、随便出价」，
   所有 pacing 相关的断言都会失去意义。所以测试统一把预算
   缩到 800 元，让约束真正起作用 —— 这一点必须先想清楚，
   否则会写出一堆「永远通过」的假测试。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from config import get_config
from src.bidding.auction import (MarketConfig, expected_cost_cpm,
                                 expected_spend_cpm, run_auction, win_prob)
from src.bidding.bidder import (ControllerPolicy, FixedBidPolicy,
                                GreedyPolicy, OraclePolicy, build_policies,
                                ecpm, ecpm_to_bid, ltv_from_ziln)
from src.bidding.controller import (WaterLevelController, cumulative_weights,
                                    diurnal_weights, slot_quality, solve_alpha,
                                    spend_weights)
from src.bidding.sim_day import make_stream, run_all_policies, simulate_day

# --------------------------------------------------------------------------
# 全局规模
# --------------------------------------------------------------------------
N_REQUESTS = 200_000
# 见模块 docstring：预算必须小到让约束真正 binding。
# 实测 α=1（无约束）的期望花费是 1090.8 元，所以 300 元对应 U/B≈3.6 ——
# 这个倍数不是随手挑的：它必须大到让 B1 贪心在**下午就断粮**，
# 从而完全错过晚高峰。U/B 只有 1.4 时贪心全天都有钱花，
# 「pacing 有什么用」就演示不出来了（实测贪心反而略胜）。
TEST_BUDGET = 300.0
SIM_SEED = 2024


@pytest.fixture(scope="module")
def bcfg():
    """出价测试专用配置：预算缩到测试规模，其余参数走默认。"""
    return get_config("mini").scaled(daily_budget=TEST_BUDGET)


@pytest.fixture(scope="module")
def market():
    return MarketConfig.from_mean(25.0, 0.6)


@pytest.fixture(scope="module")
def traffic():
    return diurnal_weights(96)


@pytest.fixture(scope="module")
def stream(bcfg, market, traffic):
    return make_stream(bcfg, market, n_requests=N_REQUESTS, seed=SIM_SEED,
                       traffic=traffic)


@pytest.fixture(scope="module")
def day_results(bcfg, market, traffic, stream):
    """B0–B5 各跑一天，整个模块复用。"""
    pols = build_policies(bcfg, market,
                          pred_value_cpm=stream.pred_value_cpm,
                          true_value_cpm=stream.true_value_cpm,
                          budget=TEST_BUDGET, traffic=traffic)
    return {r.name: r for r in run_all_policies(stream, pols, bcfg, market,
                                                budget=TEST_BUDGET, seed=SIM_SEED)}


# ============================================================ 拍卖（解析）
def test_win_prob_matches_monte_carlo(market):
    """``P(win|b) = Φ((ln b − μ)/σ)`` 要对得上蒙特卡洛。"""
    rng = np.random.default_rng(0)
    draws = rng.lognormal(market.mu_m, market.sigma_m, 4_000_000)
    for b in (10.0, 25.0, 60.0):
        mc = float((draws < b).mean())
        se = float(np.sqrt(mc * (1 - mc) / draws.size))
        assert abs(float(win_prob(np.array([b]), market)[0]) - mc) < 5 * se


def test_expected_cost_has_the_sigma_squared_correction(market):
    """条件期望扣费里的 ``−σ²`` 不能漏。

    漏掉它得到的是**无条件**均值 ``E[X] = e^{μ+σ²/2}``。
    两者在本市场下相差可观（b=15 时条件均值只有无条件均值的 78%），
    所以这条断言能真的抓住这个 bug，而不只是走个形式。
    """
    rng = np.random.default_rng(1)
    draws = rng.lognormal(market.mu_m, market.sigma_m, 8_000_000)
    for b in (12.0, 25.0, 50.0):
        win = draws < b
        n_win = int(win.sum())
        assert n_win > 1000, "样本不够，MC 标准误压不住"
        mc = float(draws[win].mean())
        se = float(draws[win].std(ddof=1) / np.sqrt(n_win))
        got = float(expected_cost_cpm(np.array([b]), market)[0])
        assert abs(got - mc) < 5 * se, f"b={b}: 解析 {got} vs MC {mc}"

    # 无修正版本必须**明显不同**，否则这条测试抓不到想抓的东西
    naive = market.mean_cpm
    corrected = float(expected_cost_cpm(np.array([15.0]), market)[0])
    assert corrected < naive * 0.95, (
        "条件均值应当明显低于无条件均值；若两者接近，"
        "说明 −σ² 项没起作用，这条测试也就没有鉴别力了")


def test_conditional_cost_is_below_the_unconditional_mean(market):
    """赢了才付钱，而赢了意味着市场价偏低 → 条件均值必然小于无条件均值。"""
    b = np.array([10.0, 20.0, 40.0, 80.0])
    assert np.all(expected_cost_cpm(b, market) < market.mean_cpm)
    # 且随出价单调递增（出价越高，赢下的那批里包含越贵的）
    assert np.all(np.diff(expected_cost_cpm(b, market)) > 0)


def test_expected_spend_is_win_prob_times_conditional_cost(market):
    b = np.array([5.0, 15.0, 30.0, 70.0])
    lhs = expected_spend_cpm(b, market)
    rhs = win_prob(b, market) * expected_cost_cpm(b, market)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0)


def test_run_auction_matches_the_analytic_expectation(market):
    """实现值（``run_auction``）的平均要收敛到解析期望。

    这条是仿真可信度的地基：如果对不上，
    「用期望做决策、用实现算账」这个分工就是自相矛盾的。
    """
    rng = np.random.default_rng(2)
    n = 3_000_000
    x = rng.lognormal(market.mu_m, market.sigma_m, n)
    b = np.full(n, 30.0)
    win, cost = run_auction(b, x)
    mc_spend = float(cost.mean())
    analytic = float(expected_spend_cpm(np.array([30.0]), market)[0])
    se = float(cost.std(ddof=1) / np.sqrt(n))
    assert abs(mc_spend - analytic) < 5 * se
    assert abs(float(win.mean()) - float(win_prob(np.array([30.0]), market)[0])) < 1e-3


def test_first_price_costs_exactly_the_bid(market):
    rng = np.random.default_rng(3)
    x = rng.lognormal(market.mu_m, market.sigma_m, 200_000)
    b = np.full(200_000, 30.0)
    win, cost = run_auction(b, x, price_rule="first")
    assert np.allclose(cost[win], 30.0)
    assert np.all(cost[~win] == 0.0)


def test_zero_bid_wins_nothing(market):
    """不出价（b=0）必须干干净净地输，不能出现 ln(0) → NaN。"""
    assert float(win_prob(np.array([0.0, -1.0, 0.0]), market).sum()) == 0.0
    cost = expected_cost_cpm(np.array([0.0]), market)
    assert np.isfinite(cost).all() and float(cost[0]) == 0.0
    win, c = run_auction(np.array([0.0]), np.array([1.0]))
    assert not bool(win[0]) and float(c[0]) == 0.0


def test_from_mean_roundtrips_the_market_mean():
    for m in (5.0, 25.0, 300.0):
        for s in (0.3, 0.6, 1.1):
            mk = MarketConfig.from_mean(m, s)
            assert mk.mean_cpm == pytest.approx(m, rel=1e-12)


def test_unknown_price_rule_is_rejected(market):
    with pytest.raises(ValueError, match="计价规则"):
        expected_cost_cpm(np.array([10.0]), market, price_rule="third")


# ============================================================ 出价换算
def test_ecpm_is_the_documented_formula():
    """``eCPM = pCTR × pCVR × LTV × 1000``，逐项精确。"""
    got = ecpm(np.array([0.05]), np.array([0.06]), np.array([30.0]))
    assert float(got[0]) == pytest.approx(0.05 * 0.06 * 30.0 * 1000.0, rel=1e-15)


def test_bid_and_ecpm_are_consistent_after_the_two_conversions():
    """``bid_cpm = eCPM × α / 1000`` 再乘回 1000 —— 两次换算必须抵消。

    这条测试存在的理由：那两次 1000 最后在表达式里抵消掉了，
    看起来像多余的代码，很容易被「顺手清理」掉。
    """
    e = ecpm(np.array([0.05, 0.02]), np.array([0.06, 0.04]), np.array([30.0, 12.0]))
    a = 0.37
    cpm_to_impression = 0.001
    explicit = e * a * cpm_to_impression * 1000.0     # 元/次 → 元/千次
    assert np.allclose(ecpm_to_bid(e, a), explicit, rtol=1e-15)


def test_zero_alpha_means_no_bid():
    e = ecpm(np.array([0.05]), np.array([0.06]), np.array([30.0]))
    assert float(ecpm_to_bid(e, 0.0)[0]) == 0.0


def test_negative_probabilities_are_rejected():
    with pytest.raises(ValueError, match="负值"):
        ecpm(np.array([-0.01]), np.array([0.06]), np.array([30.0]))


def test_ltv_carries_the_lognormal_correction():
    """``Ê[y] = p·exp(μ + σ²/2)``，不是 ``p·exp(μ)``。

    σ=0.85 时修正因子是 ``exp(0.361) = 1.435`` —— 漏掉它，
    所有出价系统性低估 43%，而 AUC / NLL 这些指标**全都正常**。
    """
    p, mu, sigma = np.array([1.0]), np.array([np.log(60.0)]), np.array([0.85])
    got = float(ltv_from_ziln(p, mu, sigma)[0])
    assert got == pytest.approx(60.0 * np.exp(0.85 ** 2 / 2.0), rel=1e-12)
    assert got > float(p[0] * np.exp(mu[0])) * 1.4


# ============================================================ 控制器
def test_dual_gradient_sign_single_step(bcfg, market, traffic):
    """**对偶梯度符号自检**：超支 → λ↑ → α↓ → 少花。

    写反了程序照样跑（而且 loss 曲线、指标全都不报错），
    只会表现为「消耗跑到 0」或「消耗爆掉」。
    所以这条断言必须显式写出中间量，而不是只看最终 α。
    """
    plan = TEST_BUDGET * bcfg.budget_safety
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    a0, lam0 = ctl.alpha_ff, ctl.lam

    # 跑到第 4 个 slot（触发一次小时级前馈更新），**严重超支**
    for t in range(4):
        ctl.alpha(t)
        ctl.observe_slot(t, plan * 0.5)      # 一个小时就花掉一半计划

    assert ctl.st.spent > ctl.planned_spend(3), "构造失败：这不算超支"
    assert ctl.lam > lam0, "超支了 λ 必须上升（影子价格变贵）"
    assert ctl.alpha_ff < a0, "λ 上升则 α=1/λ 必须下降（少买一点）"

    # 反向：严重欠花 → λ↓ → α↑
    ctl2 = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                traffic=traffic, alpha_init=0.5)
    a0b, lam0b = ctl2.alpha_ff, ctl2.lam
    for t in range(4):
        ctl2.alpha(t)
        ctl2.observe_slot(t, 0.0)
    assert ctl2.lam < lam0b and ctl2.alpha_ff > a0b, "欠花时方向应当相反"


def test_feedforward_step_is_capped(bcfg, market, traffic):
    """单步 ``|Δln α| ≤ max_dlog_alpha``，否则一次异常 slot 就能掀翻水位。"""
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    for t in range(4):
        ctl.alpha(t)
        ctl.observe_slot(t, 0.0)
    step = abs(np.log(ctl.alpha_ff / 0.5))
    assert step <= bcfg.max_dlog_alpha + 1e-12


def test_pid_three_step_recursion_matches_hand_calculation(bcfg, market, traffic):
    """PID 三步递推与手算**精确**一致（1e-10）。

    手算严格按 ``controller`` docstring 里的递推写，不图省事：
    误差 → ẽ 平滑 → D 项 → u → α。任何一步顺序错了都会在这里露馅，
    尤其是「用更新后的积分重算 u」这一步 ——
    漏掉它等价于凭空多一拍延迟，闭环照样收敛，只是慢一点。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic,
                               alpha_init=0.3, use_feedforward=False,
                               use_pid=True)
    plan = TEST_BUDGET * bcfg.budget_safety
    # 进度曲线直接用控制器自己的（spend_weights，不是流量曲线）——
    # 这里要验的是 PID 的递推，不是曲线的形状，
    # 曲线的形状由 test_progress_curve_is_more_back_loaded_than_traffic 负责。
    F = ctl.F
    spends = [3.0, 6.0, 1.0]

    I = 0.0
    err_smooth = 0.0
    spent = 0.0
    for t, sp in enumerate(spends):
        e = (plan * F[t] - spent) / plan
        prev_smooth = err_smooth
        err_smooth = bcfg.ema_rho * prev_smooth + (1 - bcfg.ema_rho) * e
        d_term = bcfg.k_d * (err_smooth - prev_smooth)

        u_max = bcfg.u_max * ctl.end_of_day_factor(t)
        u_raw = bcfg.k_p * e + I + d_term
        saturated = abs(u_raw) >= u_max
        pushing_out = (u_raw > 0 and e > 0) or (u_raw < 0 and e < 0)
        if abs(e) > bcfg.pid_deadzone and not (saturated and pushing_out):
            I += bcfg.k_i * e
        I = float(np.clip(I, -bcfg.i_max, bcfg.i_max))
        u_raw = bcfg.k_p * e + I + d_term
        u = float(np.clip(u_raw, -u_max, u_max))

        expect = float(np.clip(0.3 * np.exp(u), bcfg.alpha_min, bcfg.alpha_max))
        assert ctl.alpha(t) == pytest.approx(expect, abs=1e-10), f"第 {t} 步不符"
        assert ctl.integral == pytest.approx(I, abs=1e-10)

        ctl.observe_slot(t, sp)
        spent += sp


def test_alpha_is_idempotent_within_a_slot(bcfg, market, traffic):
    """同一个 slot 内反复问 α，必须返回同一个值。

    仿真里每个请求都要问一次 α。如果每次调用都推进一次积分，
    那么「一个 slot 里有多少请求」就会悄悄改变控制器行为 ——
    而这个 bug 在任何指标上都看不出来，只会让人调参怎么调都不对。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.4)
    first = ctl.alpha(0)
    i_after = ctl.integral
    for _ in range(500):
        assert ctl.alpha(0) == first
    assert ctl.integral == i_after


def test_slots_must_be_non_decreasing(bcfg, market, traffic):
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic)
    ctl.alpha(10)
    with pytest.raises(ValueError, match="单调递增"):
        ctl.alpha(3)
    with pytest.raises(ValueError, match="超出一天"):
        ctl.alpha(96)


def test_integral_stays_within_its_limits(bcfg, market, traffic):
    """一路欠花 96 个 slot，积分必须始终被夹在 ``±i_max`` 内。

    ⚠️ 注意这里**不能**断言积分最终等于 ``i_max``。带着真实的
       ``k_p=6`` 时，误差稍微一大 u 就顶到 ``u_max``，触发抗饱和冻结 ——
       积分会停在**远低于** ``i_max`` 的位置。这正是抗饱和在起作用，
       所以只断言上界；「冻结」由下面单独的测试负责。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    for t in range(96):
        ctl.alpha(t)
        ctl.observe_slot(t, 0.0)
        assert abs(ctl.integral) <= bcfg.i_max + 1e-12
    assert ctl.integral > 0.0, "一直欠花，积分方向应当为正"


def test_integral_decays_hourly(bcfg, market, traffic):
    """每小时边界上 ``I ← 0.95·I``。

    把 ``k_p`` 和 ``k_i`` 取成特殊值来**隔离**这一步：
    ``k_p=0, k_d=0`` 使 ``u ≡ I``（不会饱和），
    ``k_i=1`` 使累加量就等于误差本身。这样手算可以逐位对上，
    也顺带钉死「衰减发生在累加之后、每个整点一次」这个顺序。
    """
    c = bcfg.scaled(k_p=0.0, k_i=1.0, k_d=0.0, ema_rho=0.0)
    ctl = WaterLevelController(c, market, budget=TEST_BUDGET, traffic=traffic,
                              alpha_init=0.5, use_feedforward=False)
    expect = 0.0
    for t in range(8):
        before = expect
        e = (ctl.planned_spend(t) - ctl.st.spent) / ctl.plan
        if abs(e) > c.pid_deadzone:
            expect = float(np.clip(before + e, -c.i_max, c.i_max))
        assert ctl.alpha(t) == pytest.approx(
            float(np.clip(0.5 * np.exp(np.clip(expect, -c.u_max, c.u_max)),
                          c.alpha_min, c.alpha_max)), abs=1e-12)
        ctl.observe_slot(t, 0.0)
        if (t + 1) % 4 == 0:
            expect *= c.integral_decay
        assert ctl.integral == pytest.approx(expect, abs=1e-12), f"slot {t} 后不符"


def test_deadzone_suppresses_the_integral(bcfg, market, traffic):
    """误差在死区内时积分不动 —— 否则噪声会被一路积分成漂移。

    构造方式是让花费**精确跟随**计划曲线，此时 ``e = 0`` 落在死区内。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    # 直接把进度摆到计划曲线上（e=0，落在死区内）。
    # 用 observe_slot 摆不出来：第 0 个 slot 的误差是 F(0)≈0.02，
    # 已经远在死区之外了 —— 这本身就是「死区只对小偏差生效」的体现。
    ctl.st.spent = ctl.planned_spend(0)
    for t in range(20):
        ctl.alpha(t)
        assert ctl.integral == 0.0, f"死区内积分不该动，slot {t}"
        # 让下一拍的进度仍然精确落在计划曲线上
        ctl.observe_slot(t, ctl.planned_spend(t + 1) - ctl.st.spent)


def test_integral_is_untouched_when_pid_is_disabled(bcfg, market, traffic):
    """PID 关掉时积分连累加都不该发生（B2 消融档）。"""
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5,
                               use_feedforward=True, use_pid=False)
    for t in range(20):
        ctl.alpha(t)
        ctl.observe_slot(t, 0.0)
    assert ctl.integral == 0.0


def test_integral_freezes_when_output_is_saturated(bcfg, market, traffic):
    """输出已经顶到 u_max 还在往同方向累加 → 冻结积分。

    不冻结的话，等偏差反向时要花十几个 slot 才能把积分放掉，
    这段时间消耗是失控的 —— 而且因为 α 被 u_max 挡住，
    从 α 的轨迹上完全看不出积分已经攒了一大坨。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    frozen: list[float] = []
    for t in range(12):
        ctl.alpha(t)
        frozen.append(ctl.integral)
        ctl.observe_slot(t, 0.0)       # 一直欠花 → 一直正误差
    # 后期应当被冻结：积分不再增长
    assert frozen[-1] == frozen[-2] == frozen[-3], (
        f"积分没有被冻结，仍在增长：{frozen[-4:]}")


def test_traffic_curve_is_normalized_with_a_night_trough():
    w = diurnal_weights(96)
    assert w.shape == (96,)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w > 0)
    F = cumulative_weights(w)
    assert F[0] < F[-1] == pytest.approx(1.0)
    assert np.all(np.diff(F) > 0)

    # 凌晨 3-5 点必须是低谷，晚 8 点必须是高峰
    hour = np.arange(96) // 4
    night = w[(hour >= 3) & (hour <= 5)].sum()
    evening = w[(hour >= 19) & (hour <= 21)].sum()
    assert evening > 3 * night, "晚高峰没有明显高于凌晨低谷，流量曲线形状不对"

    # 午夜两侧必须连续（环状距离），不能在 23:00 → 00:00 处断崖
    assert abs(w[-1] - w[0]) < 0.3 * w.max()
    with pytest.raises(ValueError, match="24 整除"):
        diurnal_weights(95)


def test_progress_curve_is_more_back_loaded_than_traffic():
    """进度基准曲线必须比流量曲线更靠后（价值密度叠加）。

    这是那个「系统性少花 1.6%」的根因。按价值出价的买手，
    花费密度正比于 ``流量 × 价值密度``，而价值密度晚高峰更高 ——
    所以它的自然花费曲线天然比流量曲线靠后。
    拿流量曲线当基准，控制器会一直误判「我落后了」，
    然后抬价把预算**提前**花在便宜时段，正好和 pacing 的目的相反。
    """
    w = diurnal_weights(96)
    F_traffic = cumulative_weights(w)
    F_spend = cumulative_weights(spend_weights(w))
    assert np.all(F_spend <= F_traffic + 1e-12), "进度曲线应当处处不晚于流量曲线"
    # 中午（slot 47 ≈ 12:00）之前必须能看出差别，否则等于没改。
    # 实测比值 0.68：花费曲线在中午只完成了 19.8%，流量曲线已经完成 29.0%。
    ratio = F_spend[47] / F_traffic[47]
    assert 0.5 < ratio < 0.95, (
        f"正午累计占比：花费曲线 {F_spend[47]:.3f} vs 流量曲线 {F_traffic[47]:.3f}"
        f"（比值 {ratio:.3f}）。太小说明价值密度叠得过头，"
        f"接近 1 说明价值密度根本没叠进去")


def test_solve_alpha_hits_the_spend_target(bcfg, market):
    """二分反解必须真的命中目标花费（单位：**元/次曝光**）。

    ⚠️ 左边乘了 ``cpm_to_impression``。第一版忘了这一步，
       于是「目标 0.004 元/次」被当成「0.004 元/千次」，
       求出来的 α 对应的真实花费只有目标的千分之一。
       这条断言就是钉这个单位的。
    """
    rng = np.random.default_rng(4)
    v = rng.lognormal(0.0, 1.0, 50_000) * 20.0     # 任意正的 CPM 价值分布
    to_imp = bcfg.cpm_to_impression
    # ⚠️ 目标必须**小于** ``market.mean_cpm × to_imp``（这里 = 0.025 元/次）：
    #    那是「无论出多高价都赢下每一次曝光」的封顶花费。
    #    目标超过它，二分区间里根本没有解，函数按约定返回上界，
    #    断言就会失败 —— 而且看起来像公式错了，其实是测试自己提了个
    #    物理上不可能的要求。
    ceiling = market.mean_cpm * to_imp
    for target in (0.002, 0.008, 0.018):
        assert target < ceiling
        a = solve_alpha(v, market, target)
        got = float(np.mean(expected_spend_cpm(a * v, market))) * to_imp
        assert got == pytest.approx(target, rel=1e-8), f"target={target}"


def test_solve_alpha_target_is_in_yuan_per_request_not_cpm(bcfg, market):
    """目标放大 1000 倍，解出的 α 必须**明显**变大（而不是几乎不变）。

    如果内部漏了换算，两个目标的比值会被压缩得看不出来 ——
    这条测试专门盯那个换算有没有生效。
    """
    rng = np.random.default_rng(41)
    v = rng.lognormal(0.0, 1.0, 20_000) * 20.0
    a_small = solve_alpha(v, market, 1e-4)
    a_big = solve_alpha(v, market, 1e-1)          # 1000 倍
    assert a_big > a_small * 3, (
        f"目标放大 1000 倍，α 只从 {a_small:.5f} 变到 {a_big:.5f} —— "
        f"元/千次 vs 元/次的换算很可能没生效")


def test_solve_alpha_is_monotone_in_the_target(bcfg, market):
    rng = np.random.default_rng(5)
    v = rng.lognormal(0.0, 1.0, 20_000) * 20.0
    a1 = solve_alpha(v, market, 0.005)
    a2 = solve_alpha(v, market, 0.02)
    assert a2 > a1, "目标花费越高，水位就该越高"


def test_solve_alpha_rejects_degenerate_input(bcfg, market):
    with pytest.raises(ValueError, match="正的曝光价值"):
        solve_alpha(np.zeros(10), market, 1.0)
    with pytest.raises(ValueError, match="目标花费"):
        solve_alpha(np.ones(10) * 5.0, market, 0.0)


def test_end_of_day_factor_shrinks_to_0_4(bcfg, market, traffic):
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic)
    assert ctl.end_of_day_factor(0) == 1.0
    assert ctl.end_of_day_factor(87) == 1.0          # 最后 8 个 slot 之前
    assert ctl.end_of_day_factor(88) == pytest.approx(1.0)
    assert ctl.end_of_day_factor(95) == pytest.approx(0.4)
    f = [ctl.end_of_day_factor(t) for t in range(88, 96)]
    assert all(b <= a for a, b in zip(f, f[1:])), "日末收紧必须是单调的"


def test_hard_gate_keeps_expected_spend_inside_the_budget(bcfg, market, traffic):
    """硬闸门把这一批的**期望**扣费压在剩余预算内。

    ⚠️ 是期望不是最坏情况 —— 这个取舍的代价实测过：按最坏情况（全赢、
       第一价）折算时闸门比期望大 5~7 倍，会在还剩 6 个 slot 时就启动，
       让 B4 系统性少花 1.7%。完整推导见 ``auction.budget_gate_scale``。

    尾部由 reserve 吸收：单个 slot 的实际扣费相对期望只有 4~5% 的波动
    （750+ 个请求的均值），比 reserve 小一个数量级。
    """
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic)
    to_imp = bcfg.cpm_to_impression
    for t in range(96):
        ctl.alpha(t)
        bid = np.full(5000, 200.0)                   # 贵得离谱的出价
        scale = ctl.budget_scale(bid)
        assert 0.0 <= scale <= 1.0
        exp = float(np.sum(expected_spend_cpm(bid * scale, market))) * to_imp
        assert exp <= ctl.remaining + 1e-9, (
            f"slot {t} 缩放后期望花费 {exp:.4f} 超过剩余预算 {ctl.remaining:.4f}")
        # 用**比期望大 20%** 的实现值推进：模拟尾部波动，
        # 这 20% 必须由 reserve 吃掉而不是撞破预算
        ctl.observe_slot(t, min(exp * 1.2, ctl.remaining))

    assert ctl.spent <= ctl.spendable + 1e-9

    # 预算耗尽后闸门必须关闭（这是最后一道红线，不能只靠缩放）
    ctl.st.spent = ctl.spendable
    assert ctl.budget_scale(np.full(10, 50.0)) == 0.0, "预算耗尽后闸门必须关闭"


def test_gate_leaves_small_batches_untouched(bcfg, market, traffic):
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic)
    assert ctl.budget_scale(np.full(3, 1.0)) == 1.0


def test_reset_day_clears_all_state(bcfg, market, traffic):
    """**日切重置**：跨天复用 λ 和积分，第二天开头会按昨天的偏差出价。"""
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                               traffic=traffic, alpha_init=0.5)
    for t in range(20):
        ctl.alpha(t)
        ctl.observe_slot(t, 5.0)
    assert ctl.st.spent > 0 and ctl.integral != 0 and ctl.alpha_ff != 0.5

    ctl.reset_day()
    assert ctl.st.spent == 0.0
    assert ctl.integral == 0.0
    assert ctl.err_smooth == 0.0
    assert ctl.alpha_ff == pytest.approx(0.5)
    assert ctl.lam == pytest.approx(2.0)
    assert ctl.st.slot == -1
    assert ctl.remaining == pytest.approx(ctl.spendable)


def test_initial_alpha_outside_the_band_is_rejected(bcfg, market, traffic):
    with pytest.raises(ValueError, match="alpha_min"):
        WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic,
                             alpha_init=99.0)


def test_negative_slot_spend_is_rejected(bcfg, market, traffic):
    ctl = WaterLevelController(bcfg, market, budget=TEST_BUDGET, traffic=traffic)
    ctl.alpha(0)
    with pytest.raises(ValueError, match="不能为负"):
        ctl.observe_slot(0, -1.0)


def test_wrong_traffic_length_is_rejected(bcfg, market):
    with pytest.raises(ValueError, match="流量曲线长度"):
        WaterLevelController(bcfg, market, traffic=np.ones(10))


def test_controller_ablation_switches_actually_disable(bcfg, market, traffic):
    """B2 关 PID、B3 关前馈 —— 开关不生效的话消融对比就是假的。"""
    pid_off = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                   traffic=traffic, alpha_init=0.5,
                                   use_feedforward=True, use_pid=False)
    for t in range(8):
        a = pid_off.alpha(t)
        assert a == pytest.approx(max(bcfg.alpha_min, min(
            bcfg.alpha_max, pid_off.alpha_ff)), rel=1e-12), "PID 关了但 u≠0"
        pid_off.observe_slot(t, 0.0)
    assert pid_off.alpha_ff > 0.5, "前馈开着，欠花应该把 α 抬上去"

    ff_off = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                  traffic=traffic, alpha_init=0.5,
                                  use_feedforward=False, use_pid=True)
    for t in range(20):
        ff_off.alpha(t)
        ff_off.observe_slot(t, 0.0)
    assert ff_off.alpha_ff == pytest.approx(0.5), "前馈关了但 α_ff 动了"


# ============================================================ 单日闭环
def test_stream_has_the_expected_shape(bcfg, market, traffic, stream):
    assert len(stream) == N_REQUESTS
    assert stream.slot.shape == (N_REQUESTS,)
    assert np.all(np.diff(stream.slot) >= 0), "请求必须按 slot 排好序"
    assert np.all(stream.market_bid_cpm > 0)
    # 抽样的市场价均值要对得上解析参数化
    assert float(stream.market_bid_cpm.mean()) == pytest.approx(
        market.mean_cpm, rel=0.02)


def test_expected_values_are_strictly_positive(stream):
    """**出价用的 LTV 是期望值，对每个人都严格为正。**

    第一版把「这个人实际付了多少钱」当 LTV，于是 90% 的请求价值为 0、
    出价全为 0。错在把实现值当成了期望值 ——
    这条断言就是钉死这件事的（详见 ``_synthetic_values`` 的 docstring）。
    """
    assert np.all(stream.pred_value_cpm > 0), "预测价值出现 0，出价会退化成不出价"
    assert np.all(stream.true_value_cpm > 0), "真实价值出现 0"
    # 零膨胀应该体现在**金额的实现值**上，而不是期望值上
    assert float((stream.realized_ltv == 0).mean()) < 0.05


def test_realized_ltv_is_unbiased_for_the_expected_ltv(stream):
    """``E[realized_ltv] = true_ltv`` —— 否则 realized_value 这个口径就没法用。"""
    ratio = float(stream.realized_ltv.mean() / stream.true_ltv.mean())
    assert ratio == pytest.approx(1.0, rel=0.05), f"比值 {ratio:.4f}"


def test_solved_water_levels_are_inside_the_operating_band(bcfg, market, traffic,
                                                          stream):
    """反解出来的水位必须落在 ``[alpha_min, alpha_max]`` 内。

    ``build_policies`` 会兜底夹一下，但**夹住就说明标定错了** ——
    热启动白做、控制器第一步就被 clip，现象只是「消耗怎么都跑不满」。
    这条测试盯的就是那个兜底有没有被动用。
    """
    n = len(stream)
    plan = TEST_BUDGET * bcfg.budget_safety
    for label, v in (("预测价值", stream.pred_value_cpm),
                     ("真实价值（B5 的初值）", stream.true_value_cpm)):
        a = solve_alpha(v, market, plan / n)
        assert bcfg.alpha_min < a < bcfg.alpha_max, (
            f"{label} 反解出的水位 {a:.4f} 超出 "
            f"[{bcfg.alpha_min}, {bcfg.alpha_max}] —— "
            f"预算/市场价/价值分布三者的相对标定需要重看")


def test_budget_is_actually_binding(bcfg, market, traffic, stream):
    """前提检查：如果预算根本花不完，整组 pacing 测试都是假测试。

    这条测试不测产品，测的是**测试本身有没有意义**，
    所以必须放在最前面。无约束（α=1）下的期望花费要明显超过预算。
    """
    unconstrained = float(np.mean(expected_spend_cpm(
        stream.true_value_cpm, market))) * bcfg.cpm_to_impression * N_REQUESTS
    assert unconstrained > 2.0 * TEST_BUDGET, (
        f"α=1 时的期望花费只有 {unconstrained:.0f} 元，"
        f"相对 {TEST_BUDGET} 元的预算约束不 binding —— "
        f"请调低 TEST_BUDGET 或调高 N_REQUESTS")


def test_value_density_peaks_in_the_evening(stream, bcfg):
    """时段必须影响**价值密度**，不能只影响流量多少。

    如果价值在一天里同分布，「早上花光预算」和「均摊全天」
    买到的价值就完全一样，pacing 只剩「别超支」这一个作用，
    B4 赢 B1 也就无从谈起 —— 第一版正是这样，
    于是 B1 靠**花得更多**拿到了更高的价值。
    """
    hour = stream.slot // 4
    peak = stream.true_value_cpm[(hour >= 19) & (hour <= 21)].mean()
    trough = stream.true_value_cpm[(hour >= 3) & (hour <= 5)].mean()
    assert peak > 1.8 * trough, (
        f"峰谷价值密度比只有 {peak / trough:.2f}，pacing 没有用武之地")


def test_b4_never_overspends_and_hits_the_budget(day_results):
    """**超支率 < 0.5%，预算误差 < 2%**。"""
    r = day_results["B4_lambda_pid"]
    assert r.overspend_frac <= 0.005, f"超支了 {r.overspend_frac * 100:.2f}%"
    assert r.budget_error < 0.02, f"预算误差 {r.budget_error * 100:.2f}%"


def test_b4_pacing_rmse_is_small(day_results):
    assert day_results["B4_lambda_pid"].pacing_rmse < 0.03


def test_b4_beats_fixed_bid(day_results):
    """主方案必须打赢「人工固定出价」这个基线。"""
    b4, b0 = day_results["B4_lambda_pid"], day_results["B0_fixed"]
    assert b4.expected_value > b0.expected_value * 1.05, (
        f"B4 {b4.expected_value:.0f} vs B0 {b0.expected_value:.0f}")
    assert b4.roi > b0.roi


def test_greedy_buys_the_wrong_hours_and_loses(day_results, bcfg):
    """B1 贪心：预算花在错误的时段，买到的价值更少。

    这是「pacing 为什么必要」的量化证明。注意对比的**口径**：

    贪心**不超支**（平台闸门会拦住它），所以两条策略花一样的钱 ——
    差别只在于「什么时候买」。贪心 α≡1，见到值就买，
    上午就把预算用光；而晚高峰的流量既多又好，
    轮到它时已经没钱了。B4 把预算均摊到全天，买到了更好的流量。

    ⚠️ 第一版这条测试写的是「贪心会超支」+「B4 价值更高」，
       两条都不成立 —— 贪心没闸门时确实超支，但**价值也更高**
       （花得多自然买得多），拿总价值比是在惩罚省钱的一方。
       正确的对照必须先把预算合规性对齐，再比价值。
    """
    g, b4 = day_results["B1_greedy"], day_results["B4_lambda_pid"]
    assert g.overspend_frac <= 0.005, "闸门没拦住贪心"
    assert g.expected_value < b4.expected_value, (
        f"贪心 {g.expected_value:.0f} vs B4 {b4.expected_value:.0f} —— "
        f"把预算花在长尾时段居然不吃亏，说明时段价值差异没起作用")
    assert g.roi < b4.roi, "同样的花费，贪心买到的价值应当更低"


def test_greedy_runs_out_of_money_before_the_peak(day_results):
    """贪心**在晚高峰之前就断粮** —— 这是它输掉的机制证据。

    实测（预算 350 元）：贪心把 69% 的预算花在前半天，
    晚高峰（19:00–22:00，一天里价值密度最高的 3 小时）只花掉 0.3%。
    它没有做错任何单次决策 —— 每一次出价都在按价值买 ——
    错的是**没有把预算留到更值钱的时段**。
    """
    g = day_results["B1_greedy"]
    first_half = g.cumulative_spend[47] / g.spend
    evening = (g.cumulative_spend[87] - g.cumulative_spend[75]) / g.spend
    assert first_half > 0.6, f"前半天只花了 {first_half * 100:.1f}%"
    assert evening < 0.05, f"晚高峰还花了 {evening * 100:.1f}%，没断粮"


def test_paced_policies_outspend_greedy_in_the_evening(day_results):
    """B4 在晚高峰买到的量应当**远多于**贪心。

    这是上面那条价值差异的**机制证据**：如果只是价值更高，
    说不清是「买得更多」还是「买得更准」。这里直接看晚间花费占比。
    """
    g, b4 = day_results["B1_greedy"], day_results["B4_lambda_pid"]
    evening = slice(76, 88)                      # 19:00 – 22:00
    g_eve = (g.cumulative_spend[evening][-1] - g.cumulative_spend[75]) / g.spend
    b4_eve = (b4.cumulative_spend[evening][-1] - b4.cumulative_spend[75]) / b4.spend
    assert b4_eve > 5 * g_eve, (
        f"晚高峰花费占比：B4 {b4_eve * 100:.1f}% vs 贪心 {g_eve * 100:.1f}%")


def test_stationary_market_is_paced_almost_perfectly_by_feedforward_alone(
        day_results):
    """平稳市场下，**前馈 λ 一个环就够** —— PID 在这条流上帮不上忙。

    这是实测结论，和设计直觉相反，值得原样写下来：

    | 策略 | pacing RMSE | 期望价值 |
    |---|---|---|
    | B2 仅前馈 λ | 0.0047 | 3566 |
    | B3 仅 PID | 0.0150 | 3533 |
    | B4 λ+PID | 0.0149 | 3533 |

    原因不难想清楚：前馈 λ 已经是一个作用在**累计花费误差**上的积分环
    （``err = (spent − planned)/plan``），而 PID 的 I 项是**同一个误差信号**
    上的另一个积分环，只是快了 4 倍。平稳市场里慢环自己就能收敛，
    快环的 P 项（``k_p = 6.0``）反而把一个已经接近零的误差放大后
    注入到 α 上，让轨迹更抖。

    ⚠️ 所以这条测试断言的是「B2 足够好」，**不是**「B4 更好」。
       把门槛调成 B4 胜出就是编数据。PID 的价值要在**非平稳**市场里才
       测得到，见 ``test_pid_pays_off_when_the_market_shifts_midday``。
    """
    b2, b3, b4 = (day_results["B2_lambda_only"],
                  day_results["B3_pid_only"],
                  day_results["B4_lambda_pid"])
    assert b2.pacing_rmse < 0.01, (
        f"平稳市场下前馈单独应当把 pacing 做到 1% 以内，实测 {b2.pacing_rmse:.4f}")
    assert b4.pacing_rmse < 0.03, f"B4 pacing {b4.pacing_rmse:.4f} 超出 3% 门槛"

    # 三个环的**最终**价值必须都在 ±3% 以内：谁也没把这事做坏
    best = max(b2.expected_value, b3.expected_value, b4.expected_value)
    for r in (b2, b3, b4):
        assert r.expected_value > 0.97 * best, (
            f"{r.name} 的价值 {r.expected_value:.0f} 比最好的 {best:.0f} 差 3% 以上")
        assert r.budget_error < 0.05, f"{r.name} 没花完预算，对比就不公平了"


def test_pid_pays_off_when_the_market_shifts_midday(bcfg, market, traffic):
    """**非平稳市场**才是快环的主场 —— 这里 PID 才真正赢。

    平稳流上 λ 一个环就够了（上一条测试）。但真实市场每天都在变：
    竞品突然加价、大促开始、素材衰退。这些是**小时级慢环跟不上**的。

    构造：12:00–14:00（slot 48–55）市场价**翻 2.5 倍**。
    此时出价按原水位几乎赢不到，消耗会掉下去 ——
    λ 每小时才更新一次、每步最多动 ``max_dlog_alpha = 0.10``，
    要 3~4 小时才能把水位提上来（而扰动只持续 2 小时，等它追上市场已经恢复）；
    PID 每个 slot 都在看进度误差，十几个 slot 内就能补上。

    断言的是**扰动窗口内的进度误差**，不是全天 RMSE：
    全天的 RMSE 会被 90 个没受扰动的 slot 稀释掉，测不出差异。
    """
    shock_lo, shock_hi, factor = 48, 56, 2.5
    shocked = make_stream(bcfg, market, n_requests=N_REQUESTS, seed=SIM_SEED,
                          traffic=traffic,
                          market_shock=(shock_lo, shock_hi, factor))
    pols = build_policies(bcfg, market,
                          pred_value_cpm=shocked.pred_value_cpm,
                          true_value_cpm=shocked.true_value_cpm,
                          budget=TEST_BUDGET, traffic=traffic,
                          plan_weights=shocked.plan_weights)
    res = {r.name: r for r in run_all_policies(shocked, pols, bcfg, market,
                                               budget=TEST_BUDGET, seed=SIM_SEED)}
    b2, b4 = res["B2_lambda_only"], res["B4_lambda_pid"]

    def window_rmse(r):
        """扰动窗口结束时的累计进度偏差（相对预算）。"""
        plan = TEST_BUDGET * bcfg.budget_safety
        cum_planned = r.planned_spend[shock_hi - 1]
        return abs(r.cumulative_spend[shock_hi - 1] - cum_planned) / plan

    e2, e4 = window_rmse(b2), window_rmse(b4)
    assert e4 < 0.5 * e2, (
        f"市场扰动窗口结束时：B2 进度偏差 {e2:.4f} vs B4 {e4:.4f}，"
        f"快环没有明显更快地把消耗拉回来")
    # 全天看两边都要把预算花完，否则「更快」没有意义
    for r in (b2, b4):
        assert r.budget_error < 0.03, f"{r.name} 扰动后没恢复，误差 {r.budget_error:.4f}"


def test_pid_never_makes_the_value_worse(day_results):
    """加环的代价必须在噪声内 —— 不允许「pacing 好了但买的更差」。

    实测三档价值差在 ±1% 以内（3566 / 3533 / 3533）。
    ⚠️ 断言里**没有**「B4 价值 > B2」。B2 反而略高，因为它的 λ
       一路滞后，恰好把更多预算推到价值密度更高的晚高峰 ——
       这是巧合，不是设计，不能把「滞后」当成值得追求的性质。
    """
    b2, b3, b4 = (day_results["B2_lambda_only"],
                  day_results["B3_pid_only"],
                  day_results["B4_lambda_pid"])
    best_single = max(b2.expected_value, b3.expected_value)
    assert b4.expected_value > 0.97 * best_single, (
        f"B4 {b4.expected_value:.0f} 比单环最好的 {best_single:.0f} 差了 3% 以上，"
        f"说明两个环互相打架而不是互补")


def test_b4_reaches_85pct_of_the_oracle(day_results):
    """B4 应达 B5（上帝视角）的 85–100%。

    低于 85% 说明模型误差或在线试错的代价太大；
    **高于 100% 则说明真值泄漏进了策略**（比如出价用了 true_value），
    所以这里是个双边断言 —— 上界比下界更重要。
    """
    b4, b5 = day_results["B4_lambda_pid"], day_results["B5_oracle"]
    ratio = b4.expected_value / b5.expected_value
    assert 0.85 <= ratio <= 1.0, f"B4/B5 = {ratio:.4f}"


def test_oracle_has_the_best_roi(day_results):
    """上帝视角的标准不是「花得最多」，而是**每一块钱买到的价值最高**。

    ⚠️ B5 的总价值只比 B4 高不到 1%，但它的花费更低 ——
       所以用总价值做断言会时灵时不灵（预算松的时候 B5 甚至更低），
       用 ROI 才是它的定义性质。
    """
    b5 = day_results["B5_oracle"]
    assert all(b5.roi >= r.roi * 0.99 for r in day_results.values()), (
        "B5 的 ROI 不是最高的：" +
        " ".join(f"{k}={v.roi:.2f}" for k, v in day_results.items()))
    best = max(r.expected_value for r in day_results.values())
    assert b5.expected_value > 0.99 * best


def test_pacing_parameters_are_mutually_consistent(bcfg):
    """``budget_safety + reserve_frac < 1`` —— 计划不能超过闸门上限。

    违反的后果：硬闸门**天天触发**。闸门是「前两环已经失效」时的最后保险，
    正常的一天里它不该出现；而它一旦天天触发，就说明计划本身就
    要求花掉超过闸门允许的钱，控制器永远差一口气。
    这种配置错误不会报错，只会让消耗稳定地差几个百分点。

    顺带也钉住下界：完美控制器的预算误差下界就是 ``1 − budget_safety``，
    所以「误差 < 2%」这个门槛要求 ``budget_safety ≥ 0.98``。
    """
    assert bcfg.budget_safety + bcfg.reserve_frac < 1.0, (
        f"budget_safety={bcfg.budget_safety} + reserve_frac="
        f"{bcfg.reserve_frac} ≥ 1，硬闸门会天天触发")
    assert 1.0 - bcfg.budget_safety < 0.02, (
        f"budget_safety={bcfg.budget_safety} 意味着预算误差下界 "
        f"{1 - bcfg.budget_safety:.3f}，已经吃满 2% 的门槛了")


def test_predictions_are_the_only_input_to_bidding(stream, bcfg, market, traffic):
    """**真值泄漏检测**：打乱 ``true_*`` 不得改变任何出价轨迹。

    这是整个模块最重要的一条测试。``DayStream`` 同时持有预测值和真值，
    一不小心就会在出价路径上用到真值 —— 用了之后所有指标都会变好
    （因为策略「作弊」了），而 B4 和 B5 的差距会消失。
    这条测试通过「只扰动真值、看 α 轨迹变不变」来钉死这件事。
    """
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(stream))
    shuffled = type(stream)(
        slot=stream.slot,
        pred_value_cpm=stream.pred_value_cpm,          # 预测值**不动**
        true_value_cpm=stream.true_value_cpm[perm],    # 真值全打乱
        market_bid_cpm=stream.market_bid_cpm,
        true_conv_prob=stream.true_conv_prob[perm],
        true_ltv=stream.true_ltv[perm],
        realized_ltv=stream.realized_ltv[perm],
        traffic=stream.traffic,
    )
    ctl_a = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                 traffic=traffic, alpha_init=0.4)
    ctl_b = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                 traffic=traffic, alpha_init=0.4)
    r1 = simulate_day(stream, ControllerPolicy(ctl_a, "a"), bcfg, market,
                      budget=TEST_BUDGET, seed=SIM_SEED)
    r2 = simulate_day(shuffled, ControllerPolicy(ctl_b, "b"), bcfg, market,
                      budget=TEST_BUDGET, seed=SIM_SEED)
    assert np.allclose(r1.alpha_path, r2.alpha_path)
    assert r1.spend == pytest.approx(r2.spend)
    assert r1.wins == r2.wins

    # 反过来：扰动预测值**必须**改变轨迹（否则说明出价压根没用到模型）
    shuffled_pred = type(stream)(
        slot=stream.slot,
        pred_value_cpm=stream.pred_value_cpm[perm],
        true_value_cpm=stream.true_value_cpm,
        market_bid_cpm=stream.market_bid_cpm,
        true_conv_prob=stream.true_conv_prob,
        true_ltv=stream.true_ltv,
        realized_ltv=stream.realized_ltv,
        traffic=stream.traffic,
    )
    ctl_c = WaterLevelController(bcfg, market, budget=TEST_BUDGET,
                                 traffic=traffic, alpha_init=0.4)
    r3 = simulate_day(shuffled_pred, ControllerPolicy(ctl_c, "c"), bcfg, market,
                      budget=TEST_BUDGET, seed=SIM_SEED)
    assert not np.allclose(r1.alpha_path, r3.alpha_path), \
        "打乱预测值却丝毫不影响出价 —— 出价没有真的用到模型"


def test_all_policies_share_one_request_stream(stream, bcfg, market, traffic):
    """配对比较：六条策略必须跑在同一批请求上。

    每条策略各抽一天的话，差异里会混进流量抽样噪声 ——
    早晚高峰的流量差异远大于策略差异。
    """
    pols = build_policies(bcfg, market, pred_value_cpm=stream.pred_value_cpm,
                          true_value_cpm=stream.true_value_cpm,
                          budget=TEST_BUDGET, traffic=traffic)
    res = run_all_policies(stream, pols, bcfg, market, budget=TEST_BUDGET,
                           seed=SIM_SEED)
    assert len(res) == 6
    assert {r.requests for r in res} == {N_REQUESTS}
    assert len({r.name for r in res}) == 6, "策略名字必须互不相同"


def test_b0_gate_blocks_after_budget_is_gone(bcfg, market):
    p = FixedBidPolicy(bid_cpm=30.0, budget=100.0, market=market)
    p.reset()
    p.observe(0, 100.0)
    assert p.gate_scale(np.full(10, 30.0)) == 0.0
    p.observe(1, 0.0)
    assert p.gate_scale(np.full(10, 30.0)) == 0.0


def test_greedy_is_paced_only_by_the_platform_gate(bcfg, market):
    """B1 的 α 恒为 1（没有 pacing），预算合规完全靠平台闸门。

    这两件事必须能分开看：α 轨迹上没有任何控制，
    而花费仍然不超支 —— 所以 B1 输给 B4 的原因只可能是「买的时点」，
    不是「花了多少钱」。
    """
    p = GreedyPolicy(budget=100.0, market=market)
    p.reset()
    assert all(p.alpha(t) == 1.0 for t in range(96)), "B1 不该有任何 pacing"
    assert p.gate_scale(np.full(10, 500.0)) == 1.0   # 还没花钱，闸门开着
    p.observe(0, 100.0)
    assert p.gate_scale(np.full(10, 500.0)) == 0.0   # 花光了，闸门关闭


def test_only_the_oracle_views_the_ground_truth(day_results):
    """**除 B5 之外，没有任何策略能看到 ``true_value``。**

    这是整个仿真的纪律线。泄漏的表现很隐蔽：所有指标都会变好
    （B4 逼近甚至超过 B5），而看不出哪里错了 ——
    因为是「数据太好了」，不是「代码报错了」。
    所以把开关做成策略类上的一个属性，这里逐个查。

    ⚠️ 第一版就是在这里出的错：``OraclePolicy`` 只把 ``alpha_star``
       换成了真值反解，出价却仍然走基类的 ``bid()``（用预测值）。
       结果是 B5「一半作弊」，比真正的上帝视角少花 1.2%，
       而 B4/B5 的比值看起来完全正常（0.98）。
    """
    assert day_results["B5_oracle"] is not None
    assert OraclePolicy.views_truth is True, "B5 必须能看真值，否则它不是上帝视角"
    for cls in (FixedBidPolicy, GreedyPolicy, ControllerPolicy):
        assert cls.views_truth is False, f"{cls.__name__} 打开了真值开关"


def test_day_result_helpers_are_consistent(day_results):
    r = day_results["B4_lambda_pid"]
    assert r.budget_error == pytest.approx(abs(r.overspend_frac))
    assert 0.0 <= r.win_rate <= 1.0
    assert r.roi == pytest.approx(r.expected_value / r.spend)
    assert len(r.cumulative_spend) == 96
    assert np.all(np.diff(r.cumulative_spend) >= -1e-9)
    assert r.spend == pytest.approx(r.cumulative_spend[-1], rel=1e-9)
    assert "B4" in r.summary()


def test_overspend_rate_across_all_paced_policies(day_results):
    """所有带闸门的策略都必须守住红线 —— 不只是主方案。"""
    for name in ("B2_lambda_only", "B3_pid_only", "B4_lambda_pid",
                 "B5_oracle", "B0_fixed"):
        assert day_results[name].overspend_frac <= 0.005, f"{name} 超支"
