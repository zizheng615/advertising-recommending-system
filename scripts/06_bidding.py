"""06 智能出价与水位控制：在单日预算约束下跑完一天的竞价。

产出
----
``artifacts/metrics/06_bidding_{scale}.json``

**这一步在证明什么**

出价链路的正确性**无法从单测看出来**。单测能验证「一步算得对」，
但 PID 的积分抗饱和、λ 的收敛速度、硬闸门的边界，全都是**轨迹性质**
—— 只有在「一天 96 个 slot、预算有限、流量有昼夜起伏」的闭环里才暴露。
所以这一步是一整天的仿真，而不是又一组单元断言。

**三个环各管什么（这是「PID 到底控制什么」的答案）**

| 环 | 时间尺度 | 职责 | 决定了什么 |
|---|---|---|---|
| 前馈 λ | 每小时 | 每份预算的经济价值是多少 | 买**什么**（慢、平滑、有经济含义） |
| 反馈 PID | 每 15min | 预测错了怎么补 | 花**多少**（快、有界） |
| 硬闸门 | 每请求 | 绝不超支这条红线 | 兜底（触发即说明前两环已失效） |

**两个价值口径，绝不能混**

``pred_value`` 是策略唯一能看到的东西；``true_value`` 只用于打分。
真实环境里没有后者，这里能拿到是因为数据是模拟的 —— 用它打分让策略
对比的方差大幅下降，但**它不得出现在 ``DayStream`` 之外的任何策略里**，
否则 B5 上帝视角和 B4 的差距会消失，而所有指标看起来都很好。

**一个和直觉相反、但必须如实记录的结论**

平稳市场下**前馈 λ 一个环就够了**，PID 帮不上忙 —— 它的 I 项是同一个
累计误差信号上的第二个积分环。PID 的价值只在**非平稳市场**（竞品突然
加价、大促开始）里才体现得出来，因为那时慢环每小时最多动
``max_dlog_alpha`` 的步长根本追不上。所以这里跑两遍：
一遍平稳、一遍带市场扰动，**分别报数，不混在一起**。

**预算为什么要按流量反解，而不是直接用 ``cfg.daily_budget``**

预算约束必须是**真正 binding** 的，否则「pacing 有什么用」根本演示不出来：
无约束期望花费 U 只有预算的 1.4 倍时，贪心全天都有钱花，实测它甚至
**略胜** paced 策略 —— 那不是 pacing 没用，是这道题里它没有发挥空间。
所以预算按 ``U / 3.6`` 反解，并且**当场验证** U/B 确实落在 binding 区间
（证据：B0 固定出价能把预算花到 2% 以内 —— 花不完就说明约束不紧）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402

from scripts._common import (Timer, banner, cfg_for, log,    # noqa: E402
                             parse_scale, save_metrics, show_table)
from src.bidding.auction import (MarketConfig,               # noqa: E402
                                 expected_spend_cpm)
from src.bidding.bidder import build_policies                # noqa: E402
from src.bidding.controller import diurnal_weights           # noqa: E402
from src.bidding.sim_day import (make_stream, run_all_policies,  # noqa: E402
                                 simulate_day)
from src.common.seed import set_seed                        # noqa: E402

# 请求数只影响仿真的统计分辨率，不影响任何结论的量纲。
N_REQUESTS = 200_000
SIM_SEED = 2024

# 无约束期望花费 / 预算 的倍数。见模块 docstring：
# 太小 → 贪心全天有钱花，pacing 的价值测不出来；
# 太大 → 所有策略日初就断粮，差异被闸门抹平。
BUDGET_TENSION = 3.6

# 非平稳市场：12:00–14:00（slot 48–56）市场价翻 2.5 倍。
# 窗口只持续 2 小时，而 λ 每小时才动一次、单步有上限 —— 慢环追不上。
SHOCK = (48, 56, 2.5)

# 晚高峰 19:00–22:00，一天里价值密度最高的 3 小时
EVENING = slice(76, 88)


def unconstrained_spend(values_cpm: np.ndarray, market: MarketConfig,
                        cfg) -> float:
    """「见到值就全额买、且不受预算限制」的期望花费（元）。

    用**解析式**而不是跑一遍仿真：这是标定预算用的规划量，
    带抽样噪声的话每次跑出来的预算都不一样。
    """
    return float(expected_spend_cpm(values_cpm, market).sum()
                 * cfg.cpm_to_impression)


def dual_gradient_sign_check(cfg, market, traffic, alpha_init: float,
                             budget: float) -> dict:
    """对偶梯度**符号**的单步自检 —— 最容易写反、又不报错的一处。

    超支（``S > B``）时 λ 必须**变大** → α = 1/λ 变小 → 消耗下降。
    写反的话整个系统会正反馈发散：花超了就把水位提得更高，
    第一个 slot 就把预算打光，而现象看起来像「PID 参数不对」——
    然后去调 PID，越调越远。

    ⚠️ 前馈 λ **只在小时边界更新**（``observe_slot`` 里
       ``(slot+1) % slots_per_hour == 0``），所以自检必须走完一整个小时
       才能看到水位变化。只喂一个 slot 会得到「α 没动」，
       然后误判成「符号写反了」—— 实际上只是还没到更新时机。
       ``alpha_ff`` 是公开属性，不需要碰 ``_begin_slot`` 这类私有方法。
    """
    from src.bidding.controller import WaterLevelController
    slots_per_hour = cfg.t_slots // 24

    def drive(total_frac: float) -> tuple[float, float]:
        """第一个小时里总共花掉 ``total_frac × 全天计划``，返回 (前, 后) 水位。"""
        ctl = WaterLevelController(cfg, market, budget=budget,
                                   traffic=traffic, alpha_init=alpha_init)
        ctl.alpha(0)
        a_before = ctl.alpha_ff
        per_slot = total_frac * ctl.plan / slots_per_hour
        for s in range(slots_per_hour):
            ctl.alpha(s)                 # 走公开路径：α → 汇报花费
            ctl.observe_slot(s, per_slot)
        return a_before, ctl.alpha_ff

    a0_over, a1_over = drive(0.5)        # 第一个小时就花掉半天预算
    a0_under, a1_under = drive(0.0)      # 一点没花
    return {
        "alpha_before_overspend": a0_over,
        "alpha_after_overspend": a1_over,
        "alpha_before_underspend": a0_under,
        "alpha_after_underspend": a1_under,
        "overspend_lowers_alpha": a1_over < a0_over,
        "underspend_raises_alpha": a1_under > a0_under,
    }


def evening_share(r, slot_hi: int = 88, slot_lo: int = 76) -> float:
    """晚高峰（19:00–22:00）的花费占全天比例。"""
    return float((r.cumulative_spend[slot_hi - 1]
                  - r.cumulative_spend[slot_lo - 1]) / max(r.spend, 1e-9))


def first_half_share(r) -> float:
    return float(r.cumulative_spend[47] / max(r.spend, 1e-9))


def run_all(cfg, market, traffic, stream, budget: float) -> dict:
    pols = build_policies(cfg, market,
                          pred_value_cpm=stream.pred_value_cpm,
                          true_value_cpm=stream.true_value_cpm,
                          budget=budget, traffic=traffic,
                          plan_weights=stream.plan_weights)
    return {r.name: r for r in run_all_policies(stream, pols, cfg, market,
                                                budget=budget, seed=SIM_SEED)}


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)

    market = MarketConfig.from_mean(cfg.market_cpm_mean, cfg.market_sigma_m)
    traffic = diurnal_weights(cfg.t_slots)

    with Timer(f"06 构造请求流（{scale}）"):
        stream = make_stream(cfg, market, n_requests=N_REQUESTS,
                             seed=SIM_SEED, traffic=traffic)

    # ---------------- 预算标定 ----------------
    u_pred = unconstrained_spend(stream.pred_value_cpm, market, cfg)
    u_true = unconstrained_spend(stream.true_value_cpm, market, cfg)
    budget = round(u_pred / BUDGET_TENSION, -1)      # 取整到十元，便于复述
    tension = u_pred / budget
    log.info("无约束期望花费：按预测 %.0f 元 / 按真值 %.0f 元；"
             "预算 %.0f 元 → U/B = %.2f（目标 %.1f，约束必须 binding）",
             u_pred, u_true, budget, tension, BUDGET_TENSION)

    sign = dual_gradient_sign_check(
        cfg, market, traffic, alpha_init=1.0, budget=budget)
    banner("对偶梯度符号自检（单步）")
    log.info("超支时 α_ff %.4f → %.4f（应下降）%s", sign["alpha_before_overspend"],
             sign["alpha_after_overspend"],
             "✅" if sign["overspend_lowers_alpha"] else "❌ 符号写反了")
    log.info("欠花时 α_ff %.4f → %.4f（应上升）%s", sign["alpha_before_underspend"],
             sign["alpha_after_underspend"],
             "✅" if sign["underspend_raises_alpha"] else "❌ 符号写反了")

    # ---------------- 平稳市场：六条策略各跑一天 ----------------
    with Timer(f"06 平稳市场跑完 B0–B5（{scale}）"):
        res = run_all(cfg, market, traffic, stream, budget)

    rows = [{"策略": r.name, "花费": r.spend, "预算误差": r.overspend_frac,
             "期望价值": r.expected_value, "ROI": r.roi,
             "赢率": r.win_rate, "pacing": r.pacing_rmse,
             "闸门": r.gate_hits} for r in res.values()]
    banner("平稳市场：B0–B5 全天结果")
    show_table(rows, ["策略", "花费", "预算误差", "期望价值", "ROI", "赢率",
                      "pacing", "闸门"], "一天的闭环")

    b0, b1 = res["B0_fixed"], res["B1_greedy"]
    b2, b3, b4 = res["B2_lambda_only"], res["B3_pid_only"], res["B4_lambda_pid"]
    b5 = res["B5_oracle"]

    lift_vs_fixed = b4.expected_value / b0.expected_value - 1.0
    ratio_vs_oracle = b4.expected_value / b5.expected_value
    banner("主结论")
    log.info("B4（λ+PID）相对 B0（人工固定出价）：价值 %+.1f%%，ROI %.2f → %.2f",
             lift_vs_fixed * 100, b0.roi, b4.roi)
    log.info("B4 / B5（上帝视角上界）= %.4f（目标 0.85–1.00）；"
             "两者差的就是「模型误差 + 在线试错的代价」", ratio_vs_oracle)
    g_eve, b4_eve = evening_share(b1), evening_share(b4)
    log.info("B1 贪心（α≡1，无 pacing）：前半天花掉 %.1f%% 的预算，"
             "晚高峰只剩 %.2f%% —— 它每一次出价都没错，错在**没把预算留到更值钱的时段**",
             first_half_share(b1) * 100, g_eve * 100)
    # ⚠️ 贪心的晚高峰占比实测是**精确的 0**（预算在晚高峰前就花光了），
    #    所以这里不能直接写 "是它的 x 倍" —— 会打印出 2.99e8 这种数，
    #    看起来像 bug，其实只是分母为 0。分两种情形说清楚。
    if g_eve > 0:
        log.info("同期 B4 的晚高峰花费占比 %.1f%%，是贪心的 %.1f 倍",
                 b4_eve * 100, b4_eve / g_eve)
    else:
        log.info("同期 B4 的晚高峰花费占比 %.1f%% —— 贪心是精确的 0%%，"
                 "它在这 3 小时里**一分钱都没花出去**（预算更早就见底了）",
                 b4_eve * 100)

    # ---------------- 平稳市场下三个环的对照 ----------------
    banner("平稳市场下三个环的对照（结论和直觉相反，如实记录）")
    show_table([{"策略": r.name, "pacing RMSE": r.pacing_rmse,
                 "期望价值": r.expected_value, "预算误差": r.overspend_frac}
                for r in (b2, b3, b4)],
               ["策略", "pacing RMSE", "期望价值", "预算误差"], "λ / PID / 两者")
    log.info("前馈 λ 本身就是一个作用在累计花费误差上的**积分环**；"
             "PID 的 I 项是同一个误差信号上的第二个积分环，只是快了 4 倍。"
             "平稳市场里慢环自己就能收敛，快环的 P 项（k_p=%.1f）"
             "反而把已接近零的误差放大后注入 α，轨迹更抖",
             cfg.k_p)

    # ---------------- 非平稳市场：快环的主场 ----------------
    with Timer(f"06 非平稳市场（{scale}）"):
        shocked = make_stream(cfg, market, n_requests=N_REQUESTS,
                              seed=SIM_SEED, traffic=traffic,
                              market_shock=SHOCK)
        res_s = run_all(cfg, market, traffic, shocked, budget)

    lo, hi, factor = SHOCK
    plan = budget * cfg.budget_safety

    def window_err(r) -> float:
        """扰动窗口结束时的累计进度偏差（相对计划预算）。"""
        return float(abs(r.cumulative_spend[hi - 1]
                         - r.planned_spend[hi - 1]) / plan)

    e2, e4 = window_err(res_s["B2_lambda_only"]), window_err(res_s["B4_lambda_pid"])
    banner(f"非平稳市场（slot {lo}–{hi} 市场价 ×{factor}）：快环的主场")
    show_table([{"策略": r.name, "窗口末进度偏差": window_err(r),
                 "全天预算误差": r.overspend_frac,
                 "期望价值": r.expected_value}
                for r in res_s.values()],
               ["策略", "窗口末进度偏差", "全天预算误差", "期望价值"],
               "扰动窗口结束时的控制精度")
    log.info("慢环 λ 每小时最多动 max_dlog_alpha=%.2f，扰动只持续 %d 小时 —— "
             "等它追上，市场已经恢复了；PID 每个 slot 都在看进度误差。"
             "窗口末进度偏差：B2 %.4f vs B4 %.4f，快环快 %.1f 倍",
             cfg.max_dlog_alpha, (hi - lo) // 4, e2, e4, e2 / max(e4, 1e-9))

    # ---------------- 门槛 ----------------
    problems = []
    # 1. 绝不超支：这是红线，所有策略都必须守（闸门是平台的预算保护，
    #    不是某条策略的功劳 —— 所以这里对所有策略一视同仁）
    for r in res.values():
        if r.overspend_frac > 0.005:
            problems.append(f"{r.name} 超支 {r.overspend_frac * 100:.2f}%")
    # 2. 主方案的预算合规与 pacing
    if b4.budget_error >= 0.02:
        problems.append(f"B4 预算误差 {b4.budget_error * 100:.2f}% ≥ 2%")
    if b4.pacing_rmse >= 0.03:
        problems.append(f"B4 pacing RMSE {b4.pacing_rmse:.4f} ≥ 0.03")
    # 3. 智能出价必须打赢固定出价 —— 这是职责点 3 的核心主张
    if not b4.expected_value > b0.expected_value * 1.05:
        problems.append(f"B4 价值 {b4.expected_value:.0f} 没有超过 B0 "
                        f"{b0.expected_value:.0f} 的 105%")
    if not b4.roi > b0.roi:
        problems.append(f"B4 ROI {b4.roi:.3f} 不高于 B0 {b0.roi:.3f}")
    # 4. 预算约束确实 binding（否则上面的对比没有意义）
    if b0.budget_error >= 0.05:
        problems.append(f"B0 固定出价都花不完预算（误差 {b0.budget_error * 100:.2f}%），"
                        f"说明约束不紧，pacing 的价值无法体现 —— "
                        f"检查 BUDGET_TENSION={BUDGET_TENSION}")
    # 5. 贪心必须输，且输在机制上（买错了时段）
    if not b1.expected_value < b4.expected_value:
        problems.append(f"贪心价值 {b1.expected_value:.0f} 不低于 B4 "
                        f"{b4.expected_value:.0f} —— 时段价值差异没起作用")
    if first_half_share(b1) <= 0.6:
        problems.append(f"贪心前半天只花了 {first_half_share(b1) * 100:.1f}%，"
                        f"没有在早段打光预算，「买错时段」的机制不成立")
    # 「贪心在晚高峰断粮」是它输掉的**机制证据**，比价值差异本身更有说服力：
    # 单看价值更高说不清是「买得更多」还是「买得更准」。
    # ⚠️ 这里断言绝对阈值而不是「B4 是它的 N 倍」：贪心的晚高峰占比实测
    #    恰好为 0，任何以它为分母的比较都会变成 1/0。
    if not g_eve < 0.05:
        problems.append(f"贪心的晚高峰花费占比 {g_eve * 100:.1f}% ≥ 5%，"
                        f"它没有在晚高峰前断粮")
    if not b4_eve > 0.15:
        problems.append(f"B4 的晚高峰花费占比只有 {b4_eve * 100:.1f}%，"
                        f"没有把预算留到价值密度最高的时段")
    # 6. 上界：双边断言，**上界比下界更重要**
    #    > 1.0 说明真值泄漏进了策略（比如 B4 偷偷用了 true_value）
    if not 0.85 <= ratio_vs_oracle <= 1.0:
        problems.append(f"B4/B5 = {ratio_vs_oracle:.4f} 落在 [0.85, 1.00] 之外"
                        f"（>1 意味着真值泄漏进了策略）")
    if not all(b5.roi >= r.roi * 0.99 for r in res.values()):
        problems.append("B5 上帝视角的 ROI 不是最高的")
    # 7. 对偶梯度符号
    if not sign["overspend_lowers_alpha"]:
        problems.append("超支时水位没有下降 —— 对偶梯度符号写反了")
    if not sign["underspend_raises_alpha"]:
        problems.append("欠花时水位没有上升 —— 对偶梯度符号写反了")
    # 8. 非平稳市场：快环必须明显更快
    if not e4 < 0.5 * e2:
        problems.append(f"扰动窗口末进度偏差 B2 {e2:.4f} vs B4 {e4:.4f}，"
                        f"快环没有明显更快地把消耗拉回来")
    for r in (res_s["B2_lambda_only"], res_s["B4_lambda_pid"]):
        if r.budget_error >= 0.03:
            problems.append(f"扰动后 {r.name} 没恢复，预算误差 "
                            f"{r.budget_error * 100:.2f}%")
    # 9. 加环不允许让「买到的价值」变差
    best_single = max(b2.expected_value, b3.expected_value)
    if not b4.expected_value > 0.97 * best_single:
        problems.append(f"B4 价值 {b4.expected_value:.0f} 比单环最好的 "
                        f"{best_single:.0f} 差 3% 以上 —— 两个环在互相打架")

    def r_json(r) -> dict:
        return {"spend": r.spend, "overspend_frac": r.overspend_frac,
                "budget_error": r.budget_error,
                "expected_value": r.expected_value,
                "realized_value": r.realized_value, "roi": r.roi,
                "wins": r.wins, "win_rate": r.win_rate,
                "pacing_rmse": r.pacing_rmse, "gate_hits": r.gate_hits,
                "first_half_share": first_half_share(r),
                "evening_share": evening_share(r)}

    metrics = {
        "scale": scale,
        "setup": {"n_requests": N_REQUESTS, "n_slots": cfg.t_slots,
                  "market_mean_cpm": market.mean_cpm,
                  "market_sigma_m": market.sigma_m,
                  "budget": budget, "budget_tension": tension,
                  "unconstrained_spend_pred": u_pred,
                  "unconstrained_spend_true": u_true,
                  "shock": {"slot_lo": lo, "slot_hi": hi, "factor": factor}},
        "dual_gradient_sign": sign,
        "stationary": {k: r_json(v) for k, v in res.items()},
        "headline": {"lift_vs_fixed_bid": lift_vs_fixed,
                     "ratio_b4_to_oracle": ratio_vs_oracle},
        "nonstationary": {
            "per_policy": {k: {**r_json(v), "window_error": window_err(v)}
                           for k, v in res_s.items()},
            "window_error_b2": e2, "window_error_b4": e4,
            "speedup": e2 / max(e4, 1e-9),
        },
        "problems": problems,
    }
    save_metrics("06_bidding", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：无超支、B4 预算误差 %.2f%% / pacing %.4f，"
             "价值比固定出价高 %.1f%%、达上帝视角的 %.1f%%；"
             "非平稳市场下快环把窗口偏差从 %.4f 压到 %.4f",
             b4.budget_error * 100, b4.pacing_rmse, lift_vs_fixed * 100,
             ratio_vs_oracle * 100, e2, e4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
