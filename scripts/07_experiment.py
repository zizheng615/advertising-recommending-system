"""07 科学实验验证：AA 干跑排除分流偏差，再跑完整业务周期的 AB。

产出
----
``artifacts/metrics/07_experiment_{scale}.json``

**为什么第一步必须是 AA 而不是 AB**

AB 实验的分流链路会引入**内生性偏差**：哈希有偏（某类 ID 集中进一组）、
缓存串组、某一组的日志丢事件、两组抢同一台机器的资源。这些偏差在 AB 里
**看不出来** —— 它们混淆在真实的处理效应里，表现得就像「效果不错」。
只有让两组跑**完全相同**的策略（此时 H0 为真，真实差异恰好为 0），
偏差才会露出来。

**AA 的正确断言方式（这一步最容易被写错的地方）**

H0 为真时 p 值**必然**服从 U(0,1)，所以单次 AA 本来就有 5% 的概率显著。
因此：

* **不能**断言「跑一次 AA，p > 0.05」—— 那是 5% 概率失败的断言，
  写进 CI 就是每 20 次红一次，最后被人加 `flaky` 标注屏蔽掉，
  反而把真正的分流 bug 一起掩盖；
* 正确的断言是**分布**：跑 N 次独立 AA（每次换 exp_id，**重新生成数据
  且重新分流**），断言 p 值服从 U(0,1)，且假阳性率落在 [0.01, 0.12]。

⚠️ 每一次 AA 都必须**重新生成数据**。拿同一份数据只换分流种子，
N 个 p 值之间高度相关，而 KS 检验假设 p 值独立 —— 零分布失真，
检验本身失效。这一点很容易忽略，因为它跑得更快、看起来也没问题。

**功效分析会真的改变设计（这一步的产出不是数字，是决策）**

用户级指标的设计效应 ``deff`` 实测在 6 左右 —— 也就是说
**有效样本量只有名义的 1/6**。忽略它的后果不是「功效低一点」，
而是「按朴素公式算够了的样本量，实际只有三成功效」，
实验做完了却什么都测不出来。所以这里把两个数并排报出来。

**比率指标不能用普通 t 检验**：CVR 的分母（点击）本身就是随机变量，
分子分母都随机 → 用 Delta method 线性化方差。这是 ``stats`` 模块里
``ratio_delta_test`` 存在的原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402

from scripts._common import (Timer, banner, cfg_for, log,    # noqa: E402
                             parse_scale, save_metrics, show_table)
from src.common.seed import set_seed                        # noqa: E402
from src.experiment import power as PW                      # noqa: E402
from src.experiment import runner as RUN                    # noqa: E402
from src.experiment import stats as ST                      # noqa: E402

# ---------------- 规模 ----------------
# AA：次数由 cfg.aa_replications 决定（mini 档 60）。每次都要
# 重新生成数据 + 重新分流，所以这里的用户数不能太大 ——
# 6 千用户已经足够让单次的 p 值有分辨力。
AA_N_USERS = 6_000

# AB：完整业务周期 28 天（含 4 个周末）。用户数按功效分析的结果定，
# 见 ``main`` 里 power 那一段 —— 这里先给一个初值，
# 报告里会说明它对应的**实际功效**是多少，而不是假装足够。
AB_N_USERS = 100_000
AB_DAYS = 28
AB_SEED = 2024

# 处理效应：CVR **相对**提升 5%（就是功效分析里那个 MDE）
AB_CVR_LIFT = 0.05
# 附带验证「处理只覆盖一部分流量」时检验估得准不准。
# 真实 ATE = 0.20 × 0.30 = 0.06。
# ⚠️ 重复次数不能降到 1：单次估计的标准差实测有 0.0165（见 main 里的
#    长注释），和要检验的 0.06 是同一个量级，单次结果没有分辨力。
COVERAGE = 0.30
COVERAGE_LIFT = 0.20
COVERAGE_REPS = 10


def aa_report(cfg) -> dict:
    with Timer("07 AA 干跑（两组跑完全相同的策略）"):
        aa = RUN.run_aa(cfg, n_users=AA_N_USERS,
                        replications=cfg.aa_replications, seed=7)
    banner("AA 干跑：断言的是**分布**，不是单次 p 值")
    log.info("%s", aa.summary())
    log.info("单次 AA 的 p 值本身就服从 U(0,1)，所以「某一次 p>0.05」"
             "不是通过条件 —— 它本来就有 5%% 的概率失败。"
             "这里看的是 %d 次重复的整体分布",
             aa.p_values.size)
    log.info("p 值分位：p10=%.3f 中位=%.3f p90=%.3f（U(0,1) 下应约为 "
             "0.10 / 0.50 / 0.90）",
             *np.percentile(aa.p_values, [10, 50, 90]))
    return {
        "replications": int(aa.p_values.size),
        "n_per_group": int(aa.n_per_group),
        "ks_pvalue": aa.ks_p_value,
        "false_positive_rate": aa.false_positive_rate,
        "srm_min_pvalue": float(aa.srm_p_values.min()),
        "p10": float(np.percentile(aa.p_values, 10)),
        "median": float(np.median(aa.p_values)),
        "p90": float(np.percentile(aa.p_values, 90)),
    }


def power_section(cfg, data) -> dict:
    """功效分析：把「需要多少样本」和「有这么多能做到什么」一次算清。

    ⚠️ ``available_units`` 是**观测单元数（点击）**，不是用户数。
       直接把用户数传进去会同时漏掉「人均点击数」和 ``deff`` 两个因子，
       结果能差两个数量级 —— 这是 ``plan_experiment`` 文档里专门
       警告过的一个坑。
    """
    clicks = float(data.clicks.sum())
    n_users = len(data)
    units_per_user = clicks / n_users
    deff = ST.design_effect(data.conversions, data.clicks)
    baseline = float(data.conversions.sum() / clicks)

    plan = PW.plan_experiment(baseline, AB_CVR_LIFT,
                              available_units=clicks,
                              units_per_user=units_per_user, deff=deff)
    naive = PW.plan_experiment(baseline, AB_CVR_LIFT,
                               available_units=clicks,
                               units_per_user=units_per_user, deff=1.0)

    banner("功效分析（产出不是数字，是**决策**）")
    log.info("基线 CVR = %.4f，人均点击 %.1f 次，设计效应 deff = %.2f —— "
             "也就是**有效样本量只有名义的 1/%.1f**",
             baseline, units_per_user, deff, deff)
    log.info("MDE 取相对 %.0f%%（%.4f 绝对）时：", AB_CVR_LIFT * 100,
             baseline * AB_CVR_LIFT)
    show_table([
        {"口径": "朴素（deff=1）", "每臂需要点击": naive["units_per_arm"],
         "每臂需要用户": naive["users_per_arm"],
         "现有样本下的功效": naive["achieved_power"]},
        {"口径": f"deff 修正（{deff:.1f}）", "每臂需要点击": plan["units_per_arm"],
         "每臂需要用户": plan["users_per_arm"],
         "现有样本下的功效": plan["achieved_power"]},
    ], ["口径", "每臂需要点击", "每臂需要用户", "现有样本下的功效"],
        "两个口径并排")
    log.info("现有 %d 用户（%.0f 次点击）下：deff 修正后的功效 %.1f%%，"
             "朴素公式却报 %.1f%% —— 相差 %.0f 个百分点，"
             "而参考的朴素公式在任何报表里都不会报错",
             n_users, clicks, plan["achieved_power"] * 100,
             naive["achieved_power"] * 100,
             (naive["achieved_power"] - plan["achieved_power"]) * 100)
    log.info("这个样本量在 deff 修正口径下能测出的最小提升是 %.2f%%（相对）",
             plan["mde_at_available"] * 100)
    return {"baseline_cvr": baseline, "units_per_user": units_per_user,
            "deff": deff, "n_users": n_users, "clicks": clicks,
            "plan": {k: v for k, v in plan.items()},
            "naive": {k: v for k, v in naive.items()}}


def ab_report(cfg, data, split, rep, true_lift: float) -> dict:
    """把一次 AB 的关键输出打出来，并检查效应量估得准不准。"""
    banner(f"AB 实验（{rep.exp_id}）：{rep.n_control} vs {rep.n_treatment} 用户")
    log.info("%s", rep.srm)
    log.info("分流健康：%s", "是 ✅" if rep.healthy else "否 ❌ —— 以下数字不可信")

    p = rep.primary
    log.info("主指标 %s：对照 %.4f → 实验 %.4f，相对提升 %+.2f%%，"
             "p = %.3g（方法：%s）", p.metric, p.control, p.treatment,
             p.lift * 100, p.p_value, p.method)
    log.info("真实 ATE = %+.2f%%，估计值 %+.2f%% —— 偏差 %.3f 个百分点，"
             "估计值/真值 = %.2f", true_lift * 100, p.lift * 100,
             (p.lift - true_lift) * 100, p.lift / true_lift if true_lift else 0.0)
    lo, hi = ST.relative_lift_ci(p)
    log.info("95%% 置信区间 [%+.2f%%, %+.2f%%]，%s", lo * 100, hi * 100,
             "覆盖真实值 ✅" if lo <= true_lift <= hi else "**没有覆盖真值** ❌")

    rows = [{"指标": k, "相对提升": v.lift, "p": v.p_value,
             "BH 显著": bool(rep.bh_rejected.get(k, False))}
            for k, v in rep.secondary.items()]
    banner("次要指标（多重比较用 BH 校正到 q = %.2f）" % cfg.bh_q)
    show_table(rows, ["指标", "相对提升", "p", "BH 显著"], "次要指标")

    for k, v in rep.guardrail.items():
        log.info("护栏 %s：相对变化 %+.2f%%，p = %.3g", k, v.lift * 100,
                 v.p_value)

    if rep.cuped:
        banner("CUPED：用**实验前**的协变量压方差")
        log.info("ρ = %.3f，方差下降 %.1f%%", rep.cuped.get("rho", float("nan")),
                 rep.cuped.get("variance_reduction", 0.0) * 100)
        log.info("⚠️ 协变量必须来自实验之前。用实验期数据做协变量会把"
                 "处理效应本身减掉一部分，方差下降很好看，效应也被抹平了")

    days = rep.daily_lift
    same_sign = float(np.mean(np.sign(days) == np.sign(np.mean(days))))
    banner(f"完整业务周期：{data.n_days} 天逐日提升（含 "
           f"{int(data.day_is_weekend.sum())} 个周末）")
    log.info("逐日提升：均值 %+.2f%%，中位 %+.2f%%，范围 [%+.2f%%, %+.2f%%]，"
             "%d/%d 天与总方向一致", days.mean() * 100,
             float(np.median(days)) * 100, days.min() * 100, days.max() * 100,
             int(round(same_sign * days.size)), days.size)
    log.info("只跑一天的话，周末和工作日的差异会被当成处理效应 —— "
             "所以「跨完整业务周期」不是跑久一点，而是让周期内的"
             "每个阶段都有机会反驳这个结论")

    strata_rows = [{"分层": k, "相对提升": v.lift, "p": v.p_value}
                   for k, v in rep.strata.items()]
    if strata_rows:
        show_table(strata_rows, ["分层", "相对提升", "p"], "工作日 / 周末分层")
    for w in rep.warnings:
        log.warning("实验告警：%s", w)
    return {
        "exp_id": rep.exp_id,
        "healthy": rep.healthy,
        **rep.to_dict(),
        "true_lift": true_lift,
        "lift_ci": [lo, hi],
        "ci_covers_truth": bool(lo <= true_lift <= hi),
        "daily_same_sign_frac": same_sign,
        "daily_min": float(days.min()), "daily_max": float(days.max()),
        "daily_mean": float(days.mean()),
    }


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)

    aa = aa_report(cfg)
    log.info("⏳ AA 跑完，开始构造 AB 的实验数据（%d 用户 × %d 天）",
             AB_N_USERS, AB_DAYS)

    with Timer(f"07 AB 完整业务周期（{scale}）"):
        data, split, rep = RUN.simulate_experiment(
            cfg, n_users=AB_N_USERS, seed=AB_SEED, exp_id="pacing_v2",
            days=AB_DAYS, cvr_lift=AB_CVR_LIFT)

    power = power_section(cfg, data)
    ab = ab_report(cfg, data, split, rep, AB_CVR_LIFT)

    # ---------------- 覆盖率对照：处理只覆盖一部分流量 ----------------
    #
    # ⚠️ 这里**必须重复多次看均值**，不能只跑一次。第一版只跑了一次，
    #    拿到 +10.81%（真值 6.00%，比值 1.80），看着像估计量有偏 ——
    #    但实测 12 个种子：均值 0.0686、**标准差 0.0165**。也就是说
    #    单次估计的噪声（±1.7 个百分点）比它要检验的效应（6 个百分点）
    #    小不了多少，**单次结果根本没有分辨力**。
    #    这本身就是 deff=7.6 的直接后果，和 AA 那条一样：
    #    H0 附近的量只能断言**分布**，断言单次就是在断言噪声。
    with Timer(f"07 部分覆盖对照（{scale}）× {COVERAGE_REPS} 次"):
        cov_lifts = []
        for k in range(COVERAGE_REPS):
            _, _, rk = RUN.simulate_experiment(
                cfg, n_users=AB_N_USERS, seed=AB_SEED + 1 + k,
                exp_id=f"pacing_partial_{k}", days=AB_DAYS,
                cvr_lift=COVERAGE_LIFT, lift_fraction=COVERAGE)
            cov_lifts.append(float(rk.primary.lift))
    cov = np.asarray(cov_lifts)
    true_ate = COVERAGE_LIFT * COVERAGE
    cov_se = float(cov.std(ddof=1) / np.sqrt(cov.size))
    cov_ratio = float(cov.mean() / true_ate) if true_ate else 0.0

    banner(f"部分覆盖对照：真实效应 {COVERAGE_LIFT:.0%} 只对 "
           f"{COVERAGE:.0%} 的流量生效 → 真实 ATE = {true_ate:.1%}"
           f"（跑 {COVERAGE_REPS} 次）")
    log.info("单次估计的范围 [%+.2f%%, %+.2f%%]，标准差 %.4f —— "
             "**单次结果分辨不出 %.0f%% 这个量级的效应**，"
             "这正是 deff 的后果", cov.min() * 100, cov.max() * 100,
             float(cov.std(ddof=1)), true_ate * 100)
    log.info("%d 次均值 %+.2f%%（标准误 %.4f）= 真值的 %.2f 倍 —— "
             "「处理的真实强度」和「用户身上测到的效应」不是一回事，"
             "覆盖率和效应量必须分开报告",
             COVERAGE_REPS, cov.mean() * 100, cov_se, cov_ratio)

    # ---------------- 门槛 ----------------
    problems = []
    # 1. AA：分布断言（不是单次 p 值）
    if not aa["ks_pvalue"] > 0.05:
        problems.append(f"AA 的 p 值偏离 U(0,1)（KS p={aa['ks_pvalue']:.4f}）—— "
                        f"分流链路有系统性偏差")
    if not 0.01 <= aa["false_positive_rate"] <= 0.12:
        problems.append(f"AA 假阳性率 {aa['false_positive_rate'] * 100:.1f}% "
                        f"落在 [1%, 12%] 之外（理论 5%）")
    if aa["srm_min_pvalue"] <= 0.001:
        problems.append(f"AA 里出现了 SRM（最小 p={aa['srm_min_pvalue']:.4f}）—— "
                        f"分流本身不平衡")
    # 2. AB 的分流必须是健康的 —— 这是解读一切的前提
    if not ab["healthy"]:
        problems.append(f"AB 的 SRM p={ab['srm_p']:.4f} ≤ 0.001：分流坏了，"
                        f"后面所有数字都不值得看")
    # 3. 主指标：既要有统计显著性，也要**估得准**
    pri = rep.primary
    if not pri.p_value < 0.05:
        problems.append(f"主指标不显著（p={pri.p_value:.4g}）—— "
                        f"在 {AB_N_USERS} 用户下功效不足，"
                        f"应先按功效分析扩样本而不是改结论")
    if not abs(pri.lift - AB_CVR_LIFT) < 0.5 * AB_CVR_LIFT:
        problems.append(f"主指标估计 {pri.lift * 100:+.2f}% 偏离真实 "
                        f"{AB_CVR_LIFT * 100:+.2f}% 超过一半 —— "
                        f"检验的效应量估计有偏")
    if not ab["ci_covers_truth"]:
        problems.append(f"95% 置信区间 [{ab['lift_ci'][0] * 100:+.2f}%, "
                        f"{ab['lift_ci'][1] * 100:+.2f}%] 没有覆盖真实效应 "
                        f"{AB_CVR_LIFT * 100:+.2f}%")
    # 4. 逐日：效果不能只由某一天撑起来
    signed = np.sign(rep.daily_lift)
    if not float(np.mean(signed == np.sign(rep.daily_lift.mean()))) > 0.6:
        problems.append(f"只有 "
                        f"{float(np.mean(signed == np.sign(rep.daily_lift.mean()))) * 100:.0f}% "
                        f"的天数与总方向一致 —— 效果可能由个别日期撑起来")
    # 5. 分层：工作日和周末必须**都显著且同号**
    if rep.strata:
        est = [v for v in rep.strata.values()]
        if not all(v.p_value < 0.05 for v in est):
            problems.append("工作日/周末分层里至少有一层不显著")
        if len({bool(v.lift > 0) for v in est}) != 1:
            problems.append("工作日与周末的提升**方向相反** —— "
                            "这个效应不可复现到另一个时段")
    # 6. CUPED：协变量必须真有预测力，且必须来自实验前
    if rep.cuped:
        if not rep.cuped.get("variance_reduction", 0.0) > 0.10:
            problems.append(f"CUPED 只降了 "
                            f"{rep.cuped.get('variance_reduction', 0) * 100:.1f}% 方差 "
                            f"（目标 >10%）")
    else:
        problems.append("CUPED 没有生效（协变量与指标的相关性太低）")
    # 7. 功效：必须如实报告，不能拿朴素公式糊过去
    if not 0.5 <= power["plan"]["achieved_power"] <= 1.0:
        problems.append(f"实际功效只有 "
                        f"{power['plan']['achieved_power'] * 100:.0f}% —— "
                        f"低于 50% 的实验结论不可信，样本量要先补上")
    if power["deff"] <= 1.5:
        problems.append(f"deff = {power['deff']:.2f} ≤ 1.5，"
                        f"数据没有用户级聚集性 —— 仿真可能有问题")
    # 8. 部分覆盖：**多次的均值**要接近 lift × coverage
    #    判据用均值而不是单次 —— 理由见上面那段长注释，
    #    单次的标准差就有 %.4f 量级，拿单次卡等于在卡噪声。
    if not abs(cov_ratio - 1.0) < 0.25:
        problems.append(f"部分覆盖下 {COVERAGE_REPS} 次估计的 ATE 均值 "
                        f"{cov.mean() * 100:+.2f}% 是真值 {true_ate * 100:+.2f}% 的 "
                        f"{cov_ratio:.2f} 倍，偏离超过 25% —— 效应量估计有偏")

    metrics = {
        "scale": scale,
        "aa": aa,
        "power": {"setup": {k: v for k, v in power.items()
                            if k not in ("plan", "naive")},
                  "deff_corrected": power["plan"],
                  "naive": power["naive"]},
        "ab": ab,
        "partial_coverage": {
            "coverage": COVERAGE, "cvr_lift": COVERAGE_LIFT,
            "true_ate": true_ate, "replications": COVERAGE_REPS,
            "estimated_ate_mean": float(cov.mean()),
            "estimated_ate_se": cov_se,
            "estimated_ate_min": float(cov.min()),
            "estimated_ate_max": float(cov.max()),
            "estimated_ate_sd": float(cov.std(ddof=1)),
            "ratio": cov_ratio,
        },
        "problems": problems,
    }
    save_metrics("07_experiment", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：AA 假阳性率 %.1f%%（KS p=%.3f，分流无系统性偏差）；"
             "AB 主指标 %+.2f%%（真值 %+.2f%%，p=%.2g），"
             "deff=%.2f、实际功效 %.0f%%",
             aa["false_positive_rate"] * 100, aa["ks_pvalue"],
             pri.lift * 100, AB_CVR_LIFT * 100, pri.p_value,
             power["deff"], power["plan"]["achieved_power"] * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
