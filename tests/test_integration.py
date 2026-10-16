"""集成测试 —— 验证**落盘的产物**能互相咬合，而不是验证某个模块的函数。

和其余 ``test_*.py`` 的区别（这是这个文件存在的理由）
--------------------------------------------------
别的测试都在**内存里**调 ``src/`` 的函数：给一份数据、断言一个返回值。
它们证明「每一环单独是对的」。但整条链路还有一类错误是它们**看不见**的：

* 03 用种子 A 训了模型，08 读到的却是种子 B 落的那份 —— 每一环都自洽，
  拼起来是错的；
* 08 的预算这一版改成外生了，但只改了两次仿真里的一次 ——
  两次仿真的差异里混进了「预算不同」；
* 某个脚本的 ``problems`` 里有条目，但非零退出码被 ``| tee`` 吃掉了。

**这些错误全都发生在接口上，而不是环节内部。** 所以这里的断言一律是
「文件 A 里的数 == 文件 B 里的数」或者「这个文件必须自洽」。

跑之前要先跑 ``run_all.py``
---------------------------
本文件读 ``artifacts/metrics/*.json``，那是 ``run_all.py`` 的产物。
没有产物时**跳过**而不是失败 —— 否则在一个干净的 checkout 上
``pytest tests/`` 会红一片，而真实原因只是「还没跑链路」，
和代码对不对无关。跳过信息里会写明该跑什么。
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "artifacts" / "metrics"
FIG_DIR = ROOT / "artifacts" / "figures"

# 默认跟其余测试一样用 mini 档。要验 small/full 就设环境变量，
# 但注意 README 与交付数字用的是 small。
SCALE = os.environ.get("ADREC_SCALE", "mini")

# 链路里的九步，一步一个 json —— 少一个就说明链路没跑完
STEPS = ["01_data", "02_features", "03_ranker", "04_recall", "05_ziln",
         "06_bidding", "07_experiment", "08_bidding_sim", "09_figures"]

EXPECTED_FIGURES = 9


def _load(name: str) -> dict:
    p = METRIC_DIR / f"{name}_{SCALE}.json"
    if not p.exists():
        pytest.skip(f"缺少 {p.name} —— 先跑：python run_all.py --scale {SCALE}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics() -> dict[str, dict]:
    """一次把九个 json 全读进来。

    任何一个缺失就**整体跳过**：这个文件里的断言几乎都是跨文件的，
    缺一个就没法做，与其报一堆级联错误不如明确说「链路没跑完」。
    """
    missing = [s for s in STEPS
               if not (METRIC_DIR / f"{s}_{SCALE}.json").exists()]
    if missing:
        pytest.skip(f"链路未跑完，缺 {missing} —— 先跑：python run_all.py "
                    f"--scale {SCALE}")
    return {s: _load(s) for s in STEPS}


# ============================================================ 1 完整性


def test_all_steps_produced_metrics(metrics):
    """九步都要有指标落盘，且都要声明自己是哪一档规模。

    规模这一栏看着没用，但它是防「混档」的唯一手段：跑到一半改成
    ``--scale full``，产物目录里就会同时存在两档的文件，
    而下游按名字取，取到哪一档全看文件名后缀 —— 不报错，数字全乱。
    """
    assert set(metrics) == set(STEPS)
    for name, m in metrics.items():
        assert m.get("scale") == SCALE, (
            f"{name} 的 scale 是 {m.get('scale')!r}，期望 {SCALE!r} —— "
            f"产物混档了，重跑 run_all.py")


def test_no_step_reported_problems(metrics):
    """每个脚本的 ``problems`` 必须为空。

    这是**最重要的一条**：每个脚本跑完都会自检并把不通过的项写进
    ``problems``，然后用**非零退出码**结束。但退出码会被 ``| tail``、
    被 CI 的 continue-on-error、被「先看看输出」吃掉 ——
    而 ``problems`` 写在文件里，只要读了就跑不掉。

    因此这里把「体检结论」从**进程的退出码**搬到**数据**上。
    """
    # 先卡「有没有这一栏」：某个脚本忘了写 problems，这里会 KeyError，
    # 而 KeyError 的信息量是零 —— 它看起来像测试写错了，其实是脚本漏了。
    no_key = [name for name, m in metrics.items() if "problems" not in m]
    assert not no_key, f"这些环节的指标 json 缺少 problems 栏：{no_key}"

    bad = {name: m["problems"] for name, m in metrics.items() if m["problems"]}
    assert not bad, f"以下环节体检不通过：{json.dumps(bad, ensure_ascii=False, indent=2)}"


def test_nine_figures_exist_and_are_real_pngs(metrics):
    """九张图都要在，而且是**真 PNG**，不是 0 字节的占位。

    ⚠️ 校验魔数而不是只看文件存在：``savefig`` 在字体缺字时**不会失败**，
    它会把方框画上去；而在磁盘满的时候会留下一个截断的文件。
    本机 C 盘只剩 0.6GB，这个检查不是假想出来的。
    """
    d = metrics["09_figures"]
    assert d["n_figures"] == EXPECTED_FIGURES, f"只出了 {d['n_figures']} 张图"
    assert not d["skipped"], f"有图被跳过：{d['skipped']}"

    pngs = sorted(FIG_DIR.glob("*.png"))
    assert len(pngs) == EXPECTED_FIGURES, (
        f"{FIG_DIR} 里有 {len(pngs)} 张图，期望 {EXPECTED_FIGURES} 张")
    for p in pngs:
        raw = p.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{p.name} 不是 PNG"
        assert len(raw) > 20_000, (
            f"{p.name} 只有 {len(raw)} 字节 —— 多半是空图或写坏了")
        # IHDR 的宽高：解开确认不是 0×0
        w, h = struct.unpack(">II", raw[16:24])
        assert w > 400 and h > 200, f"{p.name} 尺寸异常 {w}×{h}"


# ============================================================ 2 跨脚本咬合


def test_split_sizes_agree_across_scripts(metrics):
    """01 切出来的折大小，必须和 08 记录的完全一致。

    08 的 ``setup.n_valid/n_test`` 是它**实际拿去推理**的行数。
    这两个数对不上，说明 08 读的是一份旧的 split.pkl ——
    此时它的校准系数、出价结果全都建立在另一个时间切分上，
    而它自己**毫不知情**（因为文件读得到，形状也对）。
    """
    d1, d8 = metrics["01_data"], metrics["08_bidding_sim"]
    sizes = {s["split"]: s["n"] for s in d1["splits"]}
    assert sizes["valid"] == d8["setup"]["n_valid"]
    assert sizes["test"] == d8["setup"]["n_test"]


def test_prior_correction_beta_is_the_same_object(metrics):
    """03 拟合的 β，必须**逐位**等于 08 拿来做先验校正的那个 β。

    这是全链路里最危险的一个耦合：β 是负样本保留率，08 用它做
    ``logit += ln β``。如果 08 读到的 β 来自另一次运行（改了种子、
    改了采样比），pCTR 会被校正到一个**错的先验**上 ——
    而 eCPM 的三个因子都会被它带偏，出价结果照样"跑得出来"。

    所以这里断言的是精确相等，不是近似。
    """
    b3 = metrics["03_ranker"]["downsample"]["beta"]
    b8 = metrics["08_bidding_sim"]["setup"]["beta"]
    assert b3 == b8, f"03 的 β={b3!r} 与 08 的 β={b8!r} 不一致"


def test_level_factor_matches_its_own_factors(metrics):
    """08 里的整体缩放倍数 c，必须等于它自己那三个水平系数的乘积。

    校验的是 ``identity_check.level_factor_c`` 这个**派生量**没有和
    它的来源脱节（比如只改了某几个头、忘了重算 c）。恒等性那套推导
    全部建立在 c 上，c 错了后面的判据全是错的。
    """
    s = metrics["08_bidding_sim"]["setup"]["level_correction"]
    c_expected = s["p_ctr"] * s["p_cvr"] * s["ltv"]
    c_reported = metrics["08_bidding_sim"]["identity_check"]["level_factor_c"]
    assert c_reported == pytest.approx(c_expected, rel=1e-12)

    # ⚠️ 注意 p_cvr_dfm **不在** c 里面：dfm 变体的 q 直接就是它，
    #    不经过 pCTR × pCVR 那条乘积路径（见 08 的 build_q）。
    assert "p_cvr_dfm" in s and len(s) == 4


def test_budget_is_shared_across_the_three_simulations(metrics):
    """三次仿真的预算必须**同一个数**，否则比较不公平。

    这是真正踩过的坑：最初预算由 ``u_pred`` 反解，于是校准改变了模型
    水平 → 改变了预测花费 → 改变了预算。「校准让价值涨了 50%」里
    混着「预算也涨了 50%」，两件事分不开。

    改成由 ``true_value_cpm``（外生）反解之后，三次仿真的预算逐位相同，
    连带 B0（固定出价，不依赖任何模型）和 B5（上帝视角）的期望价值
    也应当**逐位相同** —— 它们只依赖流量、市场和预算。
    用 B0/B5 是否相同来验证「预算确实外生」，比直接比预算数字更强，
    因为它顺带证明了流量的抽样序列也是同一份（配对比较）。
    """
    runs = metrics["08_bidding_sim"]["runs"]
    assert len(runs) == 3
    b0 = {t: r["policies"]["B0_fixed"]["expected_value"] for t, r in runs.items()}
    b5 = {t: r["policies"]["B5_oracle"]["expected_value"] for t, r in runs.items()}
    assert len(set(b0.values())) == 1, f"B0 在三次仿真里不同：{b0}"
    assert len(set(b5.values())) == 1, f"B5 在三次仿真里不同：{b5}"


def test_ranking_diagnostic_is_internally_consistent(metrics):
    """08 的排序诊断要自洽，并且要和它自己的选择**同向**。

    这里钉住的是那份诊断的**结论**而不是数字：最好变体的
    弹性 γ 和紧预算截获率都应当高于另一个变体。

    ⚠️ 唯独全量 Spearman **不**参与这个断言 —— 它跟另外两个反向，
       这正是 08 要讲的那件事（按比例出价对单调变换免疫，而 Spearman
       恰好对单调变换完全免疫）。如果哪天它变得同向，说明有人把
       出价改成了按名次，那是个**设计变更**，应当来看这里而不是
       悄悄让它过去。
    """
    runs = metrics["08_bidding_sim"]["runs"]
    best = metrics["08_bidding_sim"]["best_variant"]
    others = [t for t in runs if t != best]
    assert best in runs

    for t in others:
        assert runs[best]["loglog_elasticity"] > runs[t]["loglog_elasticity"], (
            f"{best} 的 γ 并不比 {t} 高，但 B4/B5 更高 —— 诊断与结果脱节")
        assert (runs[best]["capture_at_buy_frac"]
                > runs[t]["capture_at_buy_frac"])
        assert (runs[best]["loglog_elasticity"]
                <= 1.0 + 1e-9), "γ > 1 说明预测比真值还发散，超出预期"


def test_buy_frac_is_a_sane_fraction(metrics):
    """买入比例必须落在合理区间 —— 它是截获率那条指标的分母。

    ``buy_frac`` 取的是该变体 B4 的实测 ``win_rate``。它要是 0，
    ``rank_diag`` 里 ``max(1, ...)`` 会兜住不报错，但截获率会退化成
    「只看一行」，指标失去意义且**完全不会报错**。
    """
    for t, r in metrics["08_bidding_sim"]["runs"].items():
        assert 0.001 < r["buy_frac"] < 0.5, f"{t} 的 buy_frac={r['buy_frac']}"
        assert 0.0 < r["capture_at_buy_frac"] <= 1.0


# ============================================================ 3 统计结论


def test_aa_dry_run_looks_uniform(metrics):
    """AA 空跑的 p 值要像 U(0,1)，假阳性率落在理论值附近。

    **不能**断言「某一次 AA 的 p > 0.05」—— AA 两组策略完全相同，
    H0 为真，p 值本来就服从 U(0,1)，单次有 5% 的概率落在 0.05 以下。
    那不是 bug，是定义。所以断言的是**分布**（KS 均匀性）
    加**假阳性率区间**（理论 0.05，给到 [0.01, 0.12] 以容纳
    有限次重复的抽样波动）。
    """
    aa = metrics["07_experiment"]["aa"]
    assert aa["ks_pvalue"] > 0.05, (
        f"AA 的 p 值不像均匀分布（KS p={aa['ks_pvalue']:.4f}）—— "
        f"分流可能有内生性偏置")
    assert 0.01 <= aa["false_positive_rate"] <= 0.12, (
        f"假阳性率 {aa['false_positive_rate']:.1%} 偏离理论值 5% 太多")
    assert aa["replications"] >= 30, "重复次数太少，KS 检验本身没有功效"


def test_ab_experiment_passed_its_srm_check(metrics):
    """A/B 的 SRM（样本比例失衡）卡方要过关。

    SRM 是最灵敏的 bug 探测器：分流写错、日志丢一边、埋点漏一个桶，
    都会先在这里显形，而主指标可能仍然"显著"。
    """
    ab = metrics["07_experiment"]["ab"]
    assert ab["srm_p"] > 0.001, f"SRM 卡方 p={ab['srm_p']:.4g}，分流可疑"
    assert ab["healthy"], "实验自检标记为不健康"


def test_ziln_headline_metrics_are_reported(metrics):
    """ZILN 的四个对外数字都要在，且方向正确。

    特别是 ``aggregate_bias``：漏掉 ``exp(σ²/2)`` 的那一版**必须更差**。
    这一项错了 eCPM 会系统性低估，而 AUC / Gini **完全看不出来** ——
    它们只看次序不看水平。所以必须显式断言方向。
    """
    m = metrics["05_ziln"]["metrics"]
    assert m["aggregate_bias_without_sigma2"] > m["aggregate_bias"], (
        "漏掉 exp(σ²/2) 的版本反而更好 —— 修正项的符号可能写反了")
    assert m["auc_pay"] < m["auc_oracle"], "付费 AUC 不该超过理论上限"
    assert m["auc_pay"] > 0.55
    assert m["normalized_gini"] > 0.30


def test_ranker_beats_its_baselines(metrics):
    """职责点 2 的核心主张：DeepFM 要显著打赢 LR 与 FM。

    生成器里埋了三阶/四阶交叉项和 XOR 奇偶项（FM 的 rank-1 分解
    表达不了），所以这个不等式是**可验证的命题**，不是「跑通即可」。
    """
    ab = {r["variant"]: r["auc_ctr"] for r in metrics["03_ranker"]["ablation"]}
    assert ab["deepfm"] - ab["lr"] > 0.03, f"DeepFM 相对 LR 只有 {ab}"
    assert ab["deepfm"] - ab["fm"] > 0.015, f"DeepFM 相对 FM 只有 {ab}"


def test_bidding_beats_manual_fixed_bid(metrics):
    """职责点 3 的核心主张：智能出价要打赢人工固定出价。

    合成市场（06）是干净对照；接真实模型（08）是带模型误差的版本。
    两个都要成立，而且 08 里**每个变体**都要成立 ——
    只挑最好看的那个报出来不算数。
    """
    h = metrics["06_bidding"]["headline"]
    assert h["lift_vs_fixed_bid"] > 0.20, f"合成市场只提升 {h}"
    for tag, r in metrics["08_bidding_sim"]["runs"].items():
        assert r["lift_vs_fixed_bid"] > 0.20, f"[{tag}] 只提升 {r['lift_vs_fixed_bid']:.1%}"


def test_overspend_red_line_holds_everywhere(metrics):
    """超支红线：所有要上线的策略都必须守在 0.5% 以内。

    ⚠️ B1 贪心单独放宽到 1.5%，这不是放水：它是**故意不带 pacing 的
    对照组**，存在意义就是把预算打光。它的超支来自「闸门按期望扣费
    封顶、单 slot 实际扣费有尾巴」，是它自己设计的性质。
    但它仍然必须**低于** B4 —— 那才是它要证明的东西。
    """
    LIMIT = 0.005
    GREEDY_LIMIT = 0.015
    for tag, r in metrics["08_bidding_sim"]["runs"].items():
        for pol, v in r["policies"].items():
            lim = GREEDY_LIMIT if pol == "B1_greedy" else LIMIT
            assert v["overspend_frac"] <= lim, (
                f"[{tag}] {pol} 超支 {v['overspend_frac']:.4%} > {lim:.1%}")
        res = r["policies"]
        assert res["B1_greedy"]["expected_value"] < res["B4_lambda_pid"]["expected_value"]


def test_oracle_is_an_upper_bound(metrics):
    """上帝视角必须真的在上界：B4 不能达到或超过 B5。

    如果 B4 ≥ B5，最有解释是**真值泄漏进了策略** ——
    比如把 ``true_value_cpm`` 传给了出价那一侧。
    """
    for tag, r in metrics["08_bidding_sim"]["runs"].items():
        b4 = r["policies"]["B4_lambda_pid"]["expected_value"]
        b5 = r["policies"]["B5_oracle"]["expected_value"]
        assert b4 < b5, f"[{tag}] B4={b4:.1f} 达到或超过 B5={b5:.1f} —— 上界失效"


# ============================================================ 4 口径


def test_prior_correction_only_applied_to_pctr(metrics):
    """先验校正只加在 pCTR 上，**绝不能**加到 pCVR 上。

    下采样只丢曝光负样本、保留全部点击，所以点击率被抬高了，
    而 ``P(转化|点击)`` 一点没动 —— 它的条件事件没被采样动过。
    顺手对 pCVR 也来一次 ``logit += ln β``，误差会从 −46.9% 恶化到 −76.2%。

    这个断言守的是**方向**：加上去只会更差。
    """
    a = metrics["08_bidding_sim"]["prior_correction_ablation"]
    tgt = a["target"]
    raw_err = abs(a["p_cvr_raw_mean_on_clicks"] / tgt - 1)
    wrong_err = abs(a["p_cvr_with_prior_correction"] / tgt - 1)
    assert wrong_err > raw_err, (
        "对 pCVR 也做先验校正反而更准了 —— 口径推导有问题，"
        "需要重新检查下采样到底动了什么")


def test_head_tables_use_the_right_conditioning_event(metrics):
    """四个头的「口径」必须各自正确，且校准要能把误差减半。

    ⚠️ 同一个 ``y_conv_w168``，在**全曝光**上取均值是 0.0030，
    在**点击**上取均值是 0.0575 —— 差 19 倍。DFM 估的是曝光级概率、
    DeepFM 的 CVR 头估的是点击条件概率，两个头长得一样但口径不同。
    拿错口径去量，会得出「DFM 偏了 5 倍」这种数字。

    这里断言：dfm 头的实际均值确实是曝光级那一档，cvr 头是点击那一档。
    """
    m = metrics["08_bidding_sim"]
    heads = {h["头"]: h for h in m["heads_uncalibrated"]}
    assert set(heads) == {"pCTR", "pCVR", "P(转化)", "E[金额]"}
    # DFM 的参照是曝光级（≈0.003），CVR 头的参照是点击级（≈0.058）
    assert heads["P(转化)"]["实际均值"] < heads["pCVR"]["实际均值"] / 5, (
        "DFM 头与 CVR 头的参照值太接近 —— 口径多半被写成了同一个")

    for raw, cal in zip(m["heads_uncalibrated"], m["heads_calibrated"]):
        assert abs(cal["相对误差"]) < 0.5 * abs(raw["相对误差"]), (
            f"{cal['头']} 校准后误差没减半："
            f"{raw['相对误差']:.1%} → {cal['相对误差']:.1%}")


def test_identity_verdict_matches_the_branches(metrics):
    """恒等性的结论必须和实测分支一致，不能出现自相矛盾的记录。

    恒等性分两层：``solve_alpha`` 的协变性（代数性质，**永远**成立）
    和价值级恒等（会被硬闸门打破）。把两层混成一句话，就会在
    「闸门触发」那一支里报出「恒等性已证」这种字面为假的结论 ——
    这个坑真踩过，所以在这里钉死。
    """
    ic = metrics["08_bidding_sim"]["identity_check"]
    # 判据①：α 之比必须约等于整体缩放倍数 c
    assert ic["alpha_ratio_rel_dev"] < 0.02, (
        f"α_raw/α_lvl={ic['alpha_ratio_raw_over_level']:.4f} 与 c="
        f"{ic['level_factor_c']:.4f} 差 {ic['alpha_ratio_rel_dev']:.2%}")
    assert not ic["saturated_calibrated"], "校准后的水位贴到了带边界"

    if ic["gate_fired"]:
        assert ic["gate_hits_raw"] > 0 or ic["gate_hits_calibrated"] > 0
        assert "闸门" in ic["verdict"]
        # 预注册的方向：闸门惩罚虚高出价，所以校准侧应当**不劣**
        assert ic["b4_value_calibrated"] >= ic["b4_value_raw"], (
            "闸门触发时校准侧更差 —— 与「闸门压虚高出价」的机制相反")
        assert ic["rel_gap"] > 0
    else:
        assert "恒等" in ic["verdict"]
        assert ic["rel_gap"] < 5e-3, (
            f"闸门没触发，价值却差了 {ic['rel_gap']:.4%} —— 与恒等性矛盾")
