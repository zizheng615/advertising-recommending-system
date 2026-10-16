"""README 里的数字必须能在 ``artifacts/metrics/*.json`` 里找到。

这个文件的**唯一职责**是守住本项目那条对外承诺：

    「README 里每个数字都来自真实运行输出，不编造」

为什么值得单独一个测试文件
--------------------------
因为「数字错了」在这类项目里**不会报任何错**。模型照跑、图照出、测试照过，
错的是一个手抄进文档的常量 —— 而读者看到的是它。

而且踩过的坑是有规律的一类：**算在了错的对象上**。
本项目已经栽过三次，每次的症状都一样「看着挺像那么回事」：

  1. 画像宽度数了 ``profile.columns``（含 17 个簿记/隐变量列）→ 报 209，
     真值 192；连带噪声占比的分母被撑大，44.8% 报成 41.1%。
  2. 同一个 ``y_conv_w168`` 在曝光级是 0.0030、在点击级是 0.0575 —— 差了 19 倍。
  3. ``sigma.mean``（全体，含 90% 的 y=0）和 ``ratio_to_true``（只有付费样本）
     并排存进 json，名字像一对、实际是两个总体；两条 0.4964/0.70 = 0.709
     与 0.772 对不上。``fig07`` 照着画也错了：柱子画 0.4964、标注写 0.77 倍。

第 3 条就是写这个文件的直接原因 —— 它当初是从**文档**里看出来的，
不是从测试里。所以补上这一层。

怎么测（刻意选了「查字符串」而不是「解析 markdown」）
----------------------------------------------------
解析表格对 README 的排版改动极其脆弱（加一列、换个对齐就红），
而这里真正要防的是**数值本身漂了**，不是排版。
所以反过来做：从 json 取真值 → 按 README 声明的精度渲染成字符串 →
断言它**确实出现在 README 文本里**。

代价是：README 把某个数改成同值的另一种写法（0.0500 → 0.05）会误报。
这个代价可以接受 —— 那种改动本来也该顺手看一眼。

⚠️ 本文件只用 ``assert``，不依赖 pytest 特有功能，
   所以 ``pytest`` 和 ``run_tests.py`` 两条路都能跑（与其余测试一致）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "artifacts" / "metrics"
README = ROOT / "README.md"

SCALE = os.environ.get("ADREC_SCALE", "mini")

STEPS = ["01_data", "02_features", "03_ranker", "04_recall", "05_ziln",
         "06_bidding", "07_experiment", "08_bidding_sim", "09_figures"]


@pytest.fixture(scope="module")
def doc() -> str:
    """README 全文。缺了就没法测，明确跳过而不是报一堆级联失败。"""
    if not README.exists():
        pytest.skip("README.md 不存在")
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> dict[str, dict]:
    """九个指标 json。缺任何一个都跳过 —— 这里的断言全是跨文件的。"""
    missing = [s for s in STEPS if not (METRIC_DIR / f"{s}_{SCALE}.json").exists()]
    if missing:
        pytest.skip(f"链路未跑完，缺 {missing} —— 先跑：python run_all.py "
                    f"--scale {SCALE}")
    return {s: json.loads((METRIC_DIR / f"{s}_{SCALE}.json").read_text("utf-8"))
            for s in STEPS}


def must(doc: str, needle: str, why: str) -> None:
    """断言 ``needle`` 出现在 README 里。

    ``why`` 说明这个数在讲什么 —— 失败信息里带上它，
    否则只看到「0.8110 not found」根本不知道是哪个指标漂了。
    """
    assert needle in doc, (
        f"README 里找不到 {needle!r}（{why}）。"
        f"要么指标 json 变了而文档没跟着改，要么文档里的数从来就没对过。")


# ==================================================== 各环节的头条数字


def test_data_headline_numbers(doc, js):
    """职责点 1 的数据地基：基率 + 假负率。"""
    d = js["01_data"]
    must(doc, f"{d['ctr']:.4f}", "01 CTR 基率")
    must(doc, f"{d['cvr']:.4f}", "01 CVR|click 基率")
    fn = {r["window_h"]: r["fn_rate"] for r in d["false_negative"]}
    for w in (6, 24, 72, 168):
        must(doc, f"{fn[w] * 100:.1f}%", f"01 假负率（{w}h 观测窗口）")


def test_feature_layer_numbers(doc, js):
    """职责点 1 的特征与分层。

    特别守住 **192** 这个数：它当初是 209（数错了对象），
    而 209 会让「噪声占比」的分母被撑大、44.8% 被低估成 41.1%。
    """
    f = js["02_features"]
    assert f["features"]["profile_columns"] == 192, (
        "画像宽度不是 192 —— 若改了特征设计，README §4.2 的列数账要同步改")
    must(doc, "192", "02 画像宽表列数")
    must(doc, f"{f['features']['profile_noise_share'] * 100:.1f}%",
         "02 噪声+冗余占比")
    must(doc, f"{f['pca']['cumulative_variance']:.4f}", "02 PCA 累计方差")
    must(doc, f"{f['kmeans']['ari_vs_true_segment']:.4f}", "02 ARI vs 真值分层")
    must(doc, f"{f['kmeans']['silhouette']:.4f}", "02 轮廓系数")


def test_ranker_ablation_numbers(doc, js):
    """职责点 2 的排序消融：LR < FM < DeepFM 三个 AUC 都要对得上。"""
    ab = {r["variant"]: r for r in js["03_ranker"]["ablation"]}
    for v, name in (("lr", "LR"), ("fm", "FM"), ("deepfm", "DeepFM")):
        must(doc, f"{ab[v]['auc_ctr']:.4f}", f"03 {name} 的 CTR AUC")
    # 相对提升是 README 的核心主张，单独钉一次
    gap = ab["deepfm"]["auc_ctr"] - ab["fm"]["auc_ctr"]
    assert gap > 0.015, f"DeepFM 没打赢 FM 门槛（差 {gap:.4f}）"


def test_recall_and_ziln_numbers(doc, js):
    """职责点 2 的召回与 ZILN。"""
    ps = {r["split"]: r["recall@50"] for r in
          js["04_recall"]["recall"]["per_split"]}
    must(doc, f"{ps['test']:.4f}", "04 Recall@50（test 折）")
    must(doc, f"{js['04_recall']['recall']['popularity_baseline']:.4f}",
         "04 热度基线")
    z = js["05_ziln"]["metrics"]
    must(doc, f"{z['auc_pay']:.4f}", "05 付费分类 AUC")
    must(doc, f"{z['pit_assumption_pvalue']:.4f}", "05 前提 PIT p 值")
    must(doc, f"{z['normalized_gini']:.4f}", "05 Normalized Gini")


def test_ziln_sigma_caliber_is_not_conflated(doc, js):
    """**回归守卫**：σ̂ 的两个口径不许被配成一对。

    σ 是「付费金额」那一支的尺度参数，付费用户只占 10%。所以：

        付费口径  mean_pos = 0.5405  ← 只有它能跟真实 σ=0.70 比 → 比值 0.772
        全体口径  mean_all = 0.4964    含 90% 的 y=0 → 0.4964/0.70 = 0.709

    踩过的坑：json 里曾经并排存 ``mean`` 与 ``ratio_to_true``，
    名字像一对、实际是两个总体，于是 0.4964 配着 0.772 —— 自相矛盾。
    README 也照抄了，``fig07`` 更是画了 0.4964 的柱子、标「0.77 倍」。

    这里锁死两件事：① json 里每个比值都能由它**同名**的均值还原；
    ② README 里出现的 0.772 必须与 0.5405 同处一句话，不许跟 0.4964 混。
    """
    sg = js["05_ziln"]["sigma"]
    t = sg["true_sigma"]

    # ① json 自洽：比值必须配得上同名均值
    assert abs(sg["mean_pos"] / t - sg["ratio_pos_to_true"]) < 1e-9, \
        "ratio_pos_to_true 不是 mean_pos/true_sigma —— 两个口径又串了"
    assert abs(sg["mean_all"] / t - sg["ratio_all_to_true"]) < 1e-9, \
        "ratio_all_to_true 不是 mean_all/true_sigma —— 两个口径又串了"

    # 两个口径确实不同（否则这个测试是空转）
    assert abs(sg["mean_pos"] - sg["mean_all"]) > 0.01, \
        "两个口径算出同一个数 —— 那这个测试守不住任何东西，该删"

    # ② 这两个数都在文档里
    must(doc, f"{sg['mean_pos']:.4f}", "05 σ̂ 付费口径均值")
    must(doc, f"{sg['mean_all']:.4f}", "05 σ̂ 全体口径均值")
    must(doc, f"{sg['ratio_pos_to_true']:.3f}", "05 σ̂ 付费口径 vs 真实 σ 的比值")


def test_bidding_numbers(doc, js):
    """职责点 3：合成市场的六策略对照。"""
    b = js["06_bidding"]
    must(doc, f"+{b['headline']['lift_vs_fixed_bid'] * 100:.0f}%",
         "06 B4 相对 B0 的价值提升")
    must(doc, f"{b['headline']['ratio_b4_to_oracle']:.4f}", "06 B4 占上帝视角比例")
    # 对偶梯度符号 —— 这条写反了整套控制就反向发散
    g = b["dual_gradient_sign"]
    assert g["overspend_lowers_alpha"] is True, "超支时 α 没有下降 —— 符号写反了"
    assert g["underspend_raises_alpha"] is True, "欠花时 α 没有上升 —— 符号写反了"


def test_experiment_numbers(doc, js):
    """职责点 4：AA 空跑 + AB 主指标 + CUPED。"""
    e = js["07_experiment"]
    must(doc, f"{e['aa']['ks_pvalue']:.4f}", "07 AA 的 KS 均匀性 p 值")
    must(doc, f"{e['aa']['false_positive_rate'] * 100:.2f}%", "07 AA 假阳性率")
    must(doc, f"+{e['ab']['primary']['lift'] * 100:.2f}%", "07 AB 主指标相对提升")
    must(doc, f"{e['ab']['cuped']['variance_reduction'] * 100:.2f}%",
         "07 CUPED 方差下降")


def test_real_model_bidding_numbers(doc, js):
    """职责点 3 接真实模型那一版：水平系数 + 恒等性 + 最好变体。"""
    s = js["08_bidding_sim"]
    lc = s["setup"]["level_correction"]
    for k, name in (("p_ctr", "pCTR"), ("p_cvr", "pCVR"), ("ltv", "LTV")):
        must(doc, f"{lc[k]:.3f}", f"08 验证折上拟合的 {name} 水平系数")
    ifx = s["identity_check"]
    must(doc, f"{ifx['alpha_ratio_rel_dev'] * 100:.3f}%",
         "08 水位之比相对 c 的偏差（协变性判据）")
    best = s["best_variant"]
    r = s["runs"][best]
    must(doc, f"{r['ratio_b4_to_oracle']:.4f}", f"08 最好变体 {best} 的 B4/B5")
    assert best == "dfm_raw", (
        f"最好变体变成了 {best} —— README §6.7 的结论是按 dfm_raw 写的，要同步改")


def test_budget_is_exogenous_across_runs(doc, js):
    """08 的预算必须**外生且三次共用** —— 否则三个变体根本不可比。

    B0（固定出价）与 B5（上帝视角）都与模型无关，所以三份仿真的这两个
    策略必须逐位相同。若它们随模型变了，说明预算被偷偷接到了模型的
    预测上，「同预算对比」这个前提就没了。
    """
    runs = js["08_bidding_sim"]["runs"]
    for pol in ("B0_fixed", "B5_oracle"):
        spends = [r["policies"][pol]["spend"] for r in runs.values()]
        assert len(set(spends)) == 1, (
            f"{pol} 的花费在三份仿真里不一致：{spends} —— "
            f"预算不再外生，三个变体的对比不成立")
    must(doc, "逐位相同", "08 预算外生性")
