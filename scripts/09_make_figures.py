"""09 出图 —— 把前面每一步落盘的指标画成九张图。

**为什么出图脚本要单独一步、而且要读 json 而不是重算**

图是给别人看的，指标 json 是给机器看的，两者同源。这里**只读**
``artifacts/metrics/*.json``，一行都不重算 —— 因为重算就意味着图上的
数字和 README 里的数字可能来自两次不同的运行，而**两次运行不一样**是
这个项目里最不该发生的错误（换了 seed、换了折、模型漂移都可能）。
读 json 则天然保证「图上 == 表里 == README」。

代价是**必须先把前面的脚本跑完**。缺哪一个 json 就跳过哪一张图并
在末尾点名，而不是画一张多半是空的图出来 —— 空图比没图更误导。

九张图对应职责链路的九个环节
--------------------------
1. 数据地基：假负率随窗口 + 转化延迟分布
2. 特征与分层：PCA 累计方差 + 轮廓系数选 K
3. 分层画像：各簇规模 / 纯度 / 与人工注入分层的一致性
4. 排序模型消融：LR < FM < DeepFM
5. 不平衡处理：1:5 下采样 + 先验校正
6. 延迟反馈去偏：vanilla / elapsed / DFM 三档对照
7. ZILN：σ 校准 + 两个 PIT + 聚合偏差对噪声地板
8. 智能出价：六策略对照（合成市场）+ 接真实模型后的排序诊断
9. A/B 实验：AA 的 p 值均匀性 + 方差功效 + CUPED

⚠️ matplotlib 后端在本机是 ``tkagg``，存图前必须强制 ``Agg`` ——
   否则在无显示环境（CI / 后台）里会在 ``savefig`` 处直接崩。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 必须在 pyplot 之前

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

from scripts._common import (FIG_DIR, banner, cfg_for,  # noqa: E402
                             load_metrics, log, parse_scale, show_table)

# --------------------------------------------------------------------------
# 中文字体 —— 不设的话所有中文都是方框，而且**不报错**
# --------------------------------------------------------------------------
# font.sans-serif 是**有序回退列表**，matplotlib 逐项找第一个装了的。
# 本机实测有 Microsoft YaHei / SimHei，所以中文能正常显示；末尾留 DejaVu
# 是为了万一在没装中文字体的机器上跑，至少英文和数字还能出来（中文变方框
# 但脚本不崩）。`axes.unicode_minus=False` 解决负号被画成方框的经典问题 ——
# 那个方框来自 DejaVu 有 U+2212 而中文字体没有。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DengXian", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

C = {"b": "#2b6cb0", "o": "#dd6b20", "g": "#2f855a", "r": "#c53030",
     "p": "#6b46c1", "gray": "#718096", "t": "#285e61"}


def _safe(name: str, cfg) -> dict | None:
    """读一份指标 json；缺了就返回 None 让调用方跳过这张图。"""
    try:
        return load_metrics(name, cfg)
    except SystemExit:
        return None


def _save(fig, name: str, suptitle: str | None = None,
          top: float = 0.84) -> str:
    """统一收尾：**给 suptitle 显式留白**，否则它会压住子图标题。

    踩过的坑：``fig.tight_layout()`` 的 ``rect`` 只保证**轴**之间不重叠，
    管不到**标题**。本机 figsize 高度只有 4 英寸，子图顶端约在 y=0.88，
    一个两行的子图标题能一路顶到 y=0.99 —— 正好撞上默认 ``y=0.98`` 的
    suptitle，屏幕上就是两段文字叠印在一起（fig08 / fig09 实测踩到）。
    所以这里自己留白，并把 suptitle 放回图内 y=0.99。
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if suptitle:
        fig.tight_layout(rect=[0, 0, 1, top])
        fig.suptitle(suptitle, fontsize=12.5, y=0.99)
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p.name


def _bar(ax, labels, values, colors, title, ylabel, fmt="{:.3f}",
         threshold=None, thr_label=""):
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    if threshold is not None:
        ax.axhline(threshold, color=C["r"], ls="--", lw=1.4,
                   label=thr_label or f"门槛 {threshold}")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.margins(y=0.18)


# --------------------------------------------------------------------------
# 1 数据地基
# --------------------------------------------------------------------------


def fig01(d, cfg) -> str | None:
    if d is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    fn = d["false_negative"]
    # ⚠️ 这一列**不全是小时数**：最后一项的 window_h 是字符串 "online"
    #    （代表线上真正拿到的那份标签：24h 窗口 + 数据末尾截断）。
    #    直接丢进对数 x 轴会在 set_xticks 里抛
    #    「'<' not supported between 'str' and 'int'」—— 因为 min() 要排序。
    #    它不是某一档窗口，而是**所有档位之外的现实**，所以单独画成横线。
    win = [r for r in fn if isinstance(r["window_h"], (int, float))]
    onl = [r for r in fn if not isinstance(r["window_h"], (int, float))]
    w = [r["window_h"] for r in win]
    ax[0].plot(w, [r["fn_rate"] for r in win], "o-", color=C["b"], lw=2,
               label="给定回补窗口下的假负率")
    for x, r in zip(w, win):
        ax[0].annotate(f"{r['fn_rate']:.1%}", (x, r["fn_rate"]),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=8.5)
    ax[0].set_xscale("log")
    ax[0].set_xticks(w)
    ax[0].set_xticklabels([str(v) for v in w])
    ax[0].set_xlabel("归因窗口（小时，对数轴）")
    ax[0].set_ylabel("假负率")
    if onl:
        ax[0].axhline(onl[0]["fn_rate"], color=C["r"], ls="--", lw=1.6,
                      label=f"线上实际标签（{onl[0]['fn_rate']:.1%}）")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("标签的假负率随回补窗口衰减 —— 延迟反馈的根源", fontsize=11)
    ax[0].set_ylim(0, max(r["fn_rate"] for r in fn) * 1.35)

    dl = d["delay"]
    keys = ["median_h", "p90_h", "p99_h"]
    vals = [dl[k] for k in keys]
    ax[1].bar(["中位数", "p90", "p99"], vals,
              color=[C["g"], C["o"], C["r"]], width=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.1f}h", ha="center", va="bottom", fontsize=9)
    ax[1].axvline(-0.5, alpha=0)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("小时（对数轴）")
    ax[1].set_title(f"转化延迟分位数（n={dl['n_conversions']}，"
                    f"{dl['frac_within_1h']:.0%} 在 1h 内，"
                    f"{dl['frac_beyond_24h']:.0%} 超过 24h）", fontsize=11)
    ax[1].margins(y=0.18)
    return _save(fig, "fig01_data_false_negative",
                 f"① 模拟数据地基（{cfg.scale}：{d['n_users']} 用户 / "
                 f"{d['n_items']} 素材 / {d['n_impressions']:,} 曝光）",
                 top=0.86)


# --------------------------------------------------------------------------
# 2 特征与分层
# --------------------------------------------------------------------------


def fig02(d, cfg) -> str | None:
    if d is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    n = d["pca"]["n_components"]
    cum = d["pca"]["cumulative_variance"]
    # 只有累计值，画成矩形阶梯 —— 不给假曲线，精确反映「只有这一个数」
    ax[0].bar([n], [cum], color=C["b"], width=6)
    ax[0].axhline(0.90, color=C["r"], ls="--", lw=1.4, label="门槛 0.90")
    ax[0].text(n, cum, f"保留 {n} 维\n累计方差 {cum:.3f}",
               ha="center", va="bottom", fontsize=9.5)
    ax[0].set_xticks([n])
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("累计解释方差")
    ax[0].set_title(f"PCA：StandardScaler 之后取 {n} 维\n"
                    f"（正交性误差 {d['pca']['orthogonality_error']:.1e}）",
                    fontsize=11)
    ax[0].legend(fontsize=8, loc="lower right")

    ks = sorted(int(k) for k in d["kmeans"]["silhouette_scores"])
    sil = [d["kmeans"]["silhouette_scores"][str(k)] for k in ks]
    best = d["kmeans"]["k"]
    ax[1].plot(ks, sil, "o-", color=C["p"], lw=2)
    ax[1].scatter([best], [d["kmeans"]["silhouette_scores"][str(best)]],
                  s=150, facecolor="none", edgecolor=C["r"], lw=2.2, zorder=5,
                  label=f"选中 K={best}")
    ax[1].axhline(0.20, color=C["o"], ls="--", lw=1.4, label="门槛 0.20")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("轮廓系数")
    ax[1].set_title("KMeans++ 在 PCA 空间选 K\n（与生成器的真值分层数一致）",
                    fontsize=11)
    ax[1].legend(fontsize=8)
    return _save(fig, "fig02_features_pca_kmeans",
                 "② 特征工程与人群分层",
                 top=0.79)


# --------------------------------------------------------------------------
# 3 分层画像
# --------------------------------------------------------------------------


def fig03(d, cfg) -> str | None:
    if d is None:
        return None
    km = d["kmeans"]
    cl = km["clusters"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    lab = [f"簇{c['cluster']}\n→真值层{c['top_true_segment']}" for c in cl]
    share = [c["share"] for c in cl]
    ax[0].bar(lab, share, color=[C["b"], C["o"], C["g"], C["p"], C["t"]][:len(cl)],
              width=0.6)
    for i, v in enumerate(share):
        ax[0].text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylabel("用户占比")
    ax[0].set_title(f"各簇规模（最小簇 {km['min_cluster_frac']:.1%}，"
                    f"无退化空簇）", fontsize=11)
    ax[0].margins(y=0.18)

    pur = [c["purity"] for c in cl]
    ax[1].bar(lab, pur, color=C["g"], width=0.6)
    for i, v in enumerate(pur):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylim(0, 1.12)
    ax[1].set_ylabel("簇纯度")
    ax[1].set_title(f"能恢复生成器注入的分层：ARI = {km['ari_vs_true_segment']:.4f}\n"
                    f"（加权纯度 {km['weighted_purity']:.4f}，K={km['k']} "
                    f"vs 真值层数 {km['true_segments']}）", fontsize=11)
    return _save(fig, "fig03_segments",
                 "③ 分层质量 —— 无监督分出来的簇能否对上真值分层",
                 top=0.8)


# --------------------------------------------------------------------------
# 4 排序模型消融
# --------------------------------------------------------------------------


def fig04(d, cfg) -> str | None:
    if d is None:
        return None
    ab = d["ablation"]
    names = [r["variant"].upper() for r in ab]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for j, (key, title, thr) in enumerate([
            ("auc_ctr", "CTR 的 AUC", None),
            ("auc_cvr", "CVR 的 AUC", None),
            ("auc_ctcvr", "CTCVR（曝光→转化）的 AUC", None)]):
        _bar(ax[j], names, [r[key] for r in ab],
             [C["gray"], C["o"], C["b"]][:len(ab)],
             title, "AUC", "{:.4f}", thr)
    gap = ab[-1]["auc_ctr"] - ab[0]["auc_ctr"]
    return _save(fig, "fig04_ranker_ablation",
                 f"④ 排序模型消融：DeepFM 相对 LR 的 CTR AUC 提升 {gap:+.4f}"
                 f"（生成器埋了三阶/四阶交叉项，FM 的 rank-1 分解表达不了）",
                 top=0.82)


# --------------------------------------------------------------------------
# 5 不平衡处理
# --------------------------------------------------------------------------


def fig05(d, cfg) -> str | None:
    if d is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ds = d["downsample"]
    ax[0].bar(["下采样前", "下采样后"], [ds["pos_rate_before"], ds["pos_rate_after"]],
              color=[C["gray"], C["b"]], width=0.5)
    for i, v in enumerate([ds["pos_rate_before"], ds["pos_rate_after"]]):
        ax[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
    ax[0].set_ylabel("正样本率")
    ax[0].set_title(f"1:5 负下采样（β = 负样本保留率 = {ds['beta']:.4f}，"
                    f"训练行数 {ds['n_train_rows']:,}）", fontsize=11)
    ax[0].margins(y=0.18)

    deep = [r for r in d["ablation"] if r["variant"] == "deepfm"][0]
    actual = deep["mean_p_ctr_corrected"] / (1 + 0.0)  # 占位，下面用真实基率
    # 真实基率 = 校正后应当对上的那个数（用 01 的 ctr 更准，但 03 里没有，
    # 所以用「校正后均值」和「原始均值」两点，画的是**校正前后**的对比）
    bars = [deep["mean_p_ctr_raw"], deep["mean_p_ctr_corrected"]]
    ax[1].bar(["未校正\n（采样后先验）", "先验校正后\nlogit += ln β"], bars,
              color=[C["r"], C["g"]], width=0.5)
    for i, v in enumerate(bars):
        ax[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
    ax[1].set_ylabel("pCTR 均值")
    ax[1].set_title(f"先验校正是必须的：漏掉会让 eCPM 高估 "
                    f"{deep['mean_p_ctr_raw'] / deep['mean_p_ctr_corrected']:.2f} 倍"
                    f"\n（校正后 ECE = {deep['ece_ctr']:.4f}）", fontsize=11)
    ax[1].margins(y=0.18)
    _ = actual
    return _save(fig, "fig05_imbalance_prior",
                 "⑤ 类别不平衡：下采样把正率抬上去，先验校正再把它搬回来",
                 top=0.79)


# --------------------------------------------------------------------------
# 6 延迟反馈去偏
# --------------------------------------------------------------------------


def fig06(d, cfg) -> str | None:
    if d is None:
        return None
    db = d["debias"]
    mods = db["models"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
    # model 名字很长，图上只留前半段
    lab = [m["model"].split("（")[0] for m in mods]
    bias = [m["bias_vs_w168"] * 100 for m in mods]
    cols = [C["r"], C["o"], C["g"]][:len(mods)]
    bars = ax[0].bar(lab, bias, color=cols, width=0.58)
    for b, v in zip(bars, bias):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%",
                   ha="center", va="top" if v < 0 else "bottom", fontsize=9)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].axhline(-db["false_negative_frac"] * 100, color=C["p"], ls="--",
                  lw=1.4,
                  label=f"线上标签自身的假负 {db['false_negative_frac']:.1%}")
    ax[0].set_ylabel("预测均值相对 168h 回补基率的偏差")
    ax[0].set_title(f"去偏效果：DFM 把偏差从 {abs(bias[0]):.1f}% 压到 "
                    f"{abs(bias[-1]):.1f}%", fontsize=11)
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", labelsize=8.5)
    ax[0].margins(y=0.2)

    _bar(ax[1], lab, [m["auc_vs_w168"] for m in mods], cols,
         "对 168h 回补标签的 AUC（去偏不该靠牺牲排序来换）",
         "AUC", "{:.4f}")
    ax[1].tick_params(axis="x", labelsize=8.5)
    return _save(fig, "fig06_delay_debias",
                 f"⑥ 延迟反馈去偏（受控对照：只有损失函数不同；"
                 f"elapsed 外推比例 {db['extrapolated_frac']:.0%}）",
                 top=0.79)


# --------------------------------------------------------------------------
# 7 ZILN
# --------------------------------------------------------------------------


def fig07(d, cfg) -> str | None:
    if d is None:
        return None
    sg, m = d["sigma"], d["metrics"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    # ⚠️ 这里画 **mean_pos**（付费样本口径），不画 mean_all。
    #    σ 是「付费金额」那一支的尺度参数，只有 y>0 的样本在估它；
    #    全体验证样本里有 90% 是 y=0，把它们平均进来会把 σ̄ 拉到 0.4964，
    #    跟真实 0.70 一比是 0.709 —— 而标注引的是比值 0.772，**柱子跟
    #    标注对不上**。两个口径都在 json 里（mean_all / mean_pos），
    #    图只画能跟真实 σ 比的那一个。
    _bar(ax[0], ["σ 估计均值\n(付费口径)", "真实 σ"], [sg["mean_pos"], sg["true_sigma"]],
         [C["b"], C["g"]], f"σ 头是活的（std {sg['std']:.4f}，"
         f"范围 [{sg['min']:.3f}, {sg['max']:.3f}]）\n"
         f"σ 估计系统性偏小：是真实值的 {sg['ratio_pos_to_true']:.2f} 倍（已知短板）",
         "σ", "{:.3f}")

    # 两个 PIT 的 p 值：同一个统计量、两个完全不同的问题
    pv = [m["pit_assumption_pvalue"], m["pit_model_pvalue"]]
    _bar(ax[1], ["① 真实参数\n（查建模前提）", "② 模型参数\n（查 μ/σ 估计准不准）"],
         pv, [C["g"], C["r"]], "两个 PIT：都要求 p > 0.05", "p 值", "{:.4f}",
         threshold=0.05, thr_label="门槛 0.05")
    ax[1].tick_params(axis="x", labelsize=8.5)

    vals = [m["aggregate_bias"], m["aggregate_bias_without_sigma2"],
            m["noise_floor"]]
    _bar(ax[2], ["含 exp(σ²/2)", "漏掉 σ²/2", "纯抽样噪声地板"], vals,
         [C["g"], C["r"], C["gray"]],
         "聚合偏差：漏掉 exp(σ²/2) 会让 eCPM 系统性低估", "相对偏差",
         "{:.4f}", threshold=0.50, thr_label="门槛 0.50")
    ax[2].tick_params(axis="x", labelsize=8.5)
    return _save(fig, "fig07_ziln",
                 f"⑦ ZILN 零膨胀模型（付费 AUC {m['auc_pay']:.4f} / "
                 f"理论上限 {m['auc_oracle']:.4f}；"
                 f"Normalized Gini {m['normalized_gini']:.4f}）",
                 top=0.77)


# --------------------------------------------------------------------------
# 8 智能出价
# --------------------------------------------------------------------------


def fig08(d06, d08, cfg) -> str | None:
    if d06 is None and d08 is None:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
    if d06 is not None:
        st = d06["stationary"]
        names = list(st)
        lab = [n.split("_")[0] for n in names]
        ax[0].bar(lab, [st[n]["expected_value"] for n in names],
                  color=[C["gray"], C["r"], C["o"], C["p"], C["b"], C["g"]],
                  width=0.62)
        for i, n in enumerate(names):
            ax[0].text(i, st[n]["expected_value"],
                       f"{st[n]['expected_value']:.0f}", ha="center",
                       va="bottom", fontsize=8.5)
        ax[0].set_ylabel("期望价值")
        ax[0].set_title(f"合成市场：六策略对照\nB4 相对人工固定出价 "
                        f"B0 提升 {d06['headline']['lift_vs_fixed_bid']:.0%}，"
                        f"达上帝视角的 {d06['headline']['ratio_b4_to_oracle']:.1%}",
                        fontsize=11)
        ax[0].margins(y=0.2)

        ov = [st[n]["overspend_frac"] * 100 for n in names]
        bars = ax[1].bar(lab, ov, color=[C["gray"], C["r"], C["o"], C["p"],
                                         C["b"], C["g"]], width=0.62)
        for b, v in zip(bars, ov):
            ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}%",
                       ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
        ax[1].axhline(0, color="k", lw=1)
        ax[1].axhline(0.5, color=C["r"], ls="--", lw=1.4, label="红线 0.5%")
        ax[1].set_ylabel("超支率 (%)")
        ax[1].set_title("预算控制：无一带正超支\n"
                        "（B1 贪心故意不带 pacing，用来证明 pacing 必要）",
                        fontsize=11)
        ax[1].legend(fontsize=8)
        ax[1].margins(y=0.2)
    else:
        ax[0].axis("off")
        ax[1].axis("off")

    if d08 is not None:
        runs = d08["runs"]
        tags = list(runs)
        x = np.arange(len(tags))
        b45 = [runs[t]["ratio_b4_to_oracle"] for t in tags]
        ax[2].bar(x - 0.2, b45, width=0.38, color=C["b"], label="B4/B5")
        ax[2].bar(x + 0.2, [runs[t]["loglog_elasticity"] for t in tags],
                  width=0.38, color=C["o"], label="弹性 γ")
        ax[2].axhline(0.40, color=C["r"], ls="--", lw=1.4, label="门槛 0.40")
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(tags, fontsize=8.5)
        for i, t in enumerate(tags):
            ax[2].text(i - 0.2, b45[i], f"{b45[i]:.3f}", ha="center",
                       va="bottom", fontsize=8.5)
            ax[2].text(i + 0.2, runs[t]["loglog_elasticity"],
                       f"{runs[t]['loglog_elasticity']:.3f}", ha="center",
                       va="bottom", fontsize=8.5)
        ax[2].set_title(f"接真实模型后的出价（最好变体 "
                        f"{d08['best_variant']}）\n"
                        f"Spearman 更高的那个反而更差 —— 排序要看 γ", fontsize=11)
        ax[2].legend(fontsize=8)
        ax[2].margins(y=0.22)
    else:
        ax[2].axis("off")
    return _save(fig, "fig08_bidding",
                 "⑧ 智能出价与水位控制",
                 top=0.75)


# --------------------------------------------------------------------------
# 9 A/B 实验
# --------------------------------------------------------------------------


def fig09(d, cfg) -> str | None:
    if d is None:
        return None
    aa, ab = d["aa"], d["ab"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    # AA 的 p 值分布：均匀是**应该**的（H0 为真），不是「跑得好」
    n = aa["replications"]
    ax[0].bar([0, 1], [aa["false_positive_rate"], 1 - aa["false_positive_rate"]],
              color=[C["r"], C["g"]], width=0.5)
    for i, v in enumerate([aa["false_positive_rate"],
                           1 - aa["false_positive_rate"]]):
        ax[0].text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9.5)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["p ≤ 0.05\n（假阳性）", "p > 0.05"], fontsize=9)
    ax[0].set_ylim(0, 1.15)
    ax[0].set_ylabel("占比")
    ax[0].set_title(f"AA 空跑 {n} 次：p 值应当服从 U(0,1)\n"
                    f"KS 均匀性 p = {aa['ks_pvalue']:.4f} > 0.05；"
                    f"假阳性率 {aa['false_positive_rate']:.1%}", fontsize=11)

    # 分位数条 —— 均匀分布的话应当近似 0.1/0.5/0.9
    q = [aa["p10"], aa["median"], aa["p90"]]
    _bar(ax[1], ["p10", "p50", "p90"], q,
         [C["b"], C["b"], C["b"]],
         f"p 值分位数（均匀分布的理论值 0.1 / 0.5 / 0.9）\n"
         f"单次 AA 的 p>0.05 不能当作「通过」", "p 值", "{:.3f}")
    for i, th in enumerate([0.1, 0.5, 0.9]):
        ax[1].plot([i - 0.35, i + 0.35], [th, th], color=C["r"], ls="--", lw=1.5)

    pw = d["power"]
    lab = ["忽略 deff\n（naive）", f"计 deff={pw['setup']['deff']:.2f}"]
    need = [pw["naive"]["total_users"], pw["deff_corrected"]["total_users"]]
    avail = pw["setup"]["n_users"]
    _bar(ax[2], lab, need, [C["o"], C["b"]],
         f"每臂 5% 相对 MDE / 80% 功效所需的用户数\n"
         f"可用 {avail:,}；计 deff 后达标（功效 "
         f"{pw['deff_corrected']['achieved_power']:.3f}）", "用户数", "{:,.0f}")
    ax[2].axhline(avail, color=C["r"], ls="--", lw=1.5,
                  label=f"可用 {avail:,}")
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", labelsize=8.5)
    return _save(fig, "fig09_experiment",
                 f"⑨ A/B 实验验证（{ab['exp_id']}）\n"
                 f"SRM p = {ab['srm_p']:.3f}；主指标 "
                 f"{ab['primary']['metric'].upper()} 提升 "
                 f"{ab['primary']['lift']:.2%}，p = "
                 f"{ab['primary']['p_value']:.4f}；CUPED 方差降 "
                 f"{ab['cuped']['variance_reduction']:.1%}",
                 top=0.72)


# --------------------------------------------------------------------------


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    banner(f"09 出图（{scale}）")

    d01 = _safe("01_data", cfg)
    d02 = _safe("02_features", cfg)
    d03 = _safe("03_ranker", cfg)
    d05 = _safe("05_ziln", cfg)
    d06 = _safe("06_bidding", cfg)
    d07 = _safe("07_experiment", cfg)
    d08 = _safe("08_bidding_sim", cfg)

    jobs = [
        ("fig01 数据地基", fig01, (d01, cfg)),
        ("fig02 特征与分层", fig02, (d02, cfg)),
        ("fig03 分层画像", fig03, (d02, cfg)),
        ("fig04 排序消融", fig04, (d03, cfg)),
        ("fig05 不平衡处理", fig05, (d03, cfg)),
        ("fig06 延迟去偏", fig06, (d03, cfg)),
        ("fig07 ZILN", fig07, (d05, cfg)),
        ("fig08 智能出价", fig08, (d06, d08, cfg)),
        ("fig09 A/B 实验", fig09, (d07, cfg)),
    ]
    made, skipped = [], []
    for name, fn, args in jobs:
        try:
            out = fn(*args)
        except Exception as e:                       # noqa: BLE001
            # 出图失败不该让整条链路挂掉，但**必须**记下来 ——
            # 静默少一张图，README 里的图号就会对不上。
            skipped.append(f"{name}：{type(e).__name__}: {e}")
            log.error("出图失败 %s：%s: %s", name, type(e).__name__, e)
            continue
        (made if out else skipped).append(name if out else f"{name}（缺指标 json）")

    show_table([{"图": m} for m in made], ["图"], f"已出图 {len(made)} 张 → {FIG_DIR}")
    if skipped:
        for s in skipped:
            log.warning("跳过：%s", s)

    # ``problems`` 这一栏别的脚本都有，这里也要有 —— run_all 和集成测试
    # 会统一遍历所有指标 json 检查它，缺一个键就会在那个循环里 KeyError。
    problems = [f"少图：{s}" for s in skipped]
    metrics = {"scale": scale, "n_figures": len(made), "figures": made,
               "skipped": skipped, "dir": str(FIG_DIR),
               "problems": problems}
    from scripts._common import save_metrics
    save_metrics("09_figures", metrics, cfg)

    if len(made) < 9:
        log.error("只出了 %d/9 张图 —— 先把上游脚本跑齐（run_all.py）",
                  len(made))
        return 1
    log.info("体检通过：9 张图全部出齐，全部读自 artifacts/metrics/*.json"
             "（与 README 的数字同源，不重算）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
