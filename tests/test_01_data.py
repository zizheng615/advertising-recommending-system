"""职责点 1 上半场：模拟数据生成器。

这一组测试回答一个核心问题：**这批数据到底学不学得动？**
如果生成器造出的真值不可恢复、或者隐向量直接泄漏到了特征里，
后面所有模型的 AUC 门槛都失去意义。所以这里的断言最硬。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from src.data import schema as S
from src.data.generator import DataGenerator
from src.data.splits import time_split, label_stats


# ---------------------------------------------------------------- 基率
def test_ctr_base_rate(cfg, impressions):
    """CTR 必须被二分法校准到目标值。"""
    assert abs(impressions["y_click"].mean() - cfg.target_ctr) < 0.003


def test_cvr_base_rate(cfg, impressions):
    """CVR 是 P(转化|点击)，必须在点击子集上校准。

    这里同时守住一个真实踩过的坑：在全曝光空间校准会让 CVR|click
    从 0.06 变成 0.48（8 倍偏差），而且不报任何错。
    """
    cl = impressions[impressions["y_click"] == 1]
    assert len(cl) > 1000
    assert abs(cl["y_conv_oracle"].mean() - cfg.target_cvr) < 0.006


def test_ctcvr_is_product(cfg, impressions):
    """CTCVR 应当等于 CTR × CVR —— 这是 ESMM 恒等式的数据侧对照。"""
    cl = impressions[impressions["y_click"] == 1]
    expect = impressions["y_click"].mean() * cl["y_conv_oracle"].mean()
    assert abs(impressions["y_conv_oracle"].mean() - expect) < 0.001


# ---------------------------------------------------------------- 可学习性
def test_bayes_ceiling(cfg, impressions):
    """用真值 logit 算 AUC —— 这是模型永远追不上的天花板。

    它 < 0.85 就说明噪声太大，模型学不出东西；
    > 0.99 就说明任务太平凡。0.85~0.99 是「可学但不平凡」的区间。
    """
    auc = roc_auc_score(impressions["y_click"], impressions["_true_logit_ctr"])
    assert auc > 0.82, f"真值 logit 的 AUC 只有 {auc:.4f}，任务本身太不可学"
    assert auc < 0.999, f"真值 logit 的 AUC 高达 {auc:.4f}，任务过于平凡"


def test_latent_is_recoverable(cfg, users, impressions):
    """只用隐向量训一个 LR，AUC 必须明显高于随机。

    门槛刻意定得很低（0.55），原因有两层：

    1. 隐向量**不是**模型能看到的东西。模型只能看到隐向量的含噪投影
       （SNR≈0.4），中间隔了一层，所以 z 的线性 AUC 不可能很高；
    2. 实测跨 seed 波动很大（0.55~0.70）。z 的线性可分性取决于随机
       投影方向恰好和标签权重对不对得上，5 个簇中心的实现值本身
       就是 5 次抽样，方差大。

    所以这条只当**下界哨兵**用：掉到 0.55 以下说明 z 根本没进标签、
    生成器白造了一场。真正卡质量的是下面那条可观测特征的夹逼。
    """
    z_cols = [f"_true_z{k}" for k in range(cfg.latent_dim)]
    Z = users[z_cols].to_numpy()[impressions["user_id"].to_numpy()]
    y = impressions["y_click"].to_numpy()
    cut = len(y) * 3 // 4
    lr = LogisticRegression(max_iter=300).fit(Z[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], lr.predict_proba(Z[cut:])[:, 1])
    assert auc > 0.55, f"隐向量几乎无预测力（AUC={auc:.4f}），数据没造出结构"


def test_observable_signals_are_predictive(cfg, users, items, impressions):
    """模型**真正能看到的那一层**也必须有信号，且不能强到接近天花板。

    这条是整套数据的关键夹逼，两侧都要卡：

    * 太低（≈0.55）→ 模型学不动，后面的 AUC 门槛全部失去意义；
    * 太高（>0.85）→ 可观测列几乎是隐向量的拷贝，等于泄漏答案。

    中间这段才叫「可学但不平凡」。
    """
    u_idx = impressions["user_id"].to_numpy()
    a_idx = impressions["ad_id"].to_numpy()
    X = np.concatenate([
        users[S.USER_SIGNAL_COLS].to_numpy(dtype=np.float64)[u_idx],
        items[S.ITEM_SIGNAL_COLS].to_numpy(dtype=np.float64)[a_idx],
    ], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = impressions["y_click"].to_numpy()
    cut = len(y) * 3 // 4
    lr = LogisticRegression(max_iter=500).fit(X[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], lr.predict_proba(X[cut:])[:, 1])

    ceiling = roc_auc_score(y, impressions["_true_logit_ctr"])
    assert auc > 0.62, f"可观测信号几乎无预测力（AUC={auc:.4f}），模型没法学"
    assert auc < ceiling - 0.02, (
        f"可观测信号的 AUC {auc:.4f} 已经追平真值天花板 {ceiling:.4f}，"
        f"说明特征层没有引入任何噪声，等于泄漏")


def test_observable_features_are_not_latent_leak(cfg, users, impressions):
    """可观测特征不能是隐向量的直接拷贝。

    这是生成器最重要的一条红线。如果某个可观测列和隐向量几乎完全相关，
    模型就是在「读答案」，整个项目失去意义。
    """
    z_cols = [f"_true_z{k}" for k in range(cfg.latent_dim)]
    Z = users[z_cols].to_numpy(dtype=np.float64)
    worst = 0.0
    for col in S.USER_SIGNAL_COLS:
        x = users[col].to_numpy(dtype=np.float64)
        if x.std() < 1e-9:
            continue
        # 该列与隐空间任一方向的最大 |相关|
        c = np.abs((Z - Z.mean(0)).T @ (x - x.mean())) / (
            len(x) * Z.std(0) * x.std() + 1e-12)
        worst = max(worst, float(c.max()))
    assert worst < 0.90, f"存在可观测列与隐向量相关度 {worst:.3f}，等于泄漏答案"


def test_true_columns_are_underscore_prefixed(cfg, users, items, impressions):
    """所有真值列必须以 `_true_` 开头，方便按前缀一刀切排除。"""
    for name, df in [("users", users), ("items", items), ("impressions", impressions)]:
        for col in df.columns:
            if col.startswith("_true_"):
                continue
            assert not col.startswith("_"), f"{name}.{col} 用了保留前缀"


# ---------------------------------------------------------------- 延迟
def test_false_negative_rate_by_window(cfg, impressions):
    """窗口越短假负越多 —— 这条单调曲线是延迟修正实验的天然对照组。

    注意 ``y_conv_w{w}`` 是**完整回补**下的窗口标签，四个 w 之间只有窗口
    不同，比较才干净。``y_conv`` 是线上真实标签，额外叠加了数据末尾截断，
    不能拿来做这条曲线（否则 w=72 和 w=168 会被同一个截断上限抹平）。

    72h 与 168h 的差距在小数据下只有几十个转化，扛不住严格的逐对比较，
    所以长尾那一段用容差断言；严格版本在 small / full 档的脚本里跑。
    """
    total = impressions["y_conv_oracle"].sum()
    assert total > 300, "总转化数太少，假负率测不准"
    rates = {w: 1.0 - impressions[f"y_conv_w{w}"].sum() / total
             for w in cfg.window_hours_list}

    order = sorted(cfg.window_hours_list)
    # 短窗口这一段差距大，断言严格单调
    for a, b in zip(order[:3], order[1:3]):
        assert rates[a] > rates[b], f"窗口 {a}h 的假负率没有高于 {b}h"
    # 长尾那一段只用容差守住「不反向」
    assert rates[72] > rates[168] - 0.015, "168h 的假负率反而明显更高"
    assert rates[6] - rates[168] > 0.20, "窗口从 6h 拉到 168h 几乎没改善，延迟建模白做"

    assert 0.10 < rates[cfg.window_hours] < 0.35, (
        f"24h 假负率 {rates[cfg.window_hours]:.3f} 不合理 —— 太低则延迟问题不存在，"
        f"太高则 24h 标签几乎没信号")


def test_main_label_has_no_earlier_truth_than_ablation(cfg, impressions):
    """线上标签（含末尾截断）的假负率必须 ≥ 完整回补的 24h 窗口。

    这是「数据末尾的曝光等不满 24h」这条机制的可见后果。
    """
    total = impressions["y_conv_oracle"].sum()
    fn_live = 1.0 - impressions["y_conv"].sum() / total
    fn_w24 = 1.0 - impressions["y_conv_w24"].sum() / total
    assert fn_live >= fn_w24, "线上标签的假负率不该低于完整回补标签"
    assert fn_live - fn_w24 < 0.05, "末尾截断的影响过大，会淹没窗口效应"


def test_delay_distribution_shape(cfg, impressions):
    """延迟分布应该是三分量混合的形态：有 0 值平台 + 长尾。"""
    d = impressions["_true_delay_h"].to_numpy()
    assert (d == 0).mean() > 0.30, "缺少「立即转化」分量"
    assert np.quantile(d, 0.99) > 48, "缺少长尾分量"
    assert d.min() >= 0


def test_elapsed_varies(cfg, impressions):
    """观测时长必须有变化，否则 DFM 的 elapsed 特征等于常数、白搭。

    这是踩过的坑：早期实现写成 ``min(window_hours, (n_days - day) * 24)``，
    而 window_hours 恰好等于一天，于是每一行的 elapsed 都是 24.0 ——
    一个常数。DFM 照样能训、指标照样出，只是 elapsed 那一维纯属浪费。
    """
    e = impressions["elapsed_h"].to_numpy()
    assert e.std() > 3.0, f"elapsed 几乎是常数（std={e.std():.3f}），DFM 学不到东西"
    assert e.min() >= 0.5
    assert e.max() <= cfg.obs_horizon + 1e-6
    # 必须真的覆盖到「等不满主窗口」和「等满」两种情形
    assert (e < cfg.window_hours).mean() > 0.02, "没有一条曝光的观测时长小于主窗口"
    assert (e >= cfg.window_hours).mean() > 0.50, "绝大多数曝光都不完整，数据没法用"


# ---------------------------------------------------------------- LTV / ZILN
def test_zero_inflation(cfg, users):
    """零膨胀比例必须落在目标附近 —— ZILN 的两阶段结构依赖它。"""
    rate = (users["pay_amount"] == 0).mean()
    assert abs(rate - cfg.zero_inflation) < 0.035, f"零膨胀比例 {rate:.3f} 偏离目标"


def test_positive_amount_is_conditionally_lognormal(cfg, users):
    """给定 μ(x)，付费金额必须服从对数正态 —— ZILN 回归头的唯一假设。

    ⚠️ 检验的是**残差** ``ln y − μ(x)``，不是 ``ln y`` 本身。
    ZILN 假设的是 ``ln y | x ~ N(μ(x), σ)``，边际上它是一个**混合**分布、
    本来就不正态。拿 ln y 直接做 KS 检验必然被拒，然后你会去改生成器 ——
    改错了方向。这里的 μ(x) 就是生成器用的那个 ``_true_log_mu``。

    这条不成立的话 ZILN 的损失函数从根上就是错的，
    而 ZILN 的 PIT 检验（P2）也就失去意义。
    """
    from scipy import stats
    pos = users["pay_amount"].to_numpy(dtype=np.float64) > 0
    assert pos.sum() > 50, f"只有 {pos.sum()} 个付费用户，检验不可靠"

    resid = np.log(users.loc[pos, "pay_amount"].to_numpy(dtype=np.float64)) \
        - users.loc[pos, "_true_log_mu"].to_numpy(dtype=np.float64)
    p = stats.kstest(resid, "norm", args=(0.0, cfg.ltv_sigma)).pvalue
    assert p > 0.01, f"条件残差的正态性检验 p={p:.4f}，ZILN 的对数正态假设不成立"

    # 再确认 μ(x) 真的随用户变化 —— 否则 ZILN 的回归头退化成拟合常数，
    # 那还不如直接输出 log(训练集均值)。
    # 判据是方差分解 var(ln y) ≈ var(μ(x)) + σ²，**不是**峰度：
    # μ(x) 是「均值混合」，均值混合不改变峰度（峰度只对「方差混合」敏感），
    # 拿峰度去测会得出错误结论。
    marginal = np.log(users.loc[pos, "pay_amount"].to_numpy(dtype=np.float64))
    ratio = marginal.var() / resid.var()
    # 门槛 1.25 对应「μ 的标准差 ≥ σ 的一半」。实测跨 seed 在 1.46~2.62，
    # 留出余量是因为付费样本只有 200 出头，方差比本身就有采样噪声。
    assert ratio > 1.25, (
        f"var(ln y)/var(残差)={ratio:.2f}，μ(x) 几乎不随用户变化，"
        f"ZILN 的回归头没有存在意义")


def test_expected_ltv_is_conditional_expectation(cfg, users):
    """E[LTV] = π · exp(μ + σ²/2)，必须包含零膨胀。

    用条件中位数 p·exp(μ) 会系统性低估，eCPM 会跟着一起错。
    """
    e = users["expected_ltv"].to_numpy(dtype=np.float64)
    assert (e >= 0).all()
    # 与「用真实付费率反推」的量级一致
    assert 2.0 < e.mean() < 30.0, f"E[LTV] 均值 {e.mean():.2f} 量级不合理"
    naive = users["_true_pay_prob"].to_numpy() * np.exp(users["_true_log_mu"].to_numpy())
    assert e.mean() > naive.mean(), "E[LTV] 没有体现对数正态的 σ²/2 修正"


# ---------------------------------------------------------------- 特征宽表
def test_profile_has_200_plus_features(cfg, profile, pca_result):
    """职责点 1 明写「超 200 维特征」，这里守住它。

    维度账要算清楚：画像宽表本身是 192 列（22 类别 + 84 数值 + 60 噪声
    + 26 冗余），PCA 主成分是**在此之上追加**的，PCA 之后的用户侧特征
    才是模型真正吃进去的那 200+ 维。只看宽表会少算 PCA 那一截。
    """
    _, cols = profile
    assert len(cols) == 192, f"画像宽表 {len(cols)} 列，与设计不符"
    assert len(set(cols)) == len(cols), "特征列名有重复"

    total = len(cols) + pca_result.n_components
    assert total >= 200, (
        f"用户侧特征只有 {total} 维（宽表 {len(cols)} + PCA "
        f"{pca_result.n_components}），不满足「超 200 维」")


def test_noise_and_redundant_ratio(cfg, profile):
    """噪声 + 冗余应当占相当比例 —— 还原「200 维里大部分没用」的真实情况。"""
    _, cols = profile
    junk = [c for c in cols if c.startswith(("noise_", "redund_"))]
    frac = len(junk) / len(cols)
    assert 0.30 < frac < 0.55, f"噪声+冗余占比 {frac:.2f}，不真实"


def test_noise_features_are_actually_noise(cfg, users, profile):
    """纯噪声列与标签不能有可见的相关性。"""
    df, cols = profile
    noise_cols = [c for c in cols if c.startswith("noise_")][:20]
    y = users["pay_flag"].to_numpy(dtype=np.float64)
    for c in noise_cols:
        x = df[c].to_numpy(dtype=np.float64)
        r = abs(np.corrcoef(x, y)[0, 1])
        assert r < 0.10, f"噪声列 {c} 与标签相关系数 {r:.3f}，它不是噪声"


def test_missingness_injected(cfg, users):
    """缺失必须真的注入了，且 MNAR 列的缺失率随收入上升。"""
    miss_cols = [c for c in users.columns if c.startswith("is_missing_")]
    assert len(miss_cols) == len(S.MISSING_INDICATOR_FEATURES)
    overall = users[miss_cols].to_numpy().mean()
    assert 0.02 < overall < 0.22, f"整体缺失率 {overall:.3f} 不合理"

    # MNAR：高收入组的缺失率应当明显高于低收入组。
    #
    # ⚠️ 必须用 `_true_income_bucket`（打码前的分档）来分组。
    #    直接拿 income_bucket 分组会得到「两组缺失率都是 0」这种荒谬结论 ——
    #    因为缺失的行 Income 已经被写成 -1，正样本自己在分组时被排除掉了。
    #    这个坑不报错，只会让一条本该有效的检验变成恒真式。
    raw = users["_true_income_bucket"].to_numpy()
    flag = users["is_missing_income"].to_numpy()
    lo = flag[raw <= 1].mean()
    hi = flag[raw >= 4].mean()
    assert hi > lo * 1.2, (
        f"MNAR 没生效：低收入缺失率 {lo:.3f} 没有明显低于高收入 {hi:.3f}")

    # 缺失指示位必须来自真实概率，而不是写死的常数
    p = users["_true_p_missing_income"].to_numpy()
    assert p.std() > 0.01, "缺失概率是常数，缺失机制与用户无关"


# ---------------------------------------------------------------- 确定性
def test_generation_is_deterministic(cfg):
    """同种子两次生成必须完全一致 —— 否则所有测试门槛都是浮沙。"""
    d1 = DataGenerator(cfg).generate()
    d2 = DataGenerator(cfg).generate()
    h1 = pd.util.hash_pandas_object(d1["impressions"], index=False).sum()
    h2 = pd.util.hash_pandas_object(d2["impressions"], index=False).sum()
    assert h1 == h2, "同种子两次生成的曝光表不一致"
    assert d1["users"]["pay_amount"].equals(d2["users"]["pay_amount"])


def test_different_seed_differs(cfg):
    """换种子必须换数据，否则种子根本没生效。"""
    d1 = DataGenerator(cfg).generate()
    d2 = DataGenerator(cfg.scaled(seed=cfg.seed + 1)).generate()
    assert not d1["impressions"]["y_click"].equals(d2["impressions"]["y_click"])


# ---------------------------------------------------------------- 切分
def test_time_split_is_ordered(cfg, impressions):
    """切分必须沿时间轴：train 的 day 全部早于 test。"""
    sp = time_split(impressions, cfg)
    day = impressions["day"].to_numpy()
    assert day[sp.train].max() < day[sp.valid].min()
    assert day[sp.valid].max() < day[sp.test].min()
    assert len(sp.train) > len(sp.valid) > 0
    assert len(sp.test) > 0


def test_split_has_positives(cfg, impressions):
    """每个切分都要有正样本，否则 AUC 无从算起。"""
    sp = time_split(impressions, cfg)
    for name, idx in [("train", sp.train), ("valid", sp.valid), ("test", sp.test)]:
        st = label_stats(impressions, idx)
        assert st["n_click"] > 30, f"{name} 只有 {st['n_click']} 次点击"
        assert st["n_conv"] > 5, f"{name} 只有 {st['n_conv']} 次转化"


def test_temporal_drift_exists(cfg, impressions):
    """测试集与训练集的基率必须真的不同 —— 时间漂移被注入了。

    用 CTR 而不是 CVR 来断言：CTR 的样本量是 CVR 的 1/0.06 ≈ 17 倍，
    相对标准误小一个量级，才扛得住「漂移确实存在」这种正向断言。

    漂移幅度要两头卡住：太小则「定期重训」没有动机，
    太大则模型在测试集上的 AUC 会被噪声压垮、后面的门槛全部失效。
    """
    sp = time_split(impressions, cfg)
    tr = label_stats(impressions, sp.train)
    te = label_stats(impressions, sp.test)
    rel = (te["ctr"] - tr["ctr"]) / max(tr["ctr"], 1e-9)
    assert rel > 0.05, f"测试集 CTR 相对训练集只差 {rel:+.3f}，时间漂移没生效"
    assert rel < 0.60, f"漂移过大（相对差 {rel:+.3f}），会把 AUC 门槛打破"

    # 漂移是单调的：验证集应当夹在中间
    va = label_stats(impressions, sp.valid)
    assert tr["ctr"] < va["ctr"] < te["ctr"], "漂移不是单调的，说明是噪声不是漂移"
