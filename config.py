"""全局配置：单一真相源。

所有超参数集中在此，脚本与测试都从这里取，避免"魔数散落各处"。
`get_config(scale)` 返回对应规模档的配置对象。

规模档说明
----------
mini  : 单元测试用。数据生成是秒级，但全链路**不是** —— 03 要训 LR/FM/DeepFM
        三个模型外加 DFM 去偏，把每个模型都跑满 epoch 才是它该有的样子，
        单机 CPU 上是十几分钟的量级。档位小 ≠ 训练快。
small : 默认演示档，全链路更久
full  : 对应职责描述里的「10 万活跃用户」，曝光事件按上限采样

⚠️ 不同档位之间**不要互相引用数字**：数据量、训练轮数、模型容量都变了。
    ``README.md`` 里所有数字都标注了 ``--scale mini``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
# NOTE: 项目路径含空格，全程用 pathlib，绝不手工拼 shell 字符串。
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SQL_DIR = ROOT / "sql"

for _d in (ARTIFACTS, ARTIFACTS / "data", ARTIFACTS / "models",
           ARTIFACTS / "figures", ARTIFACTS / "metrics"):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 单位约定 —— 这是最经典的 bug 源，写死常量，全局只引用它
# --------------------------------------------------------------------------
# eCPM 单位是「元 / 千次曝光」，真实出价单位是「元 / 单次曝光」。
# 90% 的「预算差 1000 倍」都是这里搞错的。
CPM_TO_IMPRESSION = 1.0 / 1000.0


@dataclass
class Config:
    """一次实验的全部配置。"""

    scale: str = "small"
    seed: int = 42

    # ---------------- 数据规模 ----------------
    n_users: int = 8_000
    n_items: int = 600
    n_days: int = 20
    max_impressions: int = 600_000
    impressions_per_user_day: int = 10

    # ---------------- 真值生成（RFF 是核心） ----------------
    # σ_ω 是「高阶交叉能量」的单一旋钮：
    #   LR 只能拟合 f 的线性部分、FM 拟合二阶、DNN 拟合剩余高阶。
    #   调大 σ_ω → DeepFM 相对 FM/LR 的优势变大。
    rff_dim: int = 256
    # σ_ω 是「高阶交叉能量」的旋钮，也是**过犹不及**的典型：
    #   调太小 → 真值近乎线性，LR 就够了，DeepFM 赢了也没意义；
    #   调太大 → ωᵀx 的频率高到真值在特征空间里像白噪声，
    #            所有模型一起学不会，AUC 全线趴下、差值消失。
    # 实测 σ_ω = 1.0 时 RFF 那一路基本是不可学的噪声，
    # 降到 0.8 后它才变成一个「DNN 能抓一部分、LR/FM 抓不到」的成分。
    rff_sigma_w: float = 0.8
    rff_scale: float = 0.7

    # 显式交叉项强度 —— 这组数是**刻意调出来的**，不是随手写的。
    #
    # emb_dim=8 时 FM 其实能表达 2×2 的 XOR（rank-2，8 维装得下），
    # 真正让 FM 无能为力的是**三阶及以上**——它的模型形式里压根没这一项。
    # 所以 DeepFM 的优势**必须**主要来自 alpha_3 / alpha_4。
    #
    # ⚠️ 初版取 alpha_2=1.2 / alpha_3=1.0 / alpha_4=0.8，实测三阶+四阶的
    #    CTR AUC 只有 0.67，而**单独一个二阶项就有 0.80**。
    #    也就是说信号几乎全在 FM 的主场里，FM 逼近上界、DeepFM 无处施展，
    #    ``AUC(DeepFM) − AUC(FM)`` 在 0 附近随机震荡（实测 ±0.04）。
    #    这属于「实验设计没给结论留出空间」——测试怎么写都会 flaky。
    #
    #    调整方向不是硬调阈值，而是让**高阶信号真的占总信号量的一席之地**：
    #    alpha_2 下调到 0.7（FM 仍拿得到，但不再是全部），
    #    alpha_3/alpha_4 上调到 1.8/1.7。
    #    调整后三阶+四阶的 std 约 2.3，与二阶的 1.4 同量级。
    #    第二轮又发现：光「同量级」还不够。各成分**单独**的 AUC 才是关键 ——
    #    实测 alpha_2=0.7 时，单看二阶项 CTR AUC 0.69、单看三阶+四阶 0.84，
    #    差距只有 0.15，而模型能从别的特征里补回来，消融依然不稳。
    #    现在取 alpha_2=0.45 / alpha_3=2.4 / alpha_4=2.2，单成分 AUC 变成
    #    **0.62 对 0.90** —— 「只会二阶」和「会三阶」的模型上限被真正拉开了。
    #    这不是为了让某个模型赢而调参，而是让实验**有能力**回答它要回答的问题。
    alpha_2: float = 0.45    # 二阶（用户信号 × 素材信号），FM 抓得到
    alpha_3: float = 2.4     # 三阶张量，FM 抓不到、DNN 抓得到
    alpha_4: float = 2.2     # 四阶张量，同上
    # 分层 × 素材。做成「低秩 + 稀疏高秩残差」：低秩那半 FM 用
    # seg×creative 的二阶交叉就能表达，稀疏残差那 6 个格子只有 DNN 能拟合。
    alpha_seg: float = 1.2
    seg_rank: int = 2        # 低秩部分的秩（FM 可表达）
    seg_sparse_n: int = 6    # 稀疏高秩残差的非零格子数（FM 不可表达）

    latent_dim: int = 8      # 隐向量维度
    n_true_segments: int = 5  # 人工注入的真实分层数（KMeans 应能恢复）

    target_ctr: float = 0.050
    target_cvr: float = 0.060          # P(转化 | 点击)

    # ---------------- 延迟反馈 ----------------
    # 三分量混合：立即 / 同会话 / 长尾
    delay_mix: tuple = (0.40, 0.25, 0.35)
    delay_uniform_max_h: float = 2.0
    delay_lognorm_mu: float = math.log(36.0)
    delay_lognorm_sigma: float = 1.0
    delay_truncate_h: float = 720.0
    window_hours: int = 24             # 主观察窗口
    window_hours_list: tuple = (6, 24, 72, 168)   # 消融用

    # ---------------- 时间漂移 ----------------
    # 模型看不到 day 特征，所以漂移是它解释不了的分布偏移：
    # 测试集 AUC 会低于验证集，概率校准会失效 —— 这正是「为什么要定期重训」。
    drift_rate: float = 0.30           # 市场季节性：logit 整体抬升/下压
    drift_fatigue: float = 0.40        # 素材疲劳：素材信号的有效系数随时间增长

    # ---------------- LTV / 付费金额（ZILN 用） ----------------
    zero_inflation: float = 0.90       # 90% 用户永不付费
    ltv_mu_log: float = math.log(60.0)
    # σ 不能太大：σ² 是 μ(x) 要打败的那个噪声底。σ=0.85 时 σ²=0.72，
    # 而付费样本里 var(μ) 只有 0.2 上下 —— 回归头就退化成拟合常数了。
    ltv_sigma: float = 0.70
    ltv_amt_min: float = 1.0
    ltv_amt_max: float = 5000.0

    # ---------------- 特征工程 ----------------
    n_cat_fields: int = 40
    n_noise_features: int = 60
    n_redundant_features: int = 26
    missing_rate_target: float = 0.06

    # ---------------- PCA / KMeans ----------------
    pca_var_keep: float = 0.90
    pca_n_max: int = 40
    kmeans_k_range: tuple = (3, 13)   # [3, 12]
    kmeans_n_init: int = 10
    kmeans_silhouette_sample: int = 10_000

    # ---------------- 模型 ----------------
    emb_dim: int = 8
    emb_dim_large: int = 4             # 大基数域（user_id/ad_id/campaign_id）
    dnn_hidden: tuple = (128, 64)
    dropout: float = 0.2
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 15
    early_stop_patience: int = 2
    grad_clip: float = 1.0
    neg_pos_ratio: int = 5
    tower_dim: int = 64
    temperature: float = 0.07

    # ---------------- ZILN ----------------
    ziln_hidden: int = 64
    sigma_min: float = 1e-3
    sigma_max: float = 5.0
    ziln_epochs: int = 20
    ziln_batch: int = 512
    ziln_lr: float = 1e-3
    ziln_loss_weight: float = 1.0

    # ---------------- 出价 / 预算控制 ----------------
    t_slots: int = 96                  # 每天 96 个 15min slot
    daily_budget: float = 5000.0       # 元
    market_cpm_mean: float = 25.0      # 市场平均 CPM（元/千次）
    market_sigma_m: float = 0.6
    k_p: float = 6.0
    k_i: float = 0.8
    k_d: float = 0.15
    u_max: float = 0.7
    i_max: float = 0.5
    pid_deadzone: float = 0.002
    eta_lambda: float = 0.12
    ema_rho: float = 0.8
    alpha_min: float = 0.05
    alpha_max: float = 5.0
    # ⚠️ 这两个参数**必须满足** budget_safety + reserve_frac < 1，否则：
    #    计划花费 (budget_safety·B) 超过了硬闸门允许的上限 ((1−reserve)·B)，
    #    闸门会天天触发。闸门触发意味着「前两环已经失效」，
    #    是最后一道保险，不该出现在正常的一天里。
    #    同时 budget_safety 也**决定了下界**：一个完美的控制器
    #    只能把预算误差做到 1−budget_safety，所以「误差<2%」这个门槛
    #    要求 budget_safety ≥ 0.98。两个约束一起把可取值范围压得很窄，
    #    改动前先看一眼 tests/test_06_bidding.py 里的
    #    test_pacing_parameters_are_mutually_consistent。
    budget_safety: float = 0.99        # 全天计划花 99%
    reserve_frac: float = 0.005        # 硬闸门留 0.5% 的兜底
    max_dlog_alpha: float = 0.10
    integral_decay: float = 0.95

    # ---------------- A/B 实验 ----------------
    alpha_sig: float = 0.05
    power: float = 0.80
    n_buckets: int = 10_000
    srm_p_threshold: float = 0.001
    aa_replications: int = 200
    cuped_min_rho: float = 0.30
    bh_q: float = 0.10

    # ---------------- 训练子集（CPU 友好） ----------------
    max_train_rows: int = 400_000

    # 派生量
    @property
    def prior_logit_shift(self) -> float:
        """1:5 负下采样后，推理时必须把 logit 平移回真实先验。

        漏掉这一步会让 eCPM 系统性高估约 4 倍。
        """
        return math.log(1.0 / self.neg_pos_ratio)

    @property
    def cpm_to_impression(self) -> float:
        return CPM_TO_IMPRESSION

    @property
    def obs_horizon(self) -> float:
        """曝光发生之后，最多能观测多久。

        取最长消融窗口：只有观测满 168h，「168h 窗口下还剩多少假负」
        才是个有意义的问题。观测时长的上界取它，下界是数据末尾那一刻。
        """
        return float(max(self.window_hours_list))

    def scaled(self, **kw) -> "Config":
        return replace(self, **kw)


# --------------------------------------------------------------------------
# 规模档
# --------------------------------------------------------------------------
_PROFILES = {
    # 单元测试档。
    #
    # ⚠️ 规模不是随便定的，是被**模型消融的统计功效**反推出来的。
    #
    #    最初用 2000 用户 / 15 万曝光。那时 CTCVR（曝光→点击→转化）的
    #    正样本在验证+测试集上只有 100 出头 —— AUC 的标准误约 0.02，
    #    而 DeepFM 与 FM 的真实差距只有 0.01 量级。结果就是
    #    ``AUC(DeepFM) − AUC(FM)`` 在不同随机种子下正负乱跳，
    #    测试怎么写都会 flaky，而且你无法区分「模型没用」和「样本不够」。
    #
    #    扩到 40 万曝光后 CTCVR 正样本约 360 个，CTR 正样本约 1.2 万，
    #    差距才稳定地显正。生成+特征+训练总共约 2 分钟，可以接受。
    "mini": dict(
        n_users=4_000, n_items=400, n_days=16, max_impressions=400_000,
        impressions_per_user_day=8, epochs=8, ziln_epochs=10,
        max_train_rows=400_000, kmeans_silhouette_sample=6_000,
        aa_replications=60,
    ),
    # 默认演示档
    "small": dict(
        n_users=8_000, n_items=600, n_days=20, max_impressions=600_000,
        impressions_per_user_day=10, epochs=8, ziln_epochs=10,
        max_train_rows=400_000, kmeans_silhouette_sample=8_000,
        aa_replications=100,
    ),
    # 对应职责描述的 10 万活跃用户
    "full": dict(
        n_users=100_000, n_items=2_000, n_days=30, max_impressions=2_000_000,
        impressions_per_user_day=10, epochs=15, ziln_epochs=20,
        max_train_rows=1_500_000, kmeans_silhouette_sample=10_000,
        aa_replications=200,
    ),
}


def get_config(scale: str = "small", **overrides) -> Config:
    """取某个规模档的配置，可再用关键字覆盖任意字段。"""
    if scale not in _PROFILES:
        raise ValueError(f"未知规模档 {scale!r}，可选：{sorted(_PROFILES)}")
    cfg = Config(scale=scale, **{**_PROFILES[scale], **overrides})
    return cfg


# 类别域 / 特征列的定义统一放在 src/data/schema.py，避免两处真相源。
# 这里只留一个引用，方便脚本按 `from config import schema` 取用。
