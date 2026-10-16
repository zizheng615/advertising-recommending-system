"""特征宽表的声明式定义。

这里是「列名」的单一真相源：生成器按它造列，模型按它取列，
测试按它断言维度。任何一处想加列，都只改这个文件。

列数账（用户画像宽表 ``profile_cols``，目标 ≥200 维）
------------------------------------------------------
    类别 22 + 行为 30 + 货币 8 + 用户侧广告 6 + 统计聚合 30
    + 缺失指示 10 + 纯噪声 60 + 冗余 26                     = 192

其中噪声 + 冗余 = 86 列，占 **44.8%** —— 刻意还原「200 维里大部分没用」
的真实情况，也让 PCA 和特征筛选确实有意义。

⚠️ 这个总数**只数特征列**，不含 ``users`` 表里的簿记与隐变量列
（``user_id`` / ``pay_amount`` / ``_true_segment`` 等 17 列）。它们会跟着
``users`` 被 copy 进画像 DataFrame，但既不进模型，也不属于「200+ 维特征」。
``scripts/02_features.py`` 一开始数的是 ``DataFrame.columns``，于是报出
209 列 —— 比声明值多出的 17 正是这批列，而那条日志的右边加起来只有 192，
自己跟自己矛盾。

PCA 主成分**不计入**这个 192：它是从这 192 列里**算出来**的降维结果
（实测 40 维，见 ``02_features.json`` 的 ``model_pca``），属于下游的
建模输入，不是画像宽表的一部分。
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# 数值特征分组
# --------------------------------------------------------------------------
# ⚠️ 信号列全部是隐向量的**含噪函数**（SNR ≈ 0.4），绝不是隐向量本身。
#    直接放隐向量的话模型三行代码就能到 AUC 0.95，整个项目就失去意义。
BEHAVIORAL_FEATURES = [
    "sessions_7d", "sessions_30d", "active_days_30d", "avg_session_len",
    "page_views_7d", "page_views_30d", "click_cnt_7d", "click_cnt_30d",
    "impression_cnt_7d", "impression_cnt_30d", "ctr_7d", "ctr_30d",
    "cvr_7d", "cvr_30d", "search_cnt_7d", "collect_cnt_30d",
    "share_cnt_30d", "comment_cnt_30d", "follow_cnt_30d", "video_view_cnt_7d",
    "ad_click_cnt_7d", "ad_click_cnt_30d", "ad_ctr_7d", "ad_ctr_30d",
    "recency_days", "frequency_7d", "monetary_30d", "interaction_depth",
    "night_owl_score", "weekend_ratio",
]

MONETARY_FEATURES = [
    "hist_ltv", "arpu_30d", "gmv_mean", "gmv_max",
    "order_cnt_30d", "avg_order_amount", "coupon_usage_rate", "refund_rate",
]

# 用户侧广告统计（可在用户画像阶段算出）
USER_AD_CONTEXT = [
    "advertiser_quality", "category_affinity", "price_affinity",
    "creative_fatigue_score", "competition_index", "slot_traffic_share",
]

# 曝光侧广告统计（依赖具体素材，只能在曝光层算）
IMPRESSION_AD_CONTEXT = [
    "user_x_category_ctr", "user_x_creative_cvr", "ad_ctr_global",
    "ad_cvr_global", "bid_floor_cpm", "user_x_category_hit",
]

# 跨域计数特征（同样只能在曝光层算 —— 它们同时依赖用户和素材）
CROSS_COUNT_NAMES = [
    "user_x_advertiser_cnt", "user_x_category_cnt", "user_x_priceband_cnt",
    "user_x_slot_cnt", "user_x_creative_cnt", "seg_x_creative_cnt",
]

# 统计聚合 / 派生比率（同样是隐向量的含噪函数）
STAT_FEATURES = [
    "ratio_click_session", "ratio_page_click", "ratio_ad_click",
    "ratio_video_session", "ratio_search_session", "ratio_collect_click",
    "roll7_ctr_std", "roll30_ctr_std", "roll7_cvr_std", "roll30_cvr_std",
    "roll7_spend_sum", "roll30_spend_sum", "roll7_order_sum", "roll30_order_sum",
    "roll7_session_mean", "roll30_session_mean", "roll7_gap_mean", "roll30_gap_mean",
    "decay_activity_7d", "decay_activity_30d", "trend_session", "trend_click",
    "trend_spend", "trend_recency", "entropy_category", "gini_category",
    "concentration_top1", "diversity_score", "loyalty_score", "churn_risk",
]

MISSING_INDICATOR_FEATURES = [
    "is_missing_income", "is_missing_city", "is_missing_occupation",
    "is_missing_education", "is_missing_marital", "is_missing_device_brand",
    "is_missing_app_category_pref", "is_missing_interest_top2",
    "is_missing_price_affinity", "is_missing_coupon_usage_rate",
]

# 用户画像阶段的全部数值列（PCA 列由 features/pca.py 追加）
USER_PROFILE_NUMERIC = (
    BEHAVIORAL_FEATURES + MONETARY_FEATURES + USER_AD_CONTEXT
    + STAT_FEATURES + MISSING_INDICATOR_FEATURES
)

# --------------------------------------------------------------------------
# 非负、重尾的列 —— 计数与金额。喂模型前必须先 log1p 再标准化。
# --------------------------------------------------------------------------
# ⚠️ 这件事漏掉过一次，后果很隐蔽，值得单独记一笔。
#
#    原先数值块是**原始值**直接过一层线性投影。曝光侧的
#    ``seg_x_creative_cnt``（分层 × 创意 的曝光计数）std 有 6921、
#    最大 2.7 万，而同一块里 ``ad_cvr_global`` 之类只有 0.03 ——
#    差 5 个数量级。FM 的二阶项是嵌入向量的**双线性型**，
#    数值块那一路的模长被放大后，二阶项的输出在**初始化那一刻**
#    就是 ±3000，epoch 1 的训练 loss 冲到 165（正常是 0.5），
#    然后花好几个 epoch 把投影权重慢慢压回来。
#
#    最坑的是它**看起来像 FM 实现有 bug**：LR 完全正常（一阶项对数值块
#    是线性的，而且 ``dense_linear`` 零初始化，起步不受影响），
#    只有带二阶项的变体会炸 —— 于是很容易得出
#    「FM 这个模型不行」这个完全错误的结论。
LOG1P_FEATURES = [
    "sessions_7d", "sessions_30d", "active_days_30d", "page_views_7d",
    "page_views_30d", "click_cnt_7d", "click_cnt_30d", "impression_cnt_7d",
    "impression_cnt_30d", "search_cnt_7d", "collect_cnt_30d", "share_cnt_30d",
    "comment_cnt_30d", "follow_cnt_30d", "video_view_cnt_7d",
    "ad_click_cnt_7d", "ad_click_cnt_30d", "frequency_7d",
    "monetary_30d", "hist_ltv", "arpu_30d", "gmv_mean", "gmv_max",
    "avg_order_amount", "order_cnt_30d",
    "roll7_spend_sum", "roll30_spend_sum", "roll7_order_sum", "roll30_order_sum",
] + CROSS_COUNT_NAMES

# 用户画像阶段的类别域（用户侧）
USER_CAT_FIELDS: list[tuple[str, int]] = [
    ("gender", 3), ("age_bucket", 7), ("city_tier", 5), ("province", 34),
    ("device_brand", 20), ("device_price_band", 5), ("os", 4),
    ("network_type", 4), ("carrier", 4), ("app_category_pref", 15),
    ("income_bucket", 6), ("marital", 3), ("education", 5), ("occupation", 12),
    ("interest_top1", 30), ("interest_top2", 30), ("os_version", 12),
    ("session_len_bucket", 6), ("purchase_power_tier", 5),
    ("recency_bucket", 6), ("frequency_bucket", 6), ("is_new_user", 2),
]

# 素材侧类别域
ITEM_CAT_FIELDS: list[tuple[str, int]] = [
    ("ad_category", 25), ("creative_type", 8), ("creative_format", 6),
    ("landing_page_type", 10), ("price_band", 8), ("color_tone", 6),
    ("has_video", 2),
]

# 上下文类别域
CONTEXT_CAT_FIELDS: list[tuple[str, int]] = [
    ("hour", 24), ("dow", 7), ("time_slot", 4),
    ("position_in_page", 10), ("page_type", 8),
]

# 大 ID 域 —— 从 CAT_FIELDS 里单独拎出来标明用更小的 embedding
ID_CAT_FIELDS: list[tuple[str, int]] = [
    ("user_id", 100_000), ("ad_id", 2_000),
    ("advertiser_id", 300), ("campaign_id", 1_000),
]

# 分段 / 交叉域（由分层产出，属于跨域特征，只能进排序层）
SEGMENT_CAT_FIELDS: list[tuple[str, int]] = [
    ("user_segment", 12), ("seg_x_creative", 96),
]


def make_noise_features(n: int) -> list[str]:
    """纯噪声列 —— 毫无信号，用来验证特征筛选确实有意义。"""
    return [f"noise_{i:03d}" for i in range(n)]


def make_redundant_features(n: int, source: list[str]) -> list[str]:
    """冗余列 —— 现有列的线性组合 + 微量噪声。"""
    return [f"redund_{i:03d}" for i in range(n)]


def make_pca_features(n: int) -> list[str]:
    return [f"pca_{i:02d}" for i in range(n)]


CROSS_COUNT_FEATURES = [
    "user_x_advertiser_cnt", "user_x_category_cnt", "user_x_priceband_cnt",
    "seg_x_creative_cnt", "user_x_slot_cnt", "ad_x_citytier_cnt",
]

# --------------------------------------------------------------------------
# 驱动标签的「信号列」（生成器与诊断共用）
# 隐向量 → 这些可观测列（含噪）→ 标签。模型只能看到中间那一层。
# --------------------------------------------------------------------------
USER_SIGNAL_COLS = [
    "sessions_7d", "ctr_7d", "cvr_30d", "recency_days",
    "interaction_depth", "night_owl_score", "hist_ltv", "arpu_30d",
    "avg_order_amount", "category_affinity", "loyalty_score", "diversity_score",
]
ITEM_SIGNAL_COLS = ["ad_ctr_global", "ad_cvr_global", "bid_floor_cpm", "item_popularity"]

# --------------------------------------------------------------------------
# 显式高阶交叉用到的分箱键（生成器与模型必须引用同一套定义）
# --------------------------------------------------------------------------
HIGH_ORDER_KEYS = ["u_act", "a_pop", "ctx_slot", "page_kind"]
HIGH_ORDER_CARD = {"u_act": 4, "a_pop": 4, "ctx_slot": 4, "page_kind": 4}

# 供 FM 使用的分箱字段：连续特征必须先分箱才能进入 FM 的二阶项
BINNED_FIELDS: list[tuple[str, int]] = [
    ("bin_activity", 5), ("bin_ctr_hist", 5), ("bin_cvr_hist", 5),
    ("bin_recency", 5), ("bin_monetary", 5), ("bin_ltv_hist", 5),
    ("bin_ad_ctr", 5), ("bin_category_affinity", 5), ("bin_price_affinity", 5),
    ("bin_fatigue", 4), ("bin_competition", 4), ("bin_session_len", 4),
]
N_PCA_COMPONENTS = 24
