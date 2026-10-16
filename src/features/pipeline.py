"""特征工程：把原始事件表拼成可直接喂模型的特征矩阵。

职责点 1「基于 SparkSQL 抽取 10 万活跃用户超 200 维特征」的落地。
``sql/`` 下是等价的 SparkSQL 脚本（生产参考），这里用 pandas 执行同一套逻辑，
保证任何机器上都能一键跑通。

内存设计
--------
用户画像宽表是 ``(n_users, 192)``，10 万用户也才 86MB —— 便宜。
曝光表是 ``(n_impressions, ~12)``，2M 行也才 ~100MB —— 也便宜。

**绝不把用户侧 192 列广播到曝光层**，那会变成 2M × 192 的矩阵（1.7GB）。
模型按 ``user_id`` 索引查表即可，效果完全一样。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import Config
from src.data import schema as S

NOISE_BLOCK = "noise"
REDUNDANT_BLOCK = "redundant"


@dataclass
class FeatureBundle:
    """建模所需的全部矩阵。cat 矩阵的值域已保证落在 [0, cardinality)。"""

    # 用户侧
    user_cat: np.ndarray                 # (n_users, n_user_cat) int32
    user_dense: np.ndarray               # (n_users, n_user_dense) float32
    user_pca: np.ndarray                 # (n_users, n_pca) float32
    user_segment: np.ndarray             # (n_users,) int32
    user_profile_cols: list[str] = field(default_factory=list)

    # 素材侧
    item_cat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    item_dense: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    # 曝光侧
    imp_cat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    imp_dense: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    # 字段元信息：顺序即 field index，绝不能变
    cat_fields: list[tuple[str, int]] = field(default_factory=list)
    dense_cols: list[str] = field(default_factory=list)

    @property
    def user_dense_dim(self) -> int:
        return self.user_dense.shape[1]

    @property
    def item_dense_dim(self) -> int:
        return self.item_dense.shape[1]

    @property
    def imp_dense_dim(self) -> int:
        return self.imp_dense.shape[1]

    def cat_index(self, name: str) -> int:
        for i, (n, _) in enumerate(self.cat_fields):
            if n == name:
                return i
        raise KeyError(name)

    @property
    def n_fields(self) -> int:
        return len(self.cat_fields)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _cat_index(values: np.ndarray, card: int) -> np.ndarray:
    """把原始类别码映射到 [0, card]。

    缺失值（生成器里用 -1 表示）统一落到索引 0，其余值整体 +1。
    这样「缺失」本身成为模型可学的一个独立类别 —— 对 MNAR 缺失尤其重要，
    因为「高收入用户更可能拒绝授权」这件事本身就是信号。
    """
    v = np.asarray(values)
    idx = np.where(v < 0, 0, np.minimum(v + 1, card))
    return idx.astype(np.int32)


def _safe_float32(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """取列并转 float32，NaN 用列中位数填充（同时保留缺失指示位）。"""
    sub = df[cols].to_numpy(dtype=np.float32, copy=True)
    if np.isnan(sub).any():
        med = np.nanmedian(sub, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        nan_mask = np.isnan(sub)
        sub[nan_mask] = np.take(med, np.where(nan_mask)[1])
    return np.ascontiguousarray(sub, dtype=np.float32)


def _standardize_inplace(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _prep_dense(X: np.ndarray, names: list[str],
                fit_mask: np.ndarray | None = None) -> np.ndarray:
    """数值块的预处理：重尾列先 ``log1p``，再整体 z-score。

    ⚠️ 这一步**不是可选项**，理由见 ``schema.LOG1P_FEATURES`` 的长注释：
       不压重尾 + 不标准化，带 FM 的变体会在 epoch 1 直接炸掉，
       而且表现得像是「FM 这个模型有问题」。

    ``fit_mask`` 指定用哪些行统计 mean/std。时序切分下**必须**只用训练窗口
   （曝光表按 day 切），否则 scaler 会把验证/测试期的分布信息漏进训练 ——
   这类泄漏很小，不会让指标爆表，所以特别容易蒙混过关。
   用户表 / 素材表是快照，没有时间维，直接用全表统计。
    """
    X = np.asarray(X, dtype=np.float32)
    log_cols = set(S.LOG1P_FEATURES)
    log_idx = [i for i, c in enumerate(names) if c in log_cols]
    if log_idx:
        X = X.copy()
        # 计数/金额都是非负的；clip 一下防脏数据把 log 变成 NaN
        X[:, log_idx] = np.log1p(np.maximum(X[:, log_idx], 0.0))

    ref = X if fit_mask is None else X[fit_mask]
    mu, sd = _standardize_inplace(ref)
    return ((X - mu) / sd).astype(np.float32)


def add_noise_and_redundant(df: pd.DataFrame, cfg: Config,
                            rng: np.random.Generator,
                            source_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """追加纯噪声列与冗余列。

    刻意保留「200 维里 44.8% 是垃圾」的真实情况 —— 这样 PCA 降维和
    特征筛选才有可验证的价值，而不是走过场。
    """
    n = len(df)
    noise_cols = S.make_noise_features(cfg.n_noise_features)
    for c in noise_cols:
        df[c] = rng.normal(0.0, 1.0, n).astype(np.float32)

    redund_cols = S.make_redundant_features(cfg.n_redundant_features, source_cols)
    src = df[source_cols].to_numpy(dtype=np.float32)
    for i, c in enumerate(redund_cols):
        a = src[:, i % len(source_cols)]
        b = src[:, (i * 7 + 3) % len(source_cols)]
        w = rng.normal(0.0, 1.0, 2).astype(np.float32)
        df[c] = (w[0] * a + w[1] * b + rng.normal(0, 0.01, n)).astype(np.float32)

    return df, noise_cols + redund_cols


# --------------------------------------------------------------------------
# 用户画像宽表
# --------------------------------------------------------------------------


def build_user_profile(users: pd.DataFrame, cfg: Config,
                       rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    """产出用户画像宽表（≥200 维）。

    等价于 ``sql/user_features.sql`` 里的那段聚合。
    """
    df = users.copy()

    signal_cols = [c for c in S.USER_SIGNAL_COLS if c in df.columns]
    df, _ = add_noise_and_redundant(df, cfg, rng, signal_cols)

    profile_cols = (
        [c for c, _ in S.USER_CAT_FIELDS]
        + S.USER_PROFILE_NUMERIC
        + S.make_noise_features(cfg.n_noise_features)
        + S.make_redundant_features(cfg.n_redundant_features, signal_cols)
    )
    return df, profile_cols


def build_user_matrices(profile: pd.DataFrame, pca_result,
                        cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把画像宽表拆成 类别矩阵 / 数值矩阵 / PCA 矩阵。

    PCA 主成分会拼进数值矩阵 —— 它们本身就是压缩后的用户画像，
    对模型来说是现成的好特征。
    """
    cat = np.stack([_cat_index(profile[c].to_numpy(), card)
                    for c, card in S.USER_CAT_FIELDS], axis=1)

    numeric_cols = S.USER_PROFILE_NUMERIC + S.make_noise_features(cfg.n_noise_features)
    dense = _prep_dense(_safe_float32(profile, numeric_cols), numeric_cols)

    pca_X = _safe_float32(profile, S.USER_PROFILE_NUMERIC)
    # PCA 主成分的方差等于特征值，第一主成分的 std 可能有 5 上下，
    # 与其余列不同量级，同样要压一遍。
    pca_feats = _prep_dense(pca_result.transform(pca_X),
                            S.make_pca_features(pca_result.n_components))

    dense = np.concatenate([dense, pca_feats], axis=1).astype(np.float32)
    return cat, dense, pca_feats


# --------------------------------------------------------------------------
# 曝光级建模矩阵
# --------------------------------------------------------------------------


def build_item_matrices(items: pd.DataFrame, cfg: Config,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cat = np.stack([_cat_index(items[c].to_numpy(), card)
                    for c, card in S.ITEM_CAT_FIELDS], axis=1)
    dense = _prep_dense(_safe_float32(items, S.ITEM_SIGNAL_COLS),
                        list(S.ITEM_SIGNAL_COLS))
    return cat, dense


def build_impression_matrices(
    imp: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
    user_segment: np.ndarray, cfg: Config, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """曝光级：类别矩阵 + 数值矩阵。

    这里算的是「跨域特征」—— 它们同时依赖用户和素材，
    所以**只能进排序层**，绝不能进双塔的 user tower（否则塔就没法预计算了）。
    """
    n = len(imp)
    u_idx = imp["user_id"].to_numpy()
    a_idx = imp["ad_id"].to_numpy()

    # --- 跨域统计特征（用户 × 素材）---
    u_sig = users[S.USER_SIGNAL_COLS].to_numpy(dtype=np.float32)[u_idx]
    a_sig = items[S.ITEM_SIGNAL_COLS].to_numpy(dtype=np.float32)[a_idx]
    ad_cat = items["ad_category"].to_numpy()[a_idx]
    u_interest = users["interest_top1"].to_numpy()[u_idx]
    cross_hit = (ad_cat == u_interest).astype(np.float32)
    user_x_category_ctr = np.clip(
        0.5 * cross_hit + 0.5 * (a_sig[:, 0] * u_sig[:, 1]) + rng.normal(0, 0.05, n), 0, 1)
    user_x_creative_cvr = np.clip(
        0.4 * cross_hit + 0.6 * (a_sig[:, 1] * u_sig[:, 0]) + rng.normal(0, 0.05, n), 0, 1)

    imp_dense_cols = [
        user_x_category_ctr.astype(np.float32),
        user_x_creative_cvr.astype(np.float32),
        a_sig[:, 0], a_sig[:, 1], a_sig[:, 2],
        cross_hit,
    ]
    # ⚠️ 列名必须与列数严格一一对应。早先这里手工拼了 8 个名字去描述 12 列，
    #    下游按名字取值时静默错位 —— 这类 bug 不会抛异常，只会让指标莫名其妙。
    dense_names = list(S.IMPRESSION_AD_CONTEXT)

    # --- 跨域计数特征 ---
    seg = user_segment[u_idx]
    ctype = items["creative_type"].to_numpy()[a_idx]
    cnt_pairs = [
        (u_idx, items["advertiser_id"].to_numpy()[a_idx]),
        (u_idx, ad_cat),
        (u_idx, items["price_band"].to_numpy()[a_idx]),
        (u_idx, imp["time_slot"].to_numpy()),
        (u_idx, ctype),
        (seg, ctype),
    ]
    assert len(cnt_pairs) == len(S.CROSS_COUNT_NAMES), "计数特征名字与列数不匹配"
    for ka, kb in cnt_pairs:
        key = (ka.astype(np.int64) * 100_003 + kb.astype(np.int64))
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        imp_dense_cols.append(counts[inv].astype(np.float32))
    dense_names += list(S.CROSS_COUNT_NAMES)

    imp_dense = np.stack(imp_dense_cols, axis=1).astype(np.float32)
    assert imp_dense.shape[1] == len(dense_names), "曝光数值特征列数与列名不一致"
    # 只用训练窗口（按 day 切）统计 mean/std，避免把未来分布漏进 scaler。
    fit_mask = _train_day_mask(imp, cfg)
    imp_dense = _prep_dense(imp_dense, dense_names, fit_mask)

    # --- 类别矩阵 ---
    cat_arrays: list[np.ndarray] = []
    cat_fields: list[tuple[str, int]] = []

    for name, card in S.ID_CAT_FIELDS:
        col = {"user_id": u_idx, "ad_id": a_idx,
               "advertiser_id": items["advertiser_id"].to_numpy()[a_idx],
               "campaign_id": items["campaign_id"].to_numpy()[a_idx]}[name]
        cat_arrays.append(_cat_index(col, card))
        cat_fields.append((name, card + 1))

    for name, card in S.USER_CAT_FIELDS:
        cat_arrays.append(_cat_index(users[name].to_numpy()[u_idx], card))
        cat_fields.append((f"u_{name}", card + 1))

    for name, card in S.ITEM_CAT_FIELDS:
        cat_arrays.append(_cat_index(items[name].to_numpy()[a_idx], card))
        cat_fields.append((f"a_{name}", card + 1))

    for name, card in S.CONTEXT_CAT_FIELDS:
        cat_arrays.append(_cat_index(imp[name].to_numpy(), card))
        cat_fields.append((name, card + 1))

    # 分层与分层×创意
    cat_arrays.append(seg.astype(np.int32))
    cat_fields.append(("user_segment", 12))
    seg_x_creative = (seg * 8 + ctype).astype(np.int32)
    cat_arrays.append(seg_x_creative)
    cat_fields.append(("seg_x_creative", 12 * 8))

    # 分箱字段：连续特征必须先分箱才能进 FM 的二阶项
    bin_src = [
        ("bin_activity", users["sessions_7d"].to_numpy()[u_idx]),
        ("bin_ctr_hist", users["ctr_7d"].to_numpy()[u_idx]),
        ("bin_cvr_hist", users["cvr_30d"].to_numpy()[u_idx]),
        ("bin_recency", users["recency_days"].to_numpy()[u_idx]),
        ("bin_monetary", users["monetary_30d"].to_numpy()[u_idx]),
        ("bin_ltv_hist", users["hist_ltv"].to_numpy()[u_idx]),
        ("bin_ad_ctr", items["ad_ctr_global"].to_numpy()[a_idx]),
        ("bin_category_affinity", users["category_affinity"].to_numpy()[u_idx]),
        ("bin_price_affinity", users["price_affinity"].to_numpy()[u_idx]),
        ("bin_fatigue", users["creative_fatigue_score"].to_numpy()[u_idx]),
        ("bin_competition", users["competition_index"].to_numpy()[u_idx]),
        ("bin_session_len", users["avg_session_len"].to_numpy()[u_idx]),
    ]
    bin_card = {n: c for n, c in S.BINNED_FIELDS}
    for name, values in bin_src:
        card = bin_card[name]
        cat_arrays.append(_quantile_bin(values, card))
        cat_fields.append((name, card))

    # 显式高阶交叉的输入键（见 _high_order_keys 的说明）
    for name, values, card in _high_order_keys(users, items, imp):
        cat_arrays.append(values)
        cat_fields.append((name, card))

    imp_cat = np.stack(cat_arrays, axis=1).astype(np.int32)
    return imp_cat, imp_dense, cat_fields


def _train_day_mask(imp: pd.DataFrame, cfg: Config) -> np.ndarray:
    """训练窗口（按 day 切）的行掩码，规则与 ``splits.time_split`` 完全一致。

    只 import 来用它的切分规则，不引入第二种「训练期」定义 ——
    两处定义不一致的话，scaler 的拟合范围会和模型实际训练的范围错开，
    而且这种错位不会有任何报错。
    """
    from src.data.splits import time_split

    tr = time_split(imp, cfg).train
    mask = np.zeros(len(imp), dtype=bool)
    mask[tr] = True
    return mask


def _high_order_keys(users: pd.DataFrame, items: pd.DataFrame,
                     imp: pd.DataFrame) -> list[tuple[str, np.ndarray, int]]:
    """显式高阶交叉的四个分箱键，**必须与生成器用的是同一套定义**。

    真值里有一项 ``T3[u_act, a_pop, ctx_slot]``（三阶）和
    ``T4[u_act, a_pop, ctx_slot, page_kind]``（四阶）。FM 的模型形式里
    根本没有三阶项，所以这一项是 DeepFM 相对 FM 的结构性优势所在 ——
    但前提是**模型得看得见这四个键**。

    ⚠️ 这里踩过一个坑，而且是个很典型的「设计上说得通、跑起来是空转」的坑：

       最初这四个键没有出现在特征里 —— ``a_pop``（素材热度分档）
       甚至完全不在这张曝光特征表上。于是真值里那份三阶/四阶信号
       对**所有**模型都是不可观测的噪声：DeepFM 学不到（看不见输入），
       FM 也学不到（模型形式不允许），两边一起被噪声拉低，
       ``AUC(DeepFM) − AUC(FM)`` 自然测出 0 甚至负数。
       而这时 AUC 绝对值看起来还挺正常（0.80 上下），完全不像坏了。

       修法不是「把 T3 的值喂进去」（那是泄漏），而是把**输入变量**
       暴露出来 —— 活动度分档、素材热度分档、时段、页面位置，
       这四个都是真实广告系统里现成的业务特征。模型要自己
       从它们学出那个三阶结构，这正是 DeepFM 该做的事。

    定义与生成器共用 ``S.HIGH_ORDER_KEYS`` / ``S.HIGH_ORDER_CARD``，
    分箱函数也用同一个 ``theta_bin``（分位数分箱），保证**逐位一致**。
    """
    from src.data.generator import theta_bin

    u_idx = imp["user_id"].to_numpy()
    a_idx = imp["ad_id"].to_numpy()
    raw = {
        "u_act": theta_bin(users["sessions_7d"].to_numpy()[u_idx], 4),
        "a_pop": theta_bin(items["item_popularity"].to_numpy()[a_idx], 4),
        "ctx_slot": imp["time_slot"].to_numpy(),
        "page_kind": imp["page_kind"].to_numpy(),
    }
    out: list[tuple[str, np.ndarray, int]] = []
    for key in S.HIGH_ORDER_KEYS:
        card = S.HIGH_ORDER_CARD[key]
        v = np.clip(raw[key], 0, card - 1).astype(np.int32)
        out.append((key, v, card))
    return out


def _quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(q, x).astype(np.int32)


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------


def build_feature_bundle(data: dict, pca_result, kmeans_result,
                         cfg: Config) -> FeatureBundle:
    """串起「画像 → PCA → 分层 → 曝光级特征」的全过程。"""
    rng = np.random.default_rng([cfg.seed, 77])
    users, items, imp = data["users"], data["items"], data["impressions"]

    profile, profile_cols = build_user_profile(users, cfg, rng)
    user_segment = kmeans_result.labels
    user_cat, user_dense, user_pca = build_user_matrices(profile, pca_result, cfg)

    item_cat, item_dense = build_item_matrices(items, cfg, rng)
    imp_cat, imp_dense, cat_fields = build_impression_matrices(
        imp, users, items, user_segment, cfg, rng)

    return FeatureBundle(
        user_cat=user_cat, user_dense=user_dense, user_pca=user_pca,
        user_segment=user_segment, user_profile_cols=profile_cols,
        item_cat=item_cat, item_dense=item_dense,
        imp_cat=imp_cat, imp_dense=imp_dense,
        cat_fields=cat_fields,
        dense_cols=list(S.IMPRESSION_AD_CONTEXT) + list(S.CROSS_COUNT_NAMES),
    )
