"""模拟数据生成器 —— 全项目的地基。

设计要点
--------
**为什么用 Random Fourier Features (RFF) 造真值？**

    f(x) = sqrt(2/D) * Σ_d a_d * cos(ω_d^T x + b_d),   ω_d ~ N(0, σ_ω² I)

它是 RBF 核高斯过程的有限秩近似 —— 函数光滑、有界、处处可学。
关键在于 ``cos(ω^T x)`` 的泰勒展开含**所有阶**的项，而 ``σ_ω`` 直接控制
高频能量占比，也就是**高阶交叉能量**。于是：

    LR  只能拟合 f 的线性部分
    FM  能拟合二阶部分（也只有二阶）
    DNN 能拟合剩余的高阶部分

σ_ω 就是控制 ``AUC(DeepFM) − AUC(FM)`` 的那个旋钮。这是本方案最核心的洞察 ——
「DeepFM 打赢基线」是**设计出来的**，不是碰运气。

**二阶之外还得有显式的三阶/四阶项。**
注意：emb_dim=8 时 FM 其实**能**表达 2×2 的 XOR（那是 rank-2，8 维嵌入装得下）。
真正让 FM 无能为力的是**三阶及以上** —— FM 的模型形式里根本没有这一项。
所以这里用随机张量 T3 / T4 显式注入三阶、四阶交叉，它们都是**可观测分箱
特征的函数**，因此 DNN 学得到、FM 原理上学不到。

**可观测特征必须是隐向量的含噪函数，绝不能是隐向量本身。**
直接放隐向量的话模型三行代码就能到 AUC 0.95，整个项目就没意义了。

工程注意
--------
所有列先在 dict 里攒齐、最后 ``pd.concat`` 一次成表。
逐列 ``df[col] = ...`` 会产生大量 DataFrame 碎片，
10 万用户 × 90 列时会明显变慢并刷屏 PerformanceWarning。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import Config
from src.data import schema as S

# --------------------------------------------------------------------------
# RFF
# --------------------------------------------------------------------------


def make_rff(d_in: int, n_basis: int, sigma_w: float,
             rng: np.random.Generator):
    """构造一组 RFF 基：W ~ N(0, σ_ω²)、b ~ U(0,2π)、a ~ N(0,1)。"""
    W = rng.normal(0.0, sigma_w, size=(n_basis, d_in)).astype(np.float32)
    b = rng.uniform(0.0, 2 * np.pi, size=n_basis).astype(np.float32)
    a = rng.normal(0.0, 1.0, size=n_basis).astype(np.float32)
    return W, b, a


def rff_apply(X: np.ndarray, W: np.ndarray, b: np.ndarray, a: np.ndarray,
              chunk: int = 100_000) -> np.ndarray:
    """分块计算 RFF，避免一次开出 (n, n_basis) 的大矩阵。

    2M 行 × 256 基 × float32 = 2GB，不切块会直接吃爆内存。
    """
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    scale = math.sqrt(2.0 / W.shape[0])
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        out[sl] = (np.cos(X[sl] @ W.T + b) * a).sum(axis=1) * scale
    return out


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60, 60)))


def _sigmoid_mean(logit: np.ndarray, shift: float) -> float:
    """mean(sigmoid(logit + shift))。

    ⚠️ 括号必须包住整个 sigmoid。写成 ``1.0 / (1.0 + np.exp(-z)).mean()``
    会算成 ``1/mean(...)`` 而不是 ``mean(1/...)``，校准结果完全跑偏 ——
    实测会让 CTR 从 0.05 变成 0.995，而且不报任何错。
    """
    return float(_sigmoid(np.asarray(logit, dtype=np.float64) + shift).mean())


def _bisect_intercept(logit: np.ndarray, target_rate: float,
                      lo: float = -30.0, hi: float = 30.0, iters: int = 60) -> float:
    """二分法求截距平移 δ，使 ``mean(sigmoid(logit + δ)) == target_rate``。

    固定截距会让基率落在任意位置，导致训练 loss 失衡。二分 60 次精度足够，
    一次写对永不返工。
    """
    logit = np.asarray(logit, dtype=np.float64)
    if _sigmoid_mean(logit, lo) > target_rate:
        return lo
    if _sigmoid_mean(logit, hi) < target_rate:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _sigmoid_mean(logit, mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def theta_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """按分位数分箱，返回 0..n_bins-1。"""
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(q, x)


# --------------------------------------------------------------------------
# 生成器
# --------------------------------------------------------------------------


class DataGenerator:
    """按配置生成用户 / 素材 / 曝光点击转化全链路模拟数据。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng([cfg.seed, 1])
        self._rff: dict = {}
        self._tensors: dict = {}

    # ------------------------------------------------------------------
    # 用户
    # ------------------------------------------------------------------
    def generate_users(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n, d = cfg.n_users, cfg.latent_dim

        # --- 真实分层：每层一个隐向量中心，KMeans 应该能恢复出来 ---
        n_seg = cfg.n_true_segments
        seg_prob = np.array([0.30, 0.25, 0.20, 0.15, 0.10][:n_seg], dtype=np.float64)
        seg_prob /= seg_prob.sum()
        centers = rng.normal(0.0, 1.6, size=(n_seg, d)).astype(np.float32)
        seg = rng.choice(n_seg, size=n, p=seg_prob)
        # 层内散度 0.6 → 簇之间分得开但不平凡
        z = (centers[seg] + rng.normal(0.0, 0.6, size=(n, d))).astype(np.float32)

        # 时间漂移：只对一部分用户加极慢的随机游走，让测试集 AUC 比验证集
        # 低 0.5~1.5 个点 —— 这正是「模型为什么要定期重训」的原因。
        drift_mask = rng.random(n) < 0.3
        z = np.where(drift_mask[:, None],
                     z + rng.normal(0.0, 0.12, size=(n, d)).astype(np.float32), z)

        cols: dict[str, np.ndarray] = {"user_id": np.arange(n, dtype=np.int64)}

        # --- 用户侧类别域：隐向量的随机投影 + 分位数切桶 ---
        # 每个域用不同的投影方向，于是不同域带的信息互补。
        user_cat_specs = [
            ("gender", 3, 0.5, 0), ("age_bucket", 7, 1.0, 0), ("city_tier", 5, 0.8, 1),
            ("province", 34, 1.2, 0), ("device_brand", 20, 1.0, 1),
            ("device_price_band", 5, 0.9, 0), ("os", 4, 0.6, 0),
            ("network_type", 4, 0.5, 0), ("carrier", 4, 0.5, 0),
            ("app_category_pref", 15, 1.1, 1), ("income_bucket", 6, 1.2, 1),
            ("marital", 3, 0.6, 1), ("education", 5, 0.8, 1),
            ("occupation", 12, 1.0, 1), ("interest_top1", 30, 1.3, 0),
            ("interest_top2", 30, 1.3, 1), ("os_version", 12, 0.8, 0),
            ("session_len_bucket", 6, 0.9, 0), ("purchase_power_tier", 5, 1.3, 0),
            ("recency_bucket", 6, 1.0, 0), ("frequency_bucket", 6, 1.0, 0),
            ("is_new_user", 2, 0.7, 0),
        ]
        for name, card, strength, _ in user_cat_specs:
            proj = rng.normal(0.0, strength, size=d).astype(np.float32)
            raw = z @ proj + rng.normal(0.0, 0.8, n)
            q = np.quantile(raw, np.linspace(0, 1, card + 1)[1:-1])
            cols[name] = np.searchsorted(q, raw).astype(np.int32)

        cols.update(self._numeric_columns(z, rng))
        cols.update(self._ltv_columns(z, seg, cols, rng))
        cols.update(self._apply_missingness(cols, n, rng))

        for k in range(d):
            cols[f"_true_z{k}"] = z[:, k]
        cols["_true_segment"] = seg.astype(np.int32)

        return pd.DataFrame(cols)

    # ------------------------------------------------------------------
    def _numeric_columns(self, z: np.ndarray, rng: np.random.Generator) -> dict:
        """行为 / 货币 / 用户侧广告 / 统计聚合 四组数值特征。

        每列都是 ``u·z + 噪声``，SNR 刻意压到 0.4 上下 ——
        太低模型学不到东西，太高就退化成「隐向量直接泄漏」。
        """
        n, d = z.shape
        out: dict[str, np.ndarray] = {}

        def proj(strength: float, noise: float) -> np.ndarray:
            u = rng.normal(0.0, 1.0, size=d).astype(np.float32)
            u /= np.linalg.norm(u) + 1e-8
            return (z @ u) * strength + rng.normal(0.0, noise, size=n)

        f32 = np.float32
        # ---- 行为统计（30 列）----
        out["sessions_7d"] = np.maximum(0, rng.poisson(6, n) + proj(2.5, 2.0)).astype(f32)
        out["sessions_30d"] = np.maximum(0, rng.poisson(22, n) + proj(7.0, 6.0)).astype(f32)
        out["active_days_30d"] = np.clip(rng.poisson(12, n) + proj(3.0, 3.0), 0, 30).astype(f32)
        out["avg_session_len"] = np.maximum(0.1, rng.gamma(3.0, 40.0, n) + proj(25.0, 30.0)).astype(f32)
        out["page_views_7d"] = np.maximum(0, rng.poisson(30, n) + proj(11.0, 9.0)).astype(f32)
        out["page_views_30d"] = np.maximum(0, rng.poisson(120, n) + proj(42.0, 35.0)).astype(f32)
        out["click_cnt_7d"] = np.maximum(0, rng.poisson(12, n) + proj(4.5, 4.0)).astype(f32)
        out["click_cnt_30d"] = np.maximum(0, rng.poisson(48, n) + proj(17.0, 15.0)).astype(f32)
        out["impression_cnt_7d"] = np.maximum(0, rng.poisson(200, n) + proj(60.0, 55.0)).astype(f32)
        out["impression_cnt_30d"] = np.maximum(0, rng.poisson(800, n) + proj(230.0, 210.0)).astype(f32)
        out["ctr_7d"] = np.clip(_sigmoid(proj(1.1, 0.85)) * 0.25, 0, 1).astype(f32)
        out["ctr_30d"] = np.clip(_sigmoid(proj(1.0, 0.85)) * 0.23, 0, 1).astype(f32)
        out["cvr_7d"] = np.clip(_sigmoid(proj(0.9, 0.95)) * 0.20, 0, 1).astype(f32)
        out["cvr_30d"] = np.clip(_sigmoid(proj(0.85, 0.95)) * 0.18, 0, 1).astype(f32)
        out["search_cnt_7d"] = np.maximum(0, rng.poisson(5, n) + proj(2.0, 2.5)).astype(f32)
        out["collect_cnt_30d"] = np.maximum(0, rng.poisson(8, n) + proj(3.0, 3.5)).astype(f32)
        out["share_cnt_30d"] = np.maximum(0, rng.poisson(2, n) + proj(1.0, 1.5)).astype(f32)
        out["comment_cnt_30d"] = np.maximum(0, rng.poisson(4, n) + proj(1.5, 2.0)).astype(f32)
        out["follow_cnt_30d"] = np.maximum(0, rng.poisson(3, n) + proj(1.2, 1.8)).astype(f32)
        out["video_view_cnt_7d"] = np.maximum(0, rng.poisson(40, n) + proj(14.0, 13.0)).astype(f32)
        out["ad_click_cnt_7d"] = np.maximum(0, rng.poisson(4, n) + proj(1.6, 1.6)).astype(f32)
        out["ad_click_cnt_30d"] = np.maximum(0, rng.poisson(16, n) + proj(6.0, 6.0)).astype(f32)
        out["ad_ctr_7d"] = np.clip(_sigmoid(proj(0.95, 0.9)) * 0.15, 0, 1).astype(f32)
        out["ad_ctr_30d"] = np.clip(_sigmoid(proj(0.9, 0.9)) * 0.14, 0, 1).astype(f32)
        out["recency_days"] = np.clip(rng.exponential(6.0, n) + proj(1.5, 3.0), 0, 90).astype(f32)
        out["frequency_7d"] = np.maximum(0, rng.poisson(9, n) + proj(3.0, 3.0)).astype(f32)
        out["monetary_30d"] = np.maximum(0.0, rng.lognormal(3.0, 1.1, n) + proj(18.0, 25.0)).astype(f32)
        out["interaction_depth"] = np.clip(_sigmoid(proj(1.0, 0.9)) * 10, 0, 10).astype(f32)
        out["night_owl_score"] = np.clip(_sigmoid(proj(0.8, 1.0)), 0, 1).astype(f32)
        out["weekend_ratio"] = np.clip(_sigmoid(proj(0.6, 1.1)), 0, 1).astype(f32)

        # ---- 货币类（8 列）----
        out["hist_ltv"] = np.maximum(0.0, rng.lognormal(2.5, 1.4, n) + proj(22.0, 30.0)).astype(f32)
        out["arpu_30d"] = np.maximum(0.0, rng.lognormal(2.0, 1.3, n) + proj(9.0, 12.0)).astype(f32)
        out["gmv_mean"] = np.maximum(0.0, rng.lognormal(3.2, 1.2, n) + proj(24.0, 32.0)).astype(f32)
        out["gmv_max"] = np.maximum(0.0, rng.lognormal(4.0, 1.4, n) + proj(60.0, 80.0)).astype(f32)
        out["order_cnt_30d"] = np.maximum(0, rng.poisson(3, n) + proj(1.2, 1.6)).astype(f32)
        out["avg_order_amount"] = np.maximum(0.0, rng.lognormal(3.4, 1.0, n) + proj(20.0, 26.0)).astype(f32)
        out["coupon_usage_rate"] = np.clip(_sigmoid(proj(0.8, 1.0)), 0, 1).astype(f32)
        out["refund_rate"] = np.clip(_sigmoid(proj(0.7, 1.1)) * 0.3, 0, 1).astype(f32)

        # ---- 用户侧广告统计（6 列；曝光侧的另 6 列在曝光层算）----
        out["advertiser_quality"] = np.clip(_sigmoid(proj(0.5, 1.2)) * 5, 0, 5).astype(f32)
        out["category_affinity"] = np.clip(_sigmoid(proj(1.0, 0.9)), 0, 1).astype(f32)
        out["price_affinity"] = np.clip(_sigmoid(proj(1.0, 0.9)), 0, 1).astype(f32)
        out["creative_fatigue_score"] = np.clip(_sigmoid(proj(0.9, 1.0)), 0, 1).astype(f32)
        out["competition_index"] = np.clip(_sigmoid(proj(0.6, 1.2)) * 2, 0, 2).astype(f32)
        out["slot_traffic_share"] = np.clip(_sigmoid(proj(0.4, 1.3)), 0, 1).astype(f32)

        # ---- 统计聚合 / 派生比率（30 列）----
        # 形态各异：有界比率走 sigmoid，计数类取绝对值，得分类可正可负。
        for i, name in enumerate(S.STAT_FEATURES):
            raw = proj(0.9, 1.0)
            style = i % 3
            if style == 0:
                out[name] = np.clip(_sigmoid(raw), 0, 1).astype(f32)
            elif style == 1:
                out[name] = np.abs(raw * 8.0 + rng.normal(0, 3, n)).astype(f32)
            else:
                out[name] = raw.astype(f32)
        return out

    # ------------------------------------------------------------------
    def _ltv_columns(self, z: np.ndarray, seg: np.ndarray, cols: dict,
                     rng: np.random.Generator) -> dict:
        """零膨胀的付费金额 —— 让 ZILN 的对数正态假设成立。

        条件期望 ``E[LTV] = π · exp(μ + σ²/2)``，含零膨胀。
        eCPM 用的必须是这个量，只取 ``exp(μ)`` 会系统性低估。

        μ(x) 里为什么必须有**分层效应**：只用 ``z @ w`` 这种随机方向的话，
        当簇中心只有 5 个时，单条随机方向在簇上的投影方差极不稳定
        （实测某次 seed 下 var(μ) 只有 0.16，ZILN 的回归头等于在拟合常数）。
        显式的分层溢价让「高价值人群」这件事稳定存在、且与聚类结果对齐 ——
        这正是 ZILN 要发现的业务结构。
        """
        cfg = self.cfg
        n = z.shape[0]
        d = z.shape[1]
        n_seg = cfg.n_true_segments

        # 分层溢价。
        #
        # ⚠️ 「会不会付费」和「会付多少」必须由**不同的因素**驱动，这不是随便定的：
        #    若两者都由分层同向驱动，高金额的层必然同时也是高付费率的层，
        #    于是「已付费」这个条件本身就把 μ 的方差筛掉大半 —— 实测
        #    var(μ|付费) 只剩 0.12，而 σ²=0.72，ZILN 的回归头退化成拟合常数，
        #    却完全不报错。
        #
        #    所以：付费率主要由隐向量决定（分层只做轻微扰动），
        #    金额的层间梯度做得很陡。付费样本因此能覆盖到各层，
        #    var(μ|付费) 留得住；同时 ZILN 的两个阶段各管各的、互不冗余。
        seg_pay = np.array([-0.20, -0.10, 0.00, 0.15, 0.30][:n_seg],
                           dtype=np.float64)[seg]
        seg_amt = np.array([-1.10, -0.45, 0.20, 0.90, 1.70][:n_seg],
                           dtype=np.float64)[seg]

        w_pay = rng.normal(0.0, 1.0, size=d).astype(np.float32)
        w_pay /= np.linalg.norm(w_pay)
        pptr = _standardize(cols["purchase_power_tier"].astype(np.float64).reshape(-1, 1)).ravel()
        pay_logit = 0.9 * (z @ w_pay) + 0.4 * pptr + seg_pay
        shift = _bisect_intercept(pay_logit, 1.0 - cfg.zero_inflation)
        pi = _sigmoid(pay_logit + shift).astype(np.float32)
        pay_flag = (rng.random(n) < pi).astype(np.int32)

        w_amt = rng.normal(0.0, 1.0, size=d).astype(np.float32)
        w_amt /= np.linalg.norm(w_amt)
        mu = cfg.ltv_mu_log + 0.30 * (z @ w_amt) + 0.25 * pptr + seg_amt
        amount = np.clip(rng.lognormal(mu, cfg.ltv_sigma),
                         cfg.ltv_amt_min, cfg.ltv_amt_max).astype(np.float32)
        amount = np.where(pay_flag == 1, amount, 0.0).astype(np.float32)

        return {
            "pay_flag": pay_flag,
            "pay_amount": amount,
            "expected_ltv": (pi * np.exp(mu + cfg.ltv_sigma ** 2 / 2.0)).astype(np.float32),
            "_true_pay_prob": pi,
            "_true_log_mu": mu.astype(np.float32),
        }

    # ------------------------------------------------------------------
    def _apply_missingness(self, cols: dict, n: int,
                           rng: np.random.Generator) -> dict:
        """注入缺失：人口属性 MNAR，行为块 MCAR。

        MNAR 更真实 —— 高收入用户更可能拒绝授权，于是「缺失」本身就是信息，
        而且这个信息是模型能用的（``is_missing_*`` 本身就是特征）。
        生成器用 -1 表示类别缺失、NaN 表示数值缺失，
        ``features/pipeline._cat_index`` 会把 -1 映射成独立的一档。
        """
        out: dict[str, np.ndarray] = {}
        # 打码前的原始分档 —— 必须留一份，否则「缺失率随收入变化」这件事
        # 在打码之后就再也验证不了了（被测的那一列自己已经被抹掉了）。
        income_raw = cols["income_bucket"].copy()
        out["_true_income_bucket"] = income_raw.astype(np.int32)

        income_z = _standardize(income_raw.astype(np.float64).reshape(-1, 1)).ravel()
        p_mnar = _sigmoid(2.0 + 0.8 * income_z)
        out["_true_p_missing_income"] = (p_mnar * 0.25).astype(np.float32)

        mnar = {
            "income_bucket": "is_missing_income",
            "city_tier": "is_missing_city",
            "occupation": "is_missing_occupation",
            "education": "is_missing_education",
            "marital": "is_missing_marital",
        }
        for col, flag in mnar.items():
            miss = rng.random(n) < p_mnar * 0.25
            cols[col] = np.where(miss, -1, cols[col]).astype(np.int32)
            out[flag] = miss.astype(np.int32)

        mcar = {
            "device_brand": "is_missing_device_brand",
            "app_category_pref": "is_missing_app_category_pref",
            "interest_top2": "is_missing_interest_top2",
        }
        for col, flag in mcar.items():
            miss = rng.random(n) < 0.05
            cols[col] = np.where(miss, -1, cols[col]).astype(np.int32)
            out[flag] = miss.astype(np.int32)

        for col, flag in [("price_affinity", "is_missing_price_affinity"),
                          ("coupon_usage_rate", "is_missing_coupon_usage_rate")]:
            miss = rng.random(n) < 0.06
            cols[col] = np.where(miss, np.nan, cols[col]).astype(np.float32)
            out[flag] = miss.astype(np.int32)
        return out

    # ------------------------------------------------------------------
    # 素材
    # ------------------------------------------------------------------
    def generate_items(self, n_categories: int = 25) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n, d = cfg.n_items, cfg.latent_dim
        v = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)

        cols: dict[str, np.ndarray] = {"ad_id": np.arange(n, dtype=np.int64)}
        cols["ad_category"] = rng.integers(0, n_categories, n).astype(np.int32)
        cols["creative_type"] = rng.integers(0, 8, n).astype(np.int32)
        cols["creative_format"] = rng.integers(0, 6, n).astype(np.int32)
        cols["advertiser_id"] = rng.integers(0, max(2, min(300, n // 4)), n).astype(np.int32)
        cols["campaign_id"] = rng.integers(0, max(2, min(1000, n // 2)), n).astype(np.int32)
        cols["landing_page_type"] = rng.integers(0, 10, n).astype(np.int32)
        cols["price_band"] = rng.integers(0, 8, n).astype(np.int32)
        cols["color_tone"] = rng.integers(0, 6, n).astype(np.int32)
        cols["has_video"] = (rng.random(n) < 0.35).astype(np.int32)

        f32 = np.float32
        cols["ad_ctr_global"] = np.clip(
            _sigmoid((v @ rng.normal(0, 1, d).astype(f32)) / math.sqrt(d)
                     + rng.normal(0, 0.8, n)) * 0.18, 0, 1).astype(f32)
        cols["ad_cvr_global"] = np.clip(
            _sigmoid((v @ rng.normal(0, 1, d).astype(f32)) / math.sqrt(d)
                     + rng.normal(0, 0.8, n)) * 0.12, 0, 1).astype(f32)
        cols["bid_floor_cpm"] = np.clip(rng.lognormal(math.log(12.0), 0.45, n), 1.0, 200.0).astype(f32)
        cols["item_popularity"] = np.clip(rng.lognormal(0.0, 0.8, n), 0.02, None).astype(f32)

        for k in range(d):
            cols[f"_true_v{k}"] = v[:, k]
        return pd.DataFrame(cols)

    # ------------------------------------------------------------------
    # 曝光 / 点击 / 转化
    # ------------------------------------------------------------------
    def generate(self) -> dict:
        users = self.generate_users()
        items = self.generate_items()
        imp = self.generate_impressions(users, items)
        return {"users": users, "items": items, "impressions": imp,
                "config": self.cfg}

    def generate_impressions(self, users: pd.DataFrame,
                             items: pd.DataFrame) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n_imp = min(cfg.n_users * cfg.n_days * cfg.impressions_per_user_day,
                    cfg.max_impressions)

        u_idx = rng.integers(0, len(users), n_imp)
        a_idx = rng.integers(0, len(items), n_imp)
        day = rng.integers(0, cfg.n_days, n_imp)
        slot = rng.integers(0, cfg.t_slots, n_imp)
        hour = np.clip(slot // (cfg.t_slots // 24), 0, 23).astype(np.int32)

        imp = pd.DataFrame({
            "user_id": u_idx.astype(np.int64),
            "ad_id": a_idx.astype(np.int64),
            "day": day.astype(np.int32),
            "slot": slot.astype(np.int32),
            "hour": hour,
            "dow": (day % 7).astype(np.int32),
            "time_slot": (hour // 6).astype(np.int32),
            "position_in_page": rng.integers(0, 10, n_imp).astype(np.int32),
            "page_type": rng.integers(0, 8, n_imp).astype(np.int32),
        })
        imp["page_kind"] = (imp["position_in_page"] // 3).astype(np.int32)

        # ---- 信号向量：全部取自**可观测**特征（隐向量的含噪函数）----
        U = _standardize(users[S.USER_SIGNAL_COLS].to_numpy(dtype=np.float32)[u_idx])
        A = _standardize(items[S.ITEM_SIGNAL_COLS].to_numpy(dtype=np.float32)[a_idx])
        ctx_oh = np.zeros((n_imp, 4), dtype=np.float32)
        ctx_oh[np.arange(n_imp), imp["time_slot"].to_numpy()] = 1.0
        x = _standardize(np.concatenate([U, A, ctx_oh], axis=1))

        u_act = np.clip(theta_bin(users["sessions_7d"].to_numpy()[u_idx], 4), 0, 3)
        a_pop = np.clip(theta_bin(items["item_popularity"].to_numpy()[a_idx], 4), 0, 3)
        page_kind = np.clip(imp["page_kind"].to_numpy(), 0, 3)

        logit_ctr, logit_cvr = self._true_logits(
            x, U, A, u_act, a_pop, imp["time_slot"].to_numpy(), page_kind,
            imp["day"].to_numpy(),
            users["_true_segment"].to_numpy()[u_idx],
            items["creative_type"].to_numpy()[a_idx])

        # ---- 基率校准 ----
        logit_ctr = (logit_ctr + _bisect_intercept(logit_ctr, cfg.target_ctr)).astype(np.float32)
        p_ctr = _sigmoid(logit_ctr).astype(np.float32)
        y_click = (rng.random(n_imp) < p_ctr).astype(np.int32)

        # ⚠️ CVR 的基率是 P(转化 | 点击)，**必须在点击子集上校准**。
        # 在全曝光空间校准会得到一个与 p_ctr 正相关的解，而点击样本天然
        # 偏向高 p_cvr 的人群 —— 实测偏差能到 8 倍（0.06 变 0.48），且不报错。
        clicked = y_click == 1
        b0_cvr = (_bisect_intercept(logit_cvr[clicked], cfg.target_cvr)
                  if clicked.sum() > 0 else 0.0)
        logit_cvr = (logit_cvr + b0_cvr).astype(np.float32)
        p_cvr = _sigmoid(logit_cvr).astype(np.float32)

        y_conv_true = (clicked & (rng.random(n_imp) < p_cvr)).astype(np.int32)

        # ---- 转化延迟 ----
        delay_h = self._sample_delay(n_imp, rng)
        # 观测时长 = 数据末尾时刻 − 曝光时刻，封顶在最长消融窗口。
        #
        # ⚠️ 这里踩过一个坑：早先写成 ``min(window_hours, (n_days - day) * 24)``，
        #    而 window_hours 恰好 = 24 = 一天，于是**所有行的 elapsed 都等于 24**，
        #    变成一个常数。DFM 的 elapsed 特征等于白给，且不报任何错。
        #    封顶到 obs_horizon(=168h) 之后，最后 7 天的观测时长从 1h 连续变到 168h。
        ts_h = (imp["day"].to_numpy(dtype=np.float64) * 24.0
                + imp["hour"].to_numpy(dtype=np.float64))
        elapsed_h = np.clip(cfg.n_days * 24.0 - ts_h, 0.5,
                            cfg.obs_horizon).astype(np.float32)

        imp["_true_logit_ctr"] = logit_ctr
        imp["_true_logit_cvr"] = logit_cvr
        imp["_true_p_ctr"] = p_ctr
        imp["_true_p_cvr"] = p_cvr
        imp["_true_delay_h"] = delay_h
        imp["_true_y_conv"] = y_conv_true

        imp["y_click"] = y_click
        imp["elapsed_h"] = elapsed_h

        # 两套标签，含义不同，绝不能混用：
        #
        # y_conv_w{w}  完整回补 + 窗口 w。离线消融用，衡量「窗口长度」的影响。
        #              四个 w 之间只有窗口不同，假负率才是一条干净的单调曲线。
        for w in cfg.window_hours_list:
            imp[f"y_conv_w{w}"] = ((y_conv_true == 1) & (delay_h <= w)).astype(np.int32)

        # y_conv       线上真实拿到的标签 = 24h 窗口 **再叠加数据末尾截断**。
        #              末尾 7 天的曝光等不满 24h，转化还没发生就已经被标成负样本。
        #              这才是训练时真正该用的那一列，也是假负样本的主要来源。
        imp["y_conv"] = ((y_conv_true == 1)
                         & (delay_h <= np.minimum(float(cfg.window_hours), elapsed_h))
                         ).astype(np.int32)

        # oracle：无限窗口，**仅用于评测，绝不进训练**
        imp["y_conv_oracle"] = y_conv_true
        return imp

    # ------------------------------------------------------------------
    def _true_logits(self, x, U, A, u_act, a_pop, ctx_slot, page_kind, day,
                     seg_true=None, ctype=None):
        """真值 logit = 线性 + 二阶 + 分层×素材 + 三阶 + 四阶 + RFF + 时间漂移。

        **每个成分对应一个模型的表达能力上限，这是整个实验设计的骨架：**

        | 成分 | 形式 | LR | FM | DeepFM |
        |---|---|---|---|---|
        | ``lin`` | x·w | ✓ | ✓ | ✓ |
        | ``cross2`` | α₂⟨U₄,A₄⟩ | ✗ | ✓ | ✓ |
        | ``M[seg,ct]`` | 分层×创意的二维表 | ✓\* | ✓ | ✓ |
        | ``T3`` | T3[u_act,a_pop,slot] | ✗ | **✗** | ✓ |
        | ``T4`` | T4[u_act,a_pop,slot,page] | ✗ | **✗** | ✓ |
        | ``rff`` | 随机傅里叶（含高阶） | 部分 | 部分 | 大部分 |

        \* ``M`` 是二维交叉，本来只有二阶模型能表达；但特征表里有
          ``seg_x_creative`` 这个**预先交叉好**的字段（基数 96），
          它的一阶项就能把整张表背下来。所以 M 对三个模型都可见 ——
          它存在的意义不是制造差距，而是让职责点 1 的分层**真的有业务价值**
          （分层不同、素材不同，收益不同），而不是一个摆设。

        真正制造差距的是 ``T3``/``T4``：它们是**四个独立字段**上的
        三阶/四阶结构，FM 的模型形式里没有三阶项，无法表达；
        DNN 可以任意组合。前提是这四个键在特征里可见 ——
        见 ``features/pipeline.py::_high_order_keys``，那里记录了一个
        「设计上说得通、跑起来是空转」的坑。

        时间漂移是刻意加的，且**不给模型 ``day`` 这个特征**，于是它变成
        模型解释不了的分布偏移。后果有两个，都是我们想要的：

        1. 测试集 AUC 天然低于验证集 → 「模型为什么要定期重训」有了实据；
        2. 概率校准会随时间失效 → 出价环节的水位控制必须能自适应。

        ``drift_rate`` 抬升/下压整体基率（概念+先验漂移），
        ``drift_fatigue`` 让素材信号的有效系数随时间线性增长（素材疲劳）。
        """
        cfg = self.cfg
        rng = self.rng
        d_in = x.shape[1]
        f32 = np.float32

        lin = x @ rng.normal(0.0, 0.55, size=d_in).astype(f32)
        # 二阶：用户信号 × 素材信号（FM 抓得到，因为这等价于两域嵌入的内积）
        cross2 = cfg.alpha_2 * (U[:, :4] * A[:, :4]).sum(axis=1)
        seg_term, seg_term_cvr = self._seg_matrix(seg_true, ctype)

        W_c, b_c, a_c = self._rff_for(("ctr", d_in, cfg.rff_dim),
                                      d_in, cfg.rff_dim, cfg.rff_sigma_w)
        W_v, b_v, a_v = self._rff_for(("cvr", d_in, cfg.rff_dim),
                                      d_in, cfg.rff_dim, cfg.rff_sigma_w * 1.15)
        rff_ctr = rff_apply(x, W_c, b_c, a_c) * cfg.rff_scale
        rff_cvr = rff_apply(x, W_v, b_v, a_v) * cfg.rff_scale

        # 三阶 / 四阶张量：FM 的模型形式里没有这一项，DNN 才学得到
        if "T3" not in self._tensors:
            self._tensors["T3"] = rng.normal(0, 1, size=(4, 4, 4)).astype(f32)
            self._tensors["T4"] = rng.normal(0, 1, size=(4, 4, 4, 4)).astype(f32)
            self._tensors["T3c"] = rng.normal(0, 1, size=(4, 4, 4)).astype(f32)
            self._tensors["T4c"] = rng.normal(0, 1, size=(4, 4, 4, 4)).astype(f32)
        T = self._tensors

        # 时间漂移：t 从 0 线性走到 1。模型看不到 day，所以这部分是纯偏移。
        t = (np.asarray(day, dtype=np.float64)
             / max(1.0, cfg.n_days - 1.0))
        drift = cfg.drift_rate * (t - 0.5)
        fatigue = cfg.drift_fatigue * t

        logit_ctr = (lin + cross2 + seg_term
                     + cfg.alpha_3 * T["T3"][u_act, a_pop, ctx_slot]
                     + cfg.alpha_4 * T["T4"][u_act, a_pop, ctx_slot, page_kind]
                     + rff_ctr + drift + fatigue * A[:, 0]).astype(np.float64)
        logit_cvr = (0.85 * lin + 0.9 * cross2 + 0.9 * seg_term_cvr
                     + cfg.alpha_3 * 0.9 * T["T3c"][u_act, a_pop, ctx_slot]
                     + cfg.alpha_4 * 0.9 * T["T4c"][u_act, a_pop, ctx_slot, page_kind]
                     + rff_cvr + 0.85 * drift + 0.8 * fatigue * A[:, 1]
                     ).astype(np.float64)
        return logit_ctr, logit_cvr

    def _seg_matrix(self, seg_true, ctype):
        """分层 × 创意的效应表：**低秩 + 稀疏高秩残差**。

        为什么要拆成两部分：低秩那半（秩 2）等价于「分层嵌入 · 创意嵌入」，
        FM 的二阶项天然表达得了；稀疏残差那 6 个格子是任意形状的，
        没有任何低秩结构，只有 DNN 的高阶组合能拟合。

        这也是「分层不是摆设」的落点：不同的（人群分层 × 素材类型）
        组合带来不同的广告效果，而这个结构是真实存在的、可被发现的。

        ⚠️ 真值里用的是**注入的真实分层** ``_true_segment``，
           而模型看到的是 KMeans 从 PCA 空间里聚出来的 ``user_segment``。
           两者只有在聚类质量足够好时才对齐（测试里断言 ARI > 0.85）。
           这个「真分层不可见、只能靠无监督方法逼近」的设定是有意的：
           如果直接把 ``_true_segment`` 喂给模型，职责点 1 就白做了。
        """
        if seg_true is None or ctype is None:
            return (np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64))

        cfg, rng = self.cfg, self.rng
        n_seg = cfg.n_true_segments
        n_ct = 8
        f32 = np.float32

        def build(key: str) -> np.ndarray:
            if key not in self._tensors:
                P = rng.normal(0.0, 1.0, size=(n_seg, cfg.seg_rank)).astype(f32)
                Q = rng.normal(0.0, 1.0, size=(n_ct, cfg.seg_rank)).astype(f32)
                M = (P @ Q.T) / math.sqrt(cfg.seg_rank)
                n_cell = n_seg * n_ct
                flat = rng.choice(n_cell, size=min(cfg.seg_sparse_n, n_cell),
                                  replace=False)
                M.reshape(-1)[flat] += rng.normal(
                    0.0, 2.0, size=len(flat)).astype(f32)
                self._tensors[key] = M
            return self._tensors[key]

        M = build("M_seg_ctr")
        Mc = build("M_seg_ctr_cvr")
        return (cfg.alpha_seg * M[seg_true, ctype].astype(np.float64),
                cfg.alpha_seg * Mc[seg_true, ctype].astype(np.float64))

    def _rff_for(self, key, d_in, n_basis, sigma_w):
        if key not in self._rff:
            self._rff[key] = make_rff(d_in, n_basis, sigma_w, self.rng)
        return self._rff[key]

    # ------------------------------------------------------------------
    def _sample_delay(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """三分量混合延迟：立即 / 同会话 / 长尾。

        比单一对数正态可控性好得多 —— 能精确调出目标假负率，
        而且天然给出「窗口越长假负越少」的单调曲线，正好当消融对照组。
        """
        cfg = self.cfg
        w = np.asarray(cfg.delay_mix, dtype=np.float64)
        comp = rng.choice(3, size=n, p=w / w.sum())
        delay = np.empty(n, dtype=np.float64)
        m0, m1, m2 = comp == 0, comp == 1, comp == 2
        delay[m0] = 0.0
        delay[m1] = rng.uniform(0.0, cfg.delay_uniform_max_h, int(m1.sum()))
        delay[m2] = rng.lognormal(cfg.delay_lognorm_mu, cfg.delay_lognorm_sigma,
                                  int(m2.sum()))
        return np.clip(delay, 0.0, cfg.delay_truncate_h).astype(np.float32)
