"""02 特征工程与用户分层：画像宽表 → PCA → KMeans++ → 建模特征包。

产出
----
``artifacts/models/pca_{scale}.pkl``       PCA 结果（含均值/方差/成分）
``artifacts/models/kmeans_{scale}.pkl``    KMeans 结果（标签/中心/K/轮廓）
``artifacts/data/bundle_{scale}.pkl``      建模用特征矩阵（类别 + 数值）
``artifacts/metrics/02_features_{scale}.json``

**这一步为什么关键**

PCA 和 KMeans 有个共同点：**它们永远不报错**。给一堆纯噪声，它们照样
吐出 8 个漂亮的簇和一条单调下降的方差曲线。所以「跑通了」在这里
一文不值 —— 唯一有意义的判据是「分出来的层和**人工注入的真实分层**
对得上吗」，也就是 **ARI**。

轮廓系数在这里是**诊断量而不是判据**，原因见 ``KM.select_k`` 的长注释：
它按点平均，天然偏袒大小均衡的球形簇，在本数据上会把真实的 5 层
选成 4 层（ARI 从 0.995 掉到 0.75）。沿用轮廓系数当门槛，
「分层分对了」这件事反而会被判成失败。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from sklearn.metrics import adjusted_rand_score             # noqa: E402

from scripts._common import (DATA_DIR, MODEL_DIR, Timer, banner,  # noqa: E402
                             cfg_for, load_data, log, parse_scale,
                             save_artifact, save_metrics, show_table)
from src.common.seed import set_seed                        # noqa: E402
from src.data import schema as S                            # noqa: E402
from src.features import kmeans as KM                       # noqa: E402
from src.features import pca as PCA                         # noqa: E402
from src.features import pipeline as P                      # noqa: E402


def bundle_leakage(bundle, profile_cols: list[str]) -> dict:
    """特征包泄漏检查 —— 这里才是真正该卡「``_true_`` 不许进模型」的地方。

    01 步检查的是原始表（那里 ``_true_*`` 是地面真值，**必须**存在）；
    这一步检查的是**模型的真实输入**。两者不能混为一谈：
    原始表里有 latent 是设计，特征包里有 latent 是故障 ——
    后者会让 AUC 飞到 0.95 以上，整条链路看着完美、其实什么都没证明。
    """
    names = ([n for n, _ in bundle.cat_fields] + list(bundle.dense_cols)
             + list(bundle.user_profile_cols) + list(profile_cols))
    bad = [n for n in names if n.startswith("_true_") or "true" in n.lower()]
    return {
        "n_named_columns": len(names),
        "leaked_columns": bad,
        "leaked": bool(bad),
    }


def feature_table_shape(bundle, profile_cols: list[str]) -> dict:
    """「特征有多少列」的完整口径 —— 分两个**不能混**的层面。

    1. **画像宽表**：职责描述里「200+ 维特征」指的就是它，
       是 SparkSQL 那张宽表，也是 PCA / KMeans 的输入。
    2. **建模特征**：模型真正吃到的东西。类别域走嵌入表而不是独热，
       所以「独热宽度 = Σ 基数」在这里是个**误导性**的数
       （``user_id`` 一个域就是 10 万）。真正有意义的三个数是
       域个数、嵌入表总规模、数值宽度。

    这两个数都从对象上**数出来**，绝不手写 —— 手写的数字迟早和代码漂移，
    而它们会被抄进 README。

    ⚠️ 数的是 ``profile_cols``（``build_user_profile`` 声明的**特征**列），
    **不是** ``profile.columns``。踩过的坑：一开始数的是后者，于是
    「画像宽表 209 列 = 类别 22 + 数值 84 + 纯噪声 60 + 冗余 26」——
    右边加起来只有 192，**这行日志自己跟自己矛盾**。多出来的 17 列是
    ``user_id`` / ``pay_amount`` / ``_true_segment`` 这类**簿记与隐变量**列：
    它们跟着 ``users`` 表一起被 copy 进来，却不是特征，也**绝不能**进模型
    （``bundle_leakage`` 就在下面专门断言它们没泄漏）。
    把它们算进"特征宽度"，等于一边声称不泄漏、一边把它们的数量当成卖点。
    连带 `profile_noise_share` 的分母也被撑大，噪声占比被**低估**
    （41.1% vs 真实的 44.8%）。
    """
    cols = list(profile_cols)
    n_noise = sum(1 for c in cols if c.startswith("noise_"))
    n_redund = sum(1 for c in cols if c.startswith("redund_"))
    n_numeric = len(S.USER_PROFILE_NUMERIC)
    n_cat = len(S.USER_CAT_FIELDS)
    cards = [c for _, c in bundle.cat_fields]
    return {
        # --- 画像宽表 ---
        "profile_columns": len(cols),
        "profile_numeric": n_numeric,
        "profile_categorical": n_cat,
        "profile_noise": n_noise,
        "profile_redundant": n_redund,
        "profile_noise_share": (n_noise + n_redund) / len(cols) if cols else 0.0,
        # --- 建模特征 ---
        "model_cat_fields": bundle.n_fields,
        "model_embedding_rows": int(sum(cards)),
        "model_user_dense": bundle.user_dense_dim,
        "model_item_dense": bundle.item_dense_dim,
        "model_imp_dense": bundle.imp_dense_dim,
        "model_pca": int(bundle.user_pca.shape[1]),
    }


def cluster_table(labels: np.ndarray, k: int,
                  users) -> tuple[list[dict], float]:
    """每个簇的规模 + 其真实分层构成。

    单看簇大小只能说明「没塌成一坨」，说明不了「分对了」。
    把**真实层的分布**打出来才有判断力：一个好分层的簇，
    真实层应当是**高度集中**的（一个簇里 90% 是同一个真层），
    而不是五个真层各占 20% —— 后者说明 KMeans 只是把噪声切开了。
    """
    seg = users["_true_segment"].to_numpy()
    rows, purities = [], []
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        cnt = Counter(seg[m].tolist())
        top_seg, top_n = (cnt.most_common(1)[0] if cnt else (-1, 0))
        purity = top_n / n if n else 0.0
        purities.append(purity)
        rows.append({
            "cluster": c,
            "n_users": n,
            "share": n / len(labels),
            "top_true_segment": top_seg,
            "purity": purity,
        })
    return rows, float(np.average(purities, weights=np.bincount(labels)))


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)

    data = load_data(cfg)
    users, items, imp = data["users"], data["items"], data["impressions"]
    rng = np.random.default_rng([cfg.seed, 77])

    with Timer(f"02 画像 + PCA（{scale}）"):
        profile, profile_cols = P.build_user_profile(users, cfg, rng)
        X = P._safe_float32(profile, S.USER_PROFILE_NUMERIC)
        pca_res = PCA.fit_pca(X, cfg)
        X_pca = pca_res.transform(X)
        save_artifact(pca_res, MODEL_DIR / "pca.pkl", cfg)

    with Timer(f"02 KMeans++ 分层（{scale}）"):
        km_res = KM.fit_kmeans(X_pca, cfg)
        save_artifact(km_res, MODEL_DIR / "kmeans.pkl", cfg)

    with Timer(f"02 曝光级特征包（{scale}）"):
        bundle = P.build_feature_bundle(data, pca_res, km_res, cfg)
        save_artifact(bundle, DATA_DIR / "bundle.pkl", cfg)

    # ---------------- 分层质量 ----------------
    ari = float(adjusted_rand_score(users["_true_segment"].to_numpy(),
                                    km_res.labels))
    rows, purity = cluster_table(km_res.labels, km_res.k, users)

    banner("PCA")
    kept = float(pca_res.cumulative_ratio[pca_res.n_components - 1])
    log.info("主成分 %d 个（上限 %d），累计方差 %.4f",
             pca_res.n_components, cfg.pca_n_max, kept)
    log.info("正交性误差 %.2e", PCA.orthogonality_error(pca_res.components))
    log.info("「累计方差到 0.90」是设计目标，实测到不了 —— 84 个信号列的 "
             "SNR 只有 0.4，硬凑 0.90 要 60+ 维，反而害了聚类")

    banner("KMeans++ 分层")
    log.info("选中 K = %d（人工注入的真实分层是 %d 层）",
             km_res.k, cfg.n_true_segments)
    log.info("轮廓系数 %.4f（**诊断量**，判据是 ARI）", km_res.silhouette)
    show_table(rows, ["cluster", "n_users", "share", "top_true_segment",
                      "purity"], "各簇规模与真实层纯度")
    log.info("加权平均纯度 %.4f，ARI = %.4f", purity, ari)

    shape = feature_table_shape(bundle, profile_cols)
    banner("特征表规模")
    log.info("画像宽表 **%d 列** = 类别 %d + 数值 %d + 纯噪声 %d + 冗余 %d"
             "（噪声+冗余占 %.1f%%，是特征筛选的对照组，不是凑数）",
             shape["profile_columns"], shape["profile_categorical"],
             shape["profile_numeric"], shape["profile_noise"],
             shape["profile_redundant"], shape["profile_noise_share"] * 100)
    log.info("建模特征：类别域 %d 个（嵌入表合计 %d 行，**不是**独热宽度）"
             "+ 用户/素材/上下文数值 %d/%d/%d + 主成分 %d",
             shape["model_cat_fields"], shape["model_embedding_rows"],
             shape["model_user_dense"], shape["model_item_dense"],
             shape["model_imp_dense"], shape["model_pca"])

    leak = bundle_leakage(bundle, profile_cols)
    banner("特征包泄漏检查")
    log.info("模型输入共 %d 个命名列，命中 latent 的：%s",
             leak["n_named_columns"], leak["leaked_columns"] or "无")

    # ---------------- 门槛 ----------------
    problems = []
    if ari <= 0.85:
        problems.append(f"ARI {ari:.4f} ≤ 0.85：分层和真实层对不上")
    if km_res.silhouette <= 0.15:
        problems.append(f"轮廓系数 {km_res.silhouette:.4f} ≤ 0.15：簇结构不明显")
    if km_res.min_cluster_frac < 0.02:
        problems.append(f"最小簇占比 {km_res.min_cluster_frac:.4f} < 2%：有簇塌了")
    if leak["leaked"]:
        problems.append(f"latent 进了模型输入：{leak['leaked_columns']}")
    if pca_res.n_components > cfg.pca_n_max:
        problems.append(f"主成分 {pca_res.n_components} 超过上限 {cfg.pca_n_max}")

    metrics = {
        "scale": scale,
        "pca": {
            "n_components": int(pca_res.n_components),
            "cumulative_variance": kept,
            "orthogonality_error": PCA.orthogonality_error(pca_res.components),
        },
        "kmeans": {
            "k": int(km_res.k),
            "true_segments": int(cfg.n_true_segments),
            "silhouette": km_res.silhouette,
            "ari_vs_true_segment": ari,
            "weighted_purity": purity,
            "min_cluster_frac": km_res.min_cluster_frac,
            "clusters": rows,
            "inertia_scores": {str(k): v for k, v in
                               km_res.inertia_scores.items()},
            "silhouette_scores": {str(k): v for k, v in
                                  km_res.silhouette_scores.items()},
        },
        "features": shape,
        "leakage": leak,
        "problems": problems,
    }
    save_metrics("02_features", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：分层恢复到 ARI=%.4f，特征包无 latent 泄漏", ari)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
