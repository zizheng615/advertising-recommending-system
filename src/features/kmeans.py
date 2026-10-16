"""KMeans++ 用户分层。

两个工程决定
------------
1. **在 PCA 空间做，不在原始 200 维做。** 原空间的噪声和冗余列会把
   欧氏距离淹没（维度灾难），簇根本拉不开。
2. **用 MiniBatchKMeans 搜索 K、用全量 KMeans 定稿。** 10 万用户 × 10 个
   候选 K × n_init 在全量 KMeans 上要跑很久；MiniBatch 搜索快一个量级，
   选定 K 后再用全量精修。

轮廓系数在 10 万点上算一次是 O(n²)，必须抽样。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from config import Config


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    k: int
    silhouette: float
    silhouette_scores: dict = field(default_factory=dict)   # k -> score
    inertia_scores: dict = field(default_factory=dict)      # k -> inertia
    cluster_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def min_cluster_frac(self) -> float:
        if self.cluster_sizes.sum() == 0:
            return 0.0
        return float(self.cluster_sizes.min() / self.cluster_sizes.sum())


def select_k(inertia: dict[int, float], sil: dict[int, float]) -> int:
    """选 K：**用肘部法（inertia 的拐点），不用轮廓系数**。

    ⚠️ 这里换过一次判据，原因是实测出来的，值得写清楚。

       最初按轮廓系数取最大，跑出来 K=4，而人工注入的真实分层是 5 层，
       ARI 只有 0.75。但**同一个数据**在 K=5 上 ARI = 0.9949 ——
       也就是说分层结构完全恢复得出来，坏的是「选 K」这一步。

       为什么轮廓系数会选错？因为它是**按点求平均**的，天然偏好
       「簇大小均衡、形状接近球形」的划分。而真实的五个层是
       1235 / 993 / 809 / 577 / 386 —— 最大最小差 3.2 倍。
       把最小的那两个（577 和 386）并成一个簇，剩下四个大小接近的簇，
       平均轮廓**反而更高**。实测 K=4 是 0.2348、K=5 是 0.2034。

       换成 Calinski-Harabasz（选 3）和 Davies-Bouldin（选 4）同样错。
       这不是实现问题，是这三个指标共同的偏好。

       肘部法看的是 inertia 的**下降速率**，对簇大小不敏感：
           K=4 → 165167，K=5 → 151598（掉了 13569），
           K=6 → 150027（只掉 1571）。
       K=5 处的落差是下一档的 8.6 倍，拐点非常干净。
       三个随机种子下肘部都落在 K=5，ARI 分别是 0.995 / 0.999 / 0.6+。

    轮廓系数仍然算、仍然返回（``KMeansResult.silhouette_scores``），
    但只作为**诊断量**：它低说明这个分层在高维噪声下簇间分离度一般，
    不代表分层错了。判断分层对不对要看 ARI。
    """
    if not inertia:
        raise ValueError("没有任何 K 的 inertia，无法选 K")
    return elbow_k(inertia)


def _silhouette_safe(X: np.ndarray, labels: np.ndarray, sample: int,
                     seed: int) -> float:
    """轮廓系数。簇数退化成 1、或每簇样本太少时返回 -1 而不是抛异常。"""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    n = len(labels)
    if n <= sample:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample, replace=False)
    sub_labels = labels[idx]
    # 抽样后必须仍然至少有 2 个簇，否则 silhouette_score 会抛错
    if len(np.unique(sub_labels)) < 2:
        return -1.0
    try:
        return float(silhouette_score(X[idx], sub_labels))
    except ValueError:
        return -1.0


def fit_kmeans(X_pca: np.ndarray, cfg: Config,
               verbose: bool = False) -> KMeansResult:
    """在 PCA 空间搜索最优 K 并定稿。

    Returns
    -------
    KMeansResult，其中 ``labels`` 是每个用户的层号。
    """
    X = np.asarray(X_pca, dtype=np.float32)
    n = X.shape[0]
    k_lo, k_hi = cfg.kmeans_k_range
    k_hi = min(k_hi, max(2, n // 10))          # 别让簇数超过样本量的 1/10
    if k_hi < k_lo:
        k_hi = k_lo

    sil: dict[int, float] = {}
    inertia: dict[int, float] = {}

    # 小数据量下用全量 KMeans 搜索：MiniBatch 的随机批次会让不同 K 的
    # 轮廓系数不可比，实测选出的 K 会偏小（真实 5 层时选到 4 层，
    # ARI 从 0.99 掉到 0.83）。只有数据真的很大时才为速度让路。
    use_full_search = n <= 30_000

    for k in range(k_lo, k_hi + 1):
        if k >= n:
            break
        if use_full_search:
            est = KMeans(n_clusters=k, init="k-means++", n_init=cfg.kmeans_n_init,
                         random_state=cfg.seed, max_iter=300)
        else:
            est = MiniBatchKMeans(
                n_clusters=k, init="k-means++", n_init=3,
                batch_size=min(4096, n), random_state=cfg.seed, max_iter=100,
            )
        labels = est.fit_predict(X)
        score = _silhouette_safe(X, labels, cfg.kmeans_silhouette_sample, cfg.seed)
        sil[k] = score
        inertia[k] = float(est.inertia_)
        if verbose:
            print(f"    K={k:2d}  silhouette={score:.4f}  inertia={est.inertia_:.1f}")

    best_k = select_k(inertia, sil)
    if verbose:
        print(f"    -> 选中 K={best_k}（肘部法），"
              f"轮廓系数 {sil.get(best_k, float('nan')):.4f}")

    # 定稿：选定 K 之后用全量 KMeans 精修
    # ⚠️ sklearn >= 1.4 的 n_init 默认是 "auto"，显式传值才能保证可复现
    km = KMeans(n_clusters=best_k, init="k-means++", n_init=cfg.kmeans_n_init,
                random_state=cfg.seed, max_iter=300)
    labels = km.fit_predict(X)
    final_score = _silhouette_safe(X, labels, cfg.kmeans_silhouette_sample, cfg.seed)
    sizes = np.bincount(labels, minlength=best_k).astype(int)

    return KMeansResult(
        labels=labels.astype(np.int32),
        centers=km.cluster_centers_.astype(np.float32),
        k=best_k,
        silhouette=final_score,
        silhouette_scores=sil,
        inertia_scores=inertia,
        cluster_sizes=sizes,
    )


def elbow_distances(inertia: dict[int, float]) -> dict[int, float]:
    """每个 K 的「到首尾连线距离」—— 肘部法里那个拐点的强度。

    距离越大，说明这个 K 越像曲线上的拐点。
    """
    if len(inertia) < 3:
        return {k: 0.0 for k in inertia}
    ks = np.array(sorted(inertia))
    ys = np.array([inertia[k] for k in ks], dtype=np.float64)
    xs = ks.astype(np.float64)
    x1, y1, x2, y2 = xs[0], ys[0], xs[-1], ys[-1]
    denom = np.hypot(y2 - y1, x2 - x1) + 1e-12
    dist = np.abs((y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1) / denom
    return {int(k): float(d) for k, d in zip(ks, dist)}


def elbow_k(inertia: dict[int, float]) -> int:
    """肘部法：取「到首尾连线距离最大」的那个 K。"""
    if not inertia:
        return 0
    d = elbow_distances(inertia)
    return max(d, key=lambda k: d[k])
