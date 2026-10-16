"""PCA 降维。

⚠️ **必须先 StandardScaler 再 PCA。** 否则方差大但毫无意义的列会主导主成分 ——
比如「曝光次数」量级 1e4，「CTR」量级 1e-2，不标准化的话前几个主成分
基本就是曝光次数的各种线性组合，降维完全失去意义。

降维本身的意义：192 维里信号的有效秩只有 15~20，噪声和冗余占了 44.8%，
PCA 能把有效结构压到前几十个主成分里，后面 KMeans 才能在这个空间里
拉得开距离（原空间做聚类会撞上维度灾难）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import Config


@dataclass
class PCAResult:
    components: np.ndarray        # (n_components, n_features)
    explained_variance_ratio: np.ndarray
    cumulative_ratio: np.ndarray
    n_components: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.scaler_mean) / self.scaler_scale
        return Xs @ self.components.T


def fit_pca(X: np.ndarray, cfg: Config) -> PCAResult:
    """按「累计方差保留率」自动定维数，并卡一个上限。

    上限是必须的：KMeans 的轮廓系数在高维会退化，而且
    「200 维里大部分没用」这件事本身就该让 PCA 把维数压下来。
    """
    X = np.asarray(X, dtype=np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 先取全量主成分，再按累计方差截断（n_components 传 float 也行，
    # 但显式算一遍能把曲线画出来，README 里要用）
    n_max = min(Xs.shape[0], Xs.shape[1])
    pca_full = PCA(n_components=n_max, random_state=cfg.seed)
    pca_full.fit(Xs)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum, cfg.pca_var_keep) + 1)
    n_keep = int(min(max(n_keep, 1), cfg.pca_n_max))

    return PCAResult(
        components=pca_full.components_[:n_keep].astype(np.float32),
        explained_variance_ratio=pca_full.explained_variance_ratio_[:n_keep].astype(np.float32),
        cumulative_ratio=cum.astype(np.float32),
        n_components=n_keep,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )


def orthogonality_error(components: np.ndarray) -> float:
    """||C Cᵀ − I||_F —— PCA 主成分必须正交归一，用它做自检。"""
    C = np.asarray(components, dtype=np.float64)
    G = C @ C.T
    I = np.eye(G.shape[0])
    return float(np.linalg.norm(G - I, ord="fro"))
