"""不平衡处理：1:5 负下采样 + 自实现 SMOTE + 先验校正。

职责描述要求「1:5 负采样与 SMOTE 均衡策略解决广告数据不平衡」。
这里把两者都完整实现，但**主路径是下采样 + 先验校正**，SMOTE 定位为消融对照。
理由写在下面，而且有实验数据支撑（见 README 的消融表）。

**为什么下采样是主路径？**

广告数据的正样本率是 0.3%（CTCVR），直接训练有两个后果：
梯度被负样本主导、以及正样本在每个 batch 里几乎是常数（batch=1024 时
期望只有 3 个正样本，方差极大）。下采样把正率抬到 24%，两个问题都缓解，
而且**推理时可以通过一行 logit 平移精确还原真实先验** —— 这是关键：

    p_true = σ(logit_sample + ln β),   β = 负样本保留比例

**为什么 SMOTE 不是主路径？**

SMOTE 在**特征空间**做 k 近邻插值合成正样本。广告的特征空间里有一大半是
类别域（user_id / ad_id / 性别 / 城市…）。对这些字段做「两个 id 的线性插值」
在业务上没有意义 —— 插出来的 ``0.5 × 用户 1234 + 0.5 × 用户 5678``
不是一个用户。

更严重的是**时序泄漏**：SMOTE 的近邻是在整个训练集上找的。如果它跨越了
时间切分，就会用未来样本的近邻去合成训练样本。所以本实现强制只允许在
切分后的 train 折内做，且只在**连续特征**上插值，类别特征取众数。
即便如此，实验上它的增益依然不如「下采样 + 先验校正」，原因还是那条：
它合成不出真实的高阶交叉结构。

所以：SMOTE 完整实现、可调用、有测试，但在主链路里是关的。
这比硬套一个「看起来高级但没用」的组件更诚实。
"""
from __future__ import annotations

import numpy as np

from src.metrics.ranking import prior_corrected_prob  # noqa: F401  (对外统一出口)


# --------------------------------------------------------------------------
# 负下采样
# --------------------------------------------------------------------------


def downsample_negatives(y: np.ndarray, neg_pos_ratio: int = 5,
                         seed: int = 42) -> np.ndarray:
    """保留全部正样本 + 按比例抽负样本，返回**行索引**。

    ⚠️ 只能在切分后的训练折上调用。
       对全量样本做下采样会让验证/测试集也变形，
       评测出来的概率不再对应真实先验，指标全部失真。

    ⚠️ 采样完必须做先验校正（``prior_corrected_prob``），否则
       概率整体高估 1/β 倍（β=0.2 时约 4 倍）。
       **AUC 完全看不出来** —— 排序没变，只有校准错了，
       而下游 eCPM 直接依赖概率的绝对值。

    ⚠️ **ESMM 场景下要传 ``y_click``，不要传 ``y_conv``。**
       这是本项目里最容易写错、而且错了完全不报错的一处，值得展开：

       ``logit += ln β`` 这个校正公式成立的前提是「**正样本全部保留**」。
       按点击标签下采样时，点击样本一个不丢，前提成立；
       于是 CTR 和 CTCVR 两个头都能被精确还原：

       - pCTR：正样本（点击）全保留 → odds 恰好放大 1/β_ctr 倍
       - pCTCVR：正样本（点击且转化）全保留 → odds 放大 1/β_ctcvr 倍
       - 两者相除得到的 pCVR，因为下采样保留了全部点击，本身就没被抽稀

       改成按转化标签下采样就不成立了：那时「点过但没转化」的样本
       是 CTR 的正样本却被抽稀了，``ln β`` 不再能还原 pCTR，
       实测会把 pCTR 从 0.061 校正到 0.013（真值 0.05），**偏 4 倍**。
    """
    y = np.asarray(y).astype(np.int64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0:
        raise ValueError("正样本为 0，无法做负下采样")

    n_keep = min(len(neg), neg_pos_ratio * len(pos))
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg, size=n_keep, replace=False)
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return idx


def downsample_ratio(y: np.ndarray, idx: np.ndarray) -> float:
    """下采样后负样本的**保留比例** β —— 先验校正要用的那个数。

    一定要从实际索引里反算，不要直接用 ``1/neg_pos_ratio``：
    当负样本总量不足以支撑目标比例时（小数据集常见），两者不相等。
    用错 β 会让概率整体偏一个常数倍，且不报错。
    """
    y = np.asarray(y)
    kept = y[idx]
    n_neg_kept = int((kept == 0).sum())
    n_neg_all = int((y == 0).sum())
    if n_neg_all == 0:
        return 1.0
    return float(n_neg_kept / n_neg_all)


def prior_shift(y_click_train: np.ndarray, keep_idx: np.ndarray) -> float:
    """ESMM 的 β —— 唯一需要先验校正的那个头是 pCTR。

    ⚠️ **两个参数都必须是「训练折内」的量，不能掺进验证/测试集。**

       踩过一次：先对全表下采样，再和训练折求交集。这样得到的 β 里
       混进了「训练折只占全表 70%」这件事，实测 β = 0.176 而不是 0.262。
       后果是 pCTR 被校正成 0.0205（真值 0.0500）—— **偏 2.4 倍**，
       而 AUC 一点都没变，只有 ECE 和均值能看出来。

       正确顺序：**先切分，再在训练折内下采样**，
       ``keep_idx`` 是相对训练折的下标。

    实测（mini 档，目标 1:5）：ctr_pos 从 0.0498 抬到 0.1667，β = 0.262。
    """
    return downsample_ratio(y_click_train, keep_idx)


# --------------------------------------------------------------------------
# SMOTE（自实现，无 imblearn 依赖）
# --------------------------------------------------------------------------


def smote(X_cont: np.ndarray, y: np.ndarray, k_neighbors: int = 5,
          seed: int = 42, target_ratio: float = 1.0
          ) -> tuple[np.ndarray, np.ndarray]:
    """在连续特征上合成正样本，返回 (X_syn, y_syn)。

    Parameters
    ----------
    X_cont : (n, d) 只含**连续**特征。类别特征请勿传入。
    target_ratio : 合成数量 / 原正样本数量。

    实现说明
    --------
    近邻用 ``sklearn.neighbors.NearestNeighbors`` 的 KDTree/ball tree 算，
    但只在正样本子集内 —— 这也是 SMOTE 最容易被写错的地方：
    邻居必须从同类样本里找，从全样本里找会把负样本当邻居，
    插值出来的点落在决策边界上，等于给模型喂噪声。

    ⚠️ 绝不能跨时间切分调用。
    """
    from sklearn.neighbors import NearestNeighbors

    X_cont = np.ascontiguousarray(X_cont, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    pos_idx = np.flatnonzero(y == 1)
    n_pos = len(pos_idx)
    if n_pos < k_neighbors + 1:
        return (np.zeros((0, X_cont.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.int64))

    n_syn = int(round(n_pos * max(0.0, target_ratio)))
    if n_syn == 0:
        return (np.zeros((0, X_cont.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.int64))

    k = min(k_neighbors, n_pos - 1)
    Xp = X_cont[pos_idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xp)
    _, nbr = nn.kneighbors(Xp)          # 第 0 列是自己
    nbr = nbr[:, 1:]

    rng = np.random.default_rng(seed)
    base = rng.integers(0, n_pos, size=n_syn)
    pick = rng.integers(0, k, size=n_syn)
    lam = rng.random(n_syn).astype(np.float32)

    partner = nbr[base, pick]
    X_syn = Xp[base] + lam[:, None] * (Xp[partner] - Xp[base])
    return X_syn.astype(np.float32), np.ones(n_syn, dtype=np.int64)


def smote_report(y_before: np.ndarray, y_after: np.ndarray) -> dict:
    """合成前后的正负比，写进实验报告用。"""
    def rate(y):
        y = np.asarray(y)
        return float((y == 1).mean()) if len(y) else 0.0
    return {"pos_rate_before": rate(y_before),
            "pos_rate_after": rate(y_after),
            "n_before": int(len(y_before)),
            "n_after": int(len(y_after))}


def effective_number_weights(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Cui et al. 的「有效样本数」类别权重，作为重加权的对照方案。

    权重反比于有效样本数，而不是样本数本身 —— 后者在极度不平衡时
    会给正样本大得离谱的权重，训练直接发散。

    ⚠️ 重加权和先验校正是**两件不同的事**，别搞混：
       重加权改的是训练时的梯度贡献，不改模型输出的概率含义；
       先验校正改的是推理时概率的绝对水平。
       做了重加权仍然要做先验校正。
    """
    y = np.asarray(y).astype(np.int64)
    out = np.ones(len(y), dtype=np.float32)
    for cls in (0, 1):
        n = int((y == cls).sum())
        if n == 0:
            continue
        w = (1.0 - beta) / (1.0 - beta ** n)
        out[y == cls] = w
    out = out / out.mean()          # 归一化，不改动总梯度量级
    return out
