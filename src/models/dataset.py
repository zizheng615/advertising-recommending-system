"""把 ``FeatureBundle`` + 标签打包成模型张量。

**为什么单独一个模块？**

脚本和测试必须用**完全相同的**张量构造逻辑，否则「测试里 AUC 0.72、
脚本里 0.68」这种事会反复发生，而且没人知道差在哪。所以张量拼装只有
一处实现，测试与 ``scripts/`` 都从这里取。

**为什么标签要拆成三个张量？**

ESMM 双头需要**两套标签**，而且在全量曝光上都可观测：

    y_click  点击标签     正率 ~5%
    y_conv   CTCVR 标签   正率 ~0.3%（**有偏**：24h 窗口 + 数据末尾截断）
    y        上面两列叠起来，喂给 ``esmm_loss`` 用

训练循环必须在 forward 之前把**所有**标签张量摘出去，否则
``model(**batch)`` 会把 ``y_click`` 当成模型参数名传进去然后报
``unexpected keyword argument``。所以这里给 ``label_keys`` 一个统一定义。
"""
from __future__ import annotations

import numpy as np
import torch

from config import Config
from src.data import schema as S
from src.features.pipeline import FeatureBundle

# 训练循环要摘掉的键。顺序即 ``y`` 的列顺序。
RANKER_LABEL_KEYS = ("y_click", "y_conv")


# --------------------------------------------------------------------------
# 排序模型（DeepFM / FM / LR / DFM）
# --------------------------------------------------------------------------


def ranker_tensors(bundle: FeatureBundle, imp, cfg: Config,
                   label_col: str = "y_conv",
                   with_elapsed: bool = False) -> dict[str, torch.Tensor]:
    """曝光级张量。`y` 的两列固定是 (点击, 转化)。

    ``label_col`` 决定第二列取哪一列标签：
      ``y_conv``        线上真实能拿到的有偏标签 —— 主链路用它
      ``y_conv_w168``   168h 完全回补 —— 只用于消融对照
    """
    if label_col not in imp.columns:
        raise KeyError(f"曝光表里没有标签列 {label_col!r}")

    t: dict[str, torch.Tensor] = {
        "cat": torch.from_numpy(np.ascontiguousarray(bundle.imp_cat)).long(),
        "dense": torch.from_numpy(np.ascontiguousarray(bundle.imp_dense)).float(),
    }
    y_click = imp["y_click"].to_numpy(np.float32).reshape(-1, 1)
    y_conv = imp[label_col].to_numpy(np.float32).reshape(-1, 1)
    t["y_click"] = torch.from_numpy(y_click)
    t["y_conv"] = torch.from_numpy(y_conv)
    # (B, 2)：第 0 列点击、第 1 列转化
    t["y"] = torch.from_numpy(np.concatenate([y_click, y_conv], axis=1))
    if with_elapsed:
        t["elapsed_h"] = torch.from_numpy(
            imp["elapsed_h"].to_numpy(np.float32).reshape(-1, 1))
    return t


# --------------------------------------------------------------------------
# ZILN：用户级
# --------------------------------------------------------------------------


def _side_fields(bundle: FeatureBundle, prefix: str,
                 source: list[tuple[str, int]],
                 n_cols: int, side: str) -> list[tuple[str, int]]:
    """按**名单**（而不是按前缀）取出一侧的字段。

    ⚠️ 不能简单地「筛出所有 ``u_`` / ``a_`` 开头的字段」。
       曝光级字段表里还有高阶交叉键 ``u_act`` / ``a_pop``
       （见 ``pipeline._high_order_keys``）—— 它们看起来像用户侧/素材侧
       字段，实际上是**用户 × 上下文**、**素材 × 上下文**的交叉键，
       既不属于 user tower 也不属于 item tower。

       实测：按前缀筛会多出 1 个 ``u_`` 字段（23 vs 22），
       而 ``EmbeddingBank`` 的「列数 == 字段数」校验会拦住它 ——
       但如果那个校验恰好也通过（比如交叉键正好有两个），
       就会变成**静默的字段错位**，比报错危险得多。

    所以：名单来自 ``S.USER_CAT_FIELDS`` / ``S.ITEM_CAT_FIELDS``（唯一真源），
    再用 ``bundle.cat_fields`` 查出各自的基数。
    """
    card = dict(bundle.cat_fields)
    fields: list[tuple[str, int]] = []
    for name, base_card in source:
        key = f"{prefix}{name}"
        if key not in card:
            raise KeyError(
                f"曝光级字段表里找不到 {side}字段 {key!r} —— "
                f"特征打包和模型侧的字段命名对不上了")
        fields.append((key, card[key]))
        # 基数在 pipeline 里统一 +1（多一档给缺失），这里做一次交叉校验
        if card[key] != base_card + 1:
            raise ValueError(
                f"{key} 的基数 {card[key]} 与 schema 里的 {base_card}+1 不符")
    if len(fields) != n_cols:
        raise ValueError(
            f"{side}字段数 {len(fields)} 与矩阵列数 {n_cols} 不一致")
    return fields


def user_cat_fields(bundle: FeatureBundle) -> list[tuple[str, int]]:
    """用户侧字段（``u_`` 前缀），顺序与 ``bundle.user_cat`` 严格一致。

    注意这里**只含** ``S.USER_CAT_FIELDS`` —— 分层结果 ``user_segment``
    不在这里，它在 ``ziln_user_tensors`` 里单独追加（排序模型则通过
    ``bundle.imp_cat`` 拿到它）。
    """
    return _side_fields(bundle, "u_", S.USER_CAT_FIELDS,
                        bundle.user_cat.shape[1], "用户侧")


def item_cat_fields(bundle: FeatureBundle) -> list[tuple[str, int]]:
    """素材侧字段（``a_`` 前缀），顺序与 ``bundle.item_cat`` 严格一致。

    为什么前缀是 ``a_`` 而不是 ``ad_``：曝光级还有 ``ad_ctr_global``
    / ``ad_cvr_global`` 这类**统计量**列，用 ``ad_`` 会让「素材属性」
    和「素材的全局统计」在名字上混在一起。
    """
    return _side_fields(bundle, "a_", S.ITEM_CAT_FIELDS,
                        bundle.item_cat.shape[1], "素材侧")


def ziln_user_tensors(bundle: FeatureBundle, users, user_idx: np.ndarray,
                      cfg: Config) -> dict[str, torch.Tensor]:
    """用户级张量：特征是用户画像，标签是付费金额。

    ⚠️ **ZILN 必须在用户级训练，不能把用户标签摊到曝光级。**

       同一个用户有几十条曝光，标签完全相同。摊到曝光级训练的话，
       「这个用户付了 300 块」这件事会被重复计几十次，测试集上的
       有效样本量被虚增几十倍 —— AUC 的置信区间窄得离谱，
       但真实的用户级不确定性一点没减少。这不是 bug 会报错的那种问题，
       它是一个「指标看起来很稳」的假象。

    所以：一行一个用户，特征只用用户侧。
    """
    seg = bundle.user_segment[user_idx].reshape(-1, 1).astype(np.int32)
    cat = np.concatenate([bundle.user_cat[user_idx], seg], axis=1)

    fields = user_cat_fields(bundle)
    # 分层是职责点 1 的产物，直接作为职责点 2 的输入 —— 这正是两个功能点的衔接处。
    fields = fields + [("user_segment", 12)]

    amount = users["pay_amount"].to_numpy(np.float32)[user_idx].reshape(-1, 1)
    return {
        "cat": torch.from_numpy(np.ascontiguousarray(cat)).long(),
        "dense": torch.from_numpy(
            np.ascontiguousarray(bundle.user_dense[user_idx])).float(),
        "y": torch.from_numpy(amount),
        "_fields": fields,          # 元信息，训练循环会自动忽略（非张量）
    }


# --------------------------------------------------------------------------
# 双塔：正负对
# --------------------------------------------------------------------------


def _seen_ads_by_user(imp, n_users: int, click_only: bool = False
                      ) -> dict[int, np.ndarray]:
    """每个用户「看过但没点」的素材列表 —— 用作难负样本。

    ⚠️ 随机采的负样本太容易区分（模型只要学会「这是不是该用户的兴趣类目」
       就能全对），学到的边界对召回毫无意义。用真实曝光过但没点的素材，
       负样本才和正样本同分布。
    """
    u = imp["user_id"].to_numpy()
    a = imp["ad_id"].to_numpy()
    if click_only:
        m = imp["y_click"].to_numpy() == 0
        u, a = u[m], a[m]
    out: dict[int, np.ndarray] = {}
    order = np.argsort(u, kind="stable")
    u_sorted = u[order]
    a_sorted = a[order]
    bounds = np.flatnonzero(np.diff(u_sorted)) + 1
    for chunk in np.split(np.arange(len(u_sorted)), bounds):
        uid = int(u_sorted[chunk[0]])
        out[uid] = np.unique(a_sorted[chunk])
    return out


def build_pairs(bundle: FeatureBundle, imp, idx: np.ndarray,
                n_neg: int = 5, seed: int = 42) -> dict[str, torch.Tensor]:
    """从曝光表里挖出「用户点过的素材」作为正样本对，配 N 个难负样本。

    返回的张量里，``i_*_neg`` 的形状是 ``(B, n_neg, ...)``。
    """
    sub = imp.iloc[idx]
    pos = sub["y_click"].to_numpy() == 1
    if int(pos.sum()) < 2:
        raise ValueError("正样本对不足，无法训练双塔")

    rows = np.flatnonzero(pos)
    u_ids = sub["user_id"].to_numpy()[rows]
    a_pos = sub["ad_id"].to_numpy()[rows]
    n = len(rows)

    seen = _seen_ads_by_user(imp, len(bundle.user_cat), click_only=True)
    rng = np.random.default_rng(seed)
    n_items = bundle.item_cat.shape[0]
    a_neg = np.empty((n, n_neg), dtype=np.int64)
    for i in range(n):
        pool = seen.get(int(u_ids[i]))
        if pool is not None:
            pool = pool[pool != a_pos[i]]
        if pool is None or len(pool) == 0:
            pool = np.arange(n_items)
        # 池子不够 n_neg 个时允许重复抽 —— 重复的负样本只是权重高一点，
        # 比「补齐成随机 id」引入的分布不一致要好。
        a_neg[i] = rng.choice(pool, size=n_neg, replace=len(pool) < n_neg)

    return {
        "u_cat": torch.from_numpy(np.ascontiguousarray(bundle.user_cat[u_ids])).long(),
        "u_dense": torch.from_numpy(
            np.ascontiguousarray(bundle.user_dense[u_ids])).float(),
        "i_cat_pos": torch.from_numpy(
            np.ascontiguousarray(bundle.item_cat[a_pos])).long(),
        "i_dense_pos": torch.from_numpy(
            np.ascontiguousarray(bundle.item_dense[a_pos])).float(),
        "i_cat_neg": torch.from_numpy(
            np.ascontiguousarray(bundle.item_cat[a_neg])).long(),
        "i_dense_neg": torch.from_numpy(
            np.ascontiguousarray(bundle.item_dense[a_neg])).float(),
    }


def recall_targets(bundle: FeatureBundle, imp, idx: np.ndarray,
                   seed: int = 42) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """评测用：去重后的 (用户, 点过的素材) 对 + 目标素材下标。"""
    sub = imp.iloc[idx]
    m = sub["y_click"].to_numpy() == 1
    u_ids = sub["user_id"].to_numpy()[m]
    a_ids = sub["ad_id"].to_numpy()[m]
    pairs = np.unique(np.stack([u_ids, a_ids], axis=1), axis=0)
    if len(pairs) == 0:
        raise ValueError("评测集里没有点击对")
    uu, aa = pairs[:, 0], pairs[:, 1]
    tensors = {
        "u_cat": torch.from_numpy(np.ascontiguousarray(bundle.user_cat[uu])).long(),
        "u_dense": torch.from_numpy(
            np.ascontiguousarray(bundle.user_dense[uu])).float(),
        "i_cat": torch.from_numpy(
            np.ascontiguousarray(bundle.item_cat)).long(),
        "i_dense": torch.from_numpy(
            np.ascontiguousarray(bundle.item_dense)).float(),
    }
    return tensors, torch.from_numpy(aa).long()
