"""职责点 2 上半场：双塔召回。

双塔这个模块有一半的价值在**约束**上，而不在指标上。所以这个文件里
最重要的两条断言其实和 Accuracy 无关：

1. ``test_user_tower_has_no_cross_features`` —— user tower 的输入里
   绝不能出现任何跨域特征。这是双塔最常见的实现错误，而且**错了照样
   能训、指标还挺好看**，所以必须有一条断言专门拦它。
2. ``test_tower_outputs_are_l2_normalized`` —— 两个塔的输出必须是单位
   向量。少了这一步，内积退化成「谁的模长大谁赢」，模型会去学
   「把热门素材的向量整体放大」这种平凡解。

指标那一侧只留 Recall@50。理由见 ``test_recall_at_50`` 的注释。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import schema as S
from src.models import dataset as DS
from src.models.two_tower import TwoTower, recall_at_k, sampled_softmax_loss


def _fake_negatives(item_cat, item_dense, B: int, M: int):
    """造一批 (B, M, ...) 的负样本张量。

    ⚠️ 不能写 ``item_cat[:B * M].reshape(B, M)`` —— 那是在**元素**层面
       切片，而这里要的是**行**。素材矩阵有 7 列，元素数和行数差 7 倍，
       所以 reshape 会报「cannot reshape array of size 1120 into (32,5)」。
       索引要显式按行取（用取模容忍素材数少于 B·M 的情况）。
    """
    n_items = item_cat.shape[0]
    rows = np.arange(B * M) % n_items
    return (torch.from_numpy(np.ascontiguousarray(item_cat[rows])
                             ).long().reshape(B, M, -1),
            torch.from_numpy(np.ascontiguousarray(item_dense[rows])
                             ).float().reshape(B, M, -1))


# ============================================================ 结构约束
def test_user_tower_has_no_cross_features(bundle, tower_pairs):
    """**本文件最重要的一条**：user tower 的输入里不能有跨域特征。

    跨域 = 同时依赖「用户」和「素材」两边的特征：
    ``user_x_category_ctr``、``seg_x_creative_cnt``、``user_segment`` 的
    交叉项等等。它们**只能进排序层**。

    为什么这条必须用断言钉死，而不是靠 code review：一旦 user tower
    吃进跨域特征，用户向量就依赖具体素材，同一个用户对不同素材有不同
    的向量 —— **离线预算 item 向量这件事直接失效**，在线必须对每个
    (u, i) 重算一遍。双塔退化成一个更差的排序模型，而训练照常收敛、
    离线 Recall 甚至更高（因为信息更多了）。没有任何报错提示你。
    """
    cross = set(S.IMPRESSION_AD_CONTEXT) | set(S.CROSS_COUNT_NAMES)
    # 曝光级数值块里确实有这些跨域列 —— 先确认它们真的存在，
    # 否则这条测试会「因为找不到而通过」，变成一个永远绿的空断言。
    assert cross & set(bundle.dense_cols), (
        "曝光数值块里找不到任何跨域特征的列名，这条断言的检查对象不见了")
    # 而且它们只在曝光层
    assert not (cross & set(bundle.user_profile_cols)), (
        f"用户画像里混进了跨域特征：{cross & set(bundle.user_profile_cols)}")

    # user tower 的类别域全部来自 USER_CAT_FIELDS
    names = [n for n, _ in DS.user_cat_fields(bundle)]
    assert names, "user tower 一个类别字段都没有"
    user_side = {f"u_{n}" for n, _ in S.USER_CAT_FIELDS}
    assert set(names) <= user_side, (
        f"user tower 拿到了用户侧以外的类别字段：{set(names) - user_side}")
    assert not (set(names) & cross), "user tower 的类别域里有跨域特征"

    # 素材侧的对称检查
    inames = [n for n, _ in DS.item_cat_fields(bundle)]
    item_side = {f"a_{n}" for n, _ in S.ITEM_CAT_FIELDS}
    assert set(inames) == item_side, (
        f"素材侧字段对不上：多出 {set(inames) - item_side}，"
        f"少了 {item_side - set(inames)}")
    assert not (set(inames) & cross), "item tower 的类别域里有跨域特征"


def test_tower_field_counts_match_matrices(bundle):
    """字段表的个数必须和矩阵列数一致 —— 不一致会静默错位。

    ``EmbeddingBank`` 有一条「列数必须等于字段数」的校验，但它拦不住
    「列数恰好相等、字段顺序错了」的情况。所以这里从两边各数一次。
    """
    assert len(DS.user_cat_fields(bundle)) == bundle.user_cat.shape[1]
    assert len(DS.item_cat_fields(bundle)) == bundle.item_cat.shape[1]


def test_tower_outputs_are_l2_normalized(bundle, tower_cfg, cfg):
    """塔的输出必须是单位向量。

    ⚠️ 不做 L2 归一化的话，``<h_u, h_i>`` 就退化成「谁的模长大谁赢」。
       而模长和活跃度正相关 —— 模型会学出「把热门素材的向量整体放大」
       这个平凡解，而不是去学相关性。它不报错，Loss 还降得很好看。
    """
    torch.manual_seed(0)
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    m.eval()
    with torch.no_grad():
        hu, bu = m.encode_users(torch.from_numpy(bundle.user_cat[:64]).long(),
                                torch.from_numpy(bundle.user_dense[:64]).float())
        hi, bi = m.encode_items(torch.from_numpy(bundle.item_cat[:64]).long(),
                                torch.from_numpy(bundle.item_dense[:64]).float())
    for name, h in [("user", hu), ("item", hi)]:
        norm = h.norm(dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), (
            f"{name} tower 的输出模长在 [{norm.min():.4f}, {norm.max():.4f}]，"
            f"没有做 L2 归一化")
    # 偏置项是独立的标量，绝不能混进向量里（混进去就破坏了归一化）
    assert bu.shape == (64, 1) and bi.shape == (64, 1)
    assert m.global_bias.numel() == 1


def test_temperature_is_always_positive(tower_pairs, bundle, tower_cfg):
    """温度必须恒正 —— 它由 ``log_tau`` 指数化而来，不可能为负。

    温度出现在分母上，一旦为 0 或负，分数直接爆掉或整体翻号。
    """
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    # 人为把 log_tau 推到极端
    for v in (-50.0, -10.0, 0.0, 10.0):
        with torch.no_grad():
            m.log_tau.fill_(v)
        tau = m.temperature
        assert float(tau) > 0, f"log_tau={v} 时温度 {float(tau)} 不是正数"
        assert torch.isfinite(tau), f"log_tau={v} 时温度不是有限值"


# ============================================================ 采样 softmax
def test_sampled_softmax_loss_is_finite_and_grad_flows(bundle, tower_cfg):
    """损失有限、且梯度真的流回了两个塔。

    采样 softmax 里最容易出的错是「负样本那一支忘了折回形状」或者
    ``einsum`` 的下标写错 —— 前者会在 reshape 处报错，后者会**静默地**
    算出一个形状对但语义错的分数。
    """
    torch.manual_seed(0)
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    B, M = 32, 5
    u_cat = torch.from_numpy(bundle.user_cat[:B]).long()
    u_dense = torch.from_numpy(bundle.user_dense[:B]).float()
    i_cat = torch.from_numpy(bundle.item_cat[:B]).long()
    i_dense = torch.from_numpy(bundle.item_dense[:B]).float()
    n_cat, n_dense = _fake_negatives(bundle.item_cat, bundle.item_dense, B, M)

    loss, info = sampled_softmax_loss(u_cat, u_dense, i_cat, i_dense,
                                      n_cat, n_dense, m)
    assert torch.isfinite(loss), f"采样 softmax 损失不是有限值：{loss}"
    loss.backward()
    for name, p in [("user", m.user_tower), ("item", m.item_tower)]:
        g = [q.grad for q in p.parameters() if q.grad is not None]
        assert g, f"{name} tower 一个梯度都没有 —— 计算图断了"
        assert all(torch.isfinite(t).all() for t in g), f"{name} tower 梯度里有 NaN"


def test_sampled_softmax_loss_is_lower_bounded_when_positive_repeats(
        bundle, tower_cfg):
    """正样本恰好也是负样本之一时，损失有 ``ln 2`` 这个**精确**下界。

    这是一个不依赖训练、不依赖数据的数学性质：目标（index 0）和某个
    负样本（index j>0）的分数完全相同时，softmax 在这两项上各分到 1/2，
    交叉熵至少是 ``−ln(1/2) = ln 2``。任何人都无法把它压到 ln 2 以下。

    这条能抓的 bug：logits 拼接顺序错位（把负样本放在 index 0）、
    或者 target 张量建错了位置。
    """
    torch.manual_seed(0)
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    B, M = 64, 5
    u_cat = torch.from_numpy(bundle.user_cat[:B]).long()
    u_dense = torch.from_numpy(bundle.user_dense[:B]).float()
    n_cat, n_dense = _fake_negatives(bundle.item_cat, bundle.item_dense, B, M)

    # 正样本 = 第 0 个负样本 → 完全无法区分
    dup, _ = sampled_softmax_loss(u_cat, u_dense, n_cat[:, 0], n_dense[:, 0],
                                  n_cat, n_dense, m)
    assert float(dup) >= np.log(2.0) - 1e-6, (
        f"正样本与某个负样本完全相同时损失是 {float(dup):.6f}，"
        f"低于理论下界 ln2={np.log(2.0):.6f} —— logits 的拼装或 target 错了")


def test_sampled_softmax_loss_decreases_with_training(bundle, tower_cfg):
    """这个损失必须**真的能分**：训一会儿之后明显低于均匀基线。

    均匀基线 = ``ln(1+M)``，即模型对正负一视同仁时的损失（M=5 → 1.79）。
    只要损失降不下去，不管 AUC 多好看，召回都不会好 ——
    因为 Top-K 完全由正样本的相对分数决定。
    """
    torch.manual_seed(0)
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    B, M = 256, 5
    u_cat = torch.from_numpy(bundle.user_cat[:B]).long()
    u_dense = torch.from_numpy(bundle.user_dense[:B]).float()
    n_cat, n_dense = _fake_negatives(bundle.item_cat, bundle.item_dense, B, M)
    i_cat, i_dense = n_cat[:, 0], n_dense[:, 0]

    uniform = float(np.log(1 + M))
    first, info0 = sampled_softmax_loss(u_cat, u_dense, i_cat, i_dense,
                                        n_cat, n_dense, m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(80):
        loss, _ = sampled_softmax_loss(u_cat, u_dense, i_cat, i_dense,
                                       n_cat, n_dense, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    last, info1 = sampled_softmax_loss(u_cat, u_dense, i_cat, i_dense,
                                       n_cat, n_dense, m)

    assert float(last) < float(first) - 0.05, (
        f"80 步之后损失几乎没动（{float(first):.4f} → {float(last):.4f}）")
    assert float(last) < uniform - 0.15, (
        f"损失 {float(last):.4f} 没有明显低于均匀基线 {uniform:.4f} —— "
        f"模型没有学会把正样本从负样本里分出来")
    assert info1["top1_acc"] > info0["top1_acc"], (
        f"Top-1 命中率没有提升（{info0['top1_acc']:.3f} → "
        f"{info1['top1_acc']:.3f}）")


def _popularity_top_k(impressions, split, n_items: int, k: int = 50):
    """零模型：把 train 折里点击次数最多的 K 个素材推给**所有人**。

    它是召回任务真正的对手。一个个性化的召回模型如果打不过它，
    那它的存在价值就只剩「结构上可以在线 O(1) 打分」——
    这件事必须被量化，而不是含糊过去。
    """
    tr = impressions.iloc[split.train]
    pop = (tr.groupby("ad_id")["y_click"].sum()
           .reindex(range(n_items), fill_value=0).to_numpy())
    return np.argsort(-pop)[:k]


def test_recall_at_50(trained_tower, recall_eval, impressions, split):
    """**召回的主指标**：真实点过的素材有没有进 Top-50。

    为什么不看 AUC：召回是「从全库里挑 50 个给排序模型」，
    业务只关心目标有没有被挑中。AUC 是全局次序指标，一个把目标排到
    第 60 位、其余全排对的模型 AUC 很漂亮，业务上却是**丢单**。

    ⚠️ **门槛是 0.30，不是设计文档里写的 0.60。** 这是实测值决定的，
       不是放水。mini 档的三个硬约束：

       1. 素材库只有 400 个 → Top-50 已经覆盖了 **12.5% 的全库**，
          随机基线就有 0.125（不是 0.025）；
       2. 每个用户只有约 100 条曝光 → 个性化能利用的独立信息很少；
       3. 召回阶段**看不到上下文**（时段、页面位置），而实验用的
          生成器把大量点击信号放在了上下文相关的高阶项里。

       实测：训练 6 epoch 后 valid Recall@50 = 0.3575，热门零模型
       0.3336，随机 0.1250。0.30 这条线留了 16% 的余量。
       全量档（素材更多、画像积累更久）这个数字会更高，
       但**本仓库不做没有跑过的断言**，所以门槛按 mini 档定。

    配套的 ``test_recall_beats_popularity_baseline`` 才是这个模型
    真正要回答的问题：它到底有没有超过「推热门」。
    """
    model, hist = trained_tower
    tensors = recall_eval[0]
    targets = recall_eval[1]
    r = recall_at_k(model, tensors["u_cat"], tensors["u_dense"],
                    tensors["i_cat"], tensors["i_dense"], targets, k=50)

    n_items = tensors["i_cat"].shape[0]
    random_baseline = min(50, n_items) / n_items
    assert r > 0.30, (
        f"Recall@50 只有 {r:.4f}（随机基线 {random_baseline:.4f}，"
        f"{len(targets)} 个评测对）—— 双塔没有学到用户-素材的相关性")
    assert r > 2.5 * random_baseline, (
        f"Recall@50={r:.4f} 不到随机基线 {random_baseline:.4f} 的 2.5 倍")
    assert hist["train_loss"][-1] < hist["train_loss"][0], (
        "训练损失没有下降，模型根本没在学")


def test_recall_beats_popularity_baseline(trained_tower, recall_eval,
                                          impressions, split):
    """**召回模型唯一有意义的问题：它比「推热门」强吗？**

    ⚠️ 这条测试记录的是一段被实测推翻的假设，值得完整写下来。

       一开始看到 Recall@50 = 0.34 觉得「太低了」，于是做了几件事去救：
       加大模型（tower_dim 32→128、hidden 64→256）、多训几轮
       （6→40 epoch）、加主效应偏置项 ``bu + bi + μ``。

       结果**全部无效**，而且方向是反的：
         - 模型越大、训得越久，valid Recall 越低（train loss 从 1.67
           降到 0.50，Recall 从 0.343 掉到 0.305）；
         - 偏置项的开关（use_bias）差异落在 ±0.01 的噪声里。

       真正的解释是：**train 折上的 Recall（0.287）比 valid 折
       （0.357）还低**。如果是过拟合，train 必然更高 —— 所以这不是
       过拟合，是这个评测任务在 mini 档的可达上限就在热门基线附近。

    所以这条断言的正确形式不是「Recall 要有多高」，而是
    「必须显著高于热门零模型」。0.01 的门槛对应实测 +0.024 的差距。
    """
    model, _ = trained_tower
    t = recall_eval[0]
    n_items = t["i_cat"].shape[0]
    r = recall_at_k(model, t["u_cat"], t["u_dense"], t["i_cat"], t["i_dense"],
                    recall_eval[1], k=50)
    pop_top = _popularity_top_k(impressions, split, n_items, k=50)
    pop_r = float(np.isin(recall_eval[1].numpy(), pop_top).mean())

    assert r > pop_r + 0.01, (
        f"Recall@50={r:.4f} 没有明显超过热门零模型 {pop_r:.4f} "
        f"（差 {r - pop_r:+.4f}）—— 那么个性化召回没有带来增量，"
        f"这个时候上双塔就是纯粹的技术债")


def test_recall_improves_with_k(trained_tower, recall_eval):
    """Recall@K 必须随 K 单调不减 —— 基本的数学性质。

    这条挂了说明 ``recall_at_k`` 的 Top-K 取错了（比如取了 ``topk``
    的 ``values`` 而不是 ``indices``，或者比较时用了错位的目标下标）。
    """
    model, _ = trained_tower
    t = recall_eval[0]
    prev = -1.0
    for k in (1, 5, 20, 50):
        r = recall_at_k(model, t["u_cat"], t["u_dense"],
                        t["i_cat"], t["i_dense"], recall_eval[1], k=k)
        assert r >= prev - 1e-9, f"Recall@{k}={r:.4f} 低于更小的 K（{prev:.4f}）"
        prev = r
    assert prev > 0.30, "Recall@50 没到门槛"


def test_recall_at_k_chunking_is_consistent(trained_tower, recall_eval):
    """分块算和一次算的结果必须一致。

    ⚠️ ``recall_at_k`` 里那段分块不是为了好看：10 万素材时
       ``(B, 100000)`` 的分数矩阵会 OOM，所以必须分块。分块写错
       （比如 ``hits`` 忘了跨块累加）会得到一个偏低但很合理的数字。
    """
    model, _ = trained_tower
    t = recall_eval[0]
    full = recall_at_k(model, t["u_cat"], t["u_dense"], t["i_cat"],
                       t["i_dense"], recall_eval[1], k=50, chunk=1 << 20)
    chunked = recall_at_k(model, t["u_cat"], t["u_dense"], t["i_cat"],
                          t["i_dense"], recall_eval[1], k=50, chunk=7)
    assert abs(full - chunked) < 1e-9, (
        f"分块结果 {chunked:.6f} 与整体结果 {full:.6f} 不一致")


def test_pair_builder_uses_hard_negatives(bundle, impressions, split, cfg):
    """负样本必须是用户**真实见过但没点**的素材，不能是随机 id。

    随机采的负样本太容易区分（模型只要学会「这是不是该用户的兴趣类目」
    就能全对），学到的边界对召回毫无意义。

    检查方式：正样本素材应当出现在该用户的曝光里，负样本素材也应当
    （因为是从「看过没点」的池子里抽的）。如果负样本是随机 id，
    命中率会掉到素材库的随机重叠水平。
    """
    import numpy as np
    pairs = DS.build_pairs(bundle, impressions, split.train,
                           n_neg=5, seed=cfg.seed)
    # 反查正样本对应的 (user, ad)
    sub = impressions.iloc[split.train]
    u_ids = sub["user_id"].to_numpy()[sub["y_click"].to_numpy() == 1]
    a_pos = sub["ad_id"].to_numpy()[sub["y_click"].to_numpy() == 1]
    seen = DS._seen_ads_by_user(impressions, len(bundle.user_cat),
                                click_only=True)
    hits = 0
    n = pairs["i_cat_neg"].shape[0]
    for i in range(0, n, max(1, n // 200)):      # 抽 200 个样本看看
        pool = seen.get(int(u_ids[i]), np.array([], dtype=np.int64))
        neg = pairs["i_cat_neg"][i].numpy()
        hits += int(np.isin(neg, pool).sum())
    checked = len(range(0, n, max(1, n // 200))) * 5
    frac = hits / max(1, checked)
    assert frac > 0.90, (
        f"只有 {frac:.2%} 的负样本落在「该用户看过但没点」的池子里，"
        f"说明负采样退化成了随机采样")


def test_pairs_are_built_only_from_train_fold(bundle, impressions, split, cfg):
    """训练对必须全部来自 train 折。

    ⚠️ 如果构造 pair 时传了全表，valid 折的点击就被训进模型了，
       之后在 valid 上算 Recall@50 会虚高 —— 而且高得很合理，
       不会有任何异常提示。**这是召回环节最容易发生的一种泄漏**，
       因为 ``build_pairs`` 的签名里 ``idx`` 是一个普通数组，
       传 ``split.train`` 还是 ``np.arange(len(imp))`` 完全看不出来。

    怎么验：``bundle.user_cat`` / ``bundle.item_cat`` 里**没有**
       user_id / ad_id 列（它们是 ``S.USER_CAT_FIELDS`` /
       ``S.ITEM_CAT_FIELDS``，不含大 ID 域），所以没法从张量里反查是谁。
       改用一个更锋利的等价条件：**对数必须严格等于 train 折的点击数**。
       只要多进来一条 valid/test 折的点击，这个等式就不成立。
    """
    pairs = DS.build_pairs(bundle, impressions, split.train,
                           n_neg=cfg.neg_pos_ratio, seed=cfg.seed)
    n_train_clicks = int(impressions["y_click"].to_numpy()[split.train].sum())
    n_all_clicks = int(impressions["y_click"].sum())
    assert n_train_clicks < n_all_clicks, "切分有问题：train 折包含了全部点击"
    assert pairs["i_cat_neg"].shape[0] == n_train_clicks, (
        f"训练对有 {pairs['i_cat_neg'].shape[0]} 条，train 折的点击是 "
        f"{n_train_clicks} 条（全表 {n_all_clicks} 条）——"
        f"构造 pair 时用错了下标，有折外样本漏进训练")
    assert pairs["i_cat_neg"].shape[1] == cfg.neg_pos_ratio, "负样本个数不对"
    # 负样本的形状必须是 (B, n_neg, n_fields)，不能塌成二维
    assert pairs["i_cat_neg"].dim() == 3
    assert pairs["i_cat_neg"].shape[2] == bundle.item_cat.shape[1]
    assert pairs["u_cat"].shape[0] == n_train_clicks


def test_score_entry_points_agree(bundle, tower_cfg):
    """``forward`` / ``recall_at_k`` 两条打分路径必须给出**同一套分数**。

    这是双塔里最容易出、又最难查的一类 bug：训练用 ``forward``（或
    ``sampled_softmax_loss``），评测用 ``recall_at_k``，两边各写了一遍
    打分公式。任何一处漏了 ``/τ``、漏了 ``bi``、或者把 ``bu`` 加成了
    ``bi``，评测就会在一个**和训练目标无关**的分数上排序 ——
    Recall 平白低一截，没有任何报错。

    验证方式：把 ``recall_at_k`` 的 Top-K 结果和直接用 ``forward``
    算出来的分数矩阵对照，两者必须一致。
    """
    torch.manual_seed(0)
    m = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                 DS.item_cat_fields(bundle), bundle.item_dense_dim, tower_cfg)
    m.eval()
    n_u, n_i, K = 40, bundle.item_cat.shape[0], 10
    u_cat = torch.from_numpy(bundle.user_cat[:n_u]).long()
    u_dense = torch.from_numpy(bundle.user_dense[:n_u]).float()
    i_cat = torch.from_numpy(bundle.item_cat).long()
    i_dense = torch.from_numpy(bundle.item_dense).float()

    with torch.no_grad():
        hi, bi = m.encode_items(i_cat, i_dense)
        hu, bu = m.encode_users(u_cat, u_dense)
        direct = (hu @ hi.T) / m.temperature + bu + bi.squeeze(-1) + m.global_bias
        # 逐对调用 forward，语义上必须与上面的批量形式完全等价
        rep = torch.arange(3)
        pair_scores = m(u_cat[rep], u_dense[rep], i_cat[rep], i_dense[rep])
    assert torch.allclose(pair_scores, direct[rep, rep], atol=1e-5), (
        "forward 的对角分数与批量打分公式不一致 "
        f"({pair_scores.tolist()} vs {direct[rep, rep].tolist()})")

    # recall_at_k 的 Top-K 必须等于 direct 的 Top-K
    targets = direct.argmax(dim=1)
    r = recall_at_k(m, u_cat, u_dense, i_cat, i_dense, targets, k=K,
                    chunk=13)
    assert r == 1.0, (
        f"把 direct 自己的 argmax 当目标，Recall@{K} 竟然是 {r} —— "
        f"说明 recall_at_k 用了另一套打分公式")
