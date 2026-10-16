"""双塔召回模型：user tower / item tower，只用内积组合。

**双塔存在的唯一理由：在线只算一次内积。**

排序模型（DeepFM）把用户特征和素材特征拼在一起喂 DNN，精度高，但
**每个「用户 × 素材」组合都要单独前向一次**。2000 个素材 × 每秒几万个
请求，这个计算量在线扛不住。

双塔把计算拆成两半：

    离线：item tower 把全量素材编码成向量，灌进向量检索库（FAISS 之类）
    在线：user tower 编码当前用户一次 → 向量检索库做 ANN → Top-K

代价是表达能力：两个塔直到最后一步才见面，只能用内积组合。
所以双塔不做精排，只做**召回**（2000 → Top-50），精排交给 DeepFM。
这就是「召回-排序」两级架构里双塔的位置。

⚠️ **user tower 的输入里绝不能有任何跨域特征。**

这是双塔最常见的实现错误，而且错了照样能训、指标还挺好看 ——
因为离线评测时你手上有全部特征，随手就把 ``user_x_category_ctr``
这种「用户 × 素材」的交叉特征塞进了 user tower。

后果是：用户向量开始依赖具体素材，同一个用户对不同素材有不同的向量，
**离线预算 item 向量这件事直接失效**，在线必须对每个 (u, i) 重算一遍 ——
双塔退化成了一个更差的排序模型，而且没有任何报错提示你。

``test_user_tower_has_no_cross_features`` 就是拦这个的。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.layers import EmbeddingBank, MLP


@dataclass
class TwoTowerConfig:
    emb_dim: int = 8
    tower_hidden: tuple = (128, 64)
    tower_dim: int = 64
    dropout: float = 0.2
    temperature: float = 0.07
    # 是否给分数加主效应偏置项，见 TwoTower.forward 的长注释
    use_bias: bool = True


class Tower(nn.Module):
    """单塔：类别域 + 数值域 → (L2 归一化向量, 主效应标量)。

    **为什么除了向量还要吐一个标量？**

    纯内积模型在数学上表达不了主效应：``<h_u, h_i>`` 里没有
    「这个用户本来就爱点」或者「这个素材本来就热门」的位置 ——
    L2 归一化又把模长（也就是「强度」）整个丢掉了。
    补上 ``bu + bi + μ`` 之后，主效应归标量项，向量只需要负责
    **真正的交互**。这是矩阵分解 / YouTube DNN 的标准做法。

    在线代价为零：``bi`` 随 item 向量一起离线预计算，``bu`` 每个请求
    算一次，没有增加任何 per-item 在线计算。

    ⚠️ **但它解决不了 mini 档的召回问题，这一点必须如实记下来。**

       原本的假设是「主效应挤占了交互项的容量，所以 Recall 上不去」。
       实测否定了这个假设：400 素材 / 4000 用户的数据上，
       ``use_bias=True/False`` 的 Recall@50 分别是 0.3366 / 0.3431 ——
       差异落在噪声里（±0.01），加上偏置并不更好。

       真正的原因是任务本身的**可达上限**就在那里：
       「最热门的 50 个素材推给所有人」这个零模型就已经拿到 0.334。
       而 mini 档每个用户只有约 100 条曝光，素材库又只有 400 个
       （Top-50 覆盖了 12.5% 的全库），个性化能挤出来的增量非常有限。
       详见 ``tests/test_04_recall.py::test_recall_beats_popularity_baseline``。

       保留 ``use_bias`` 开关的意义在于：它把这个对照实验固化下来了，
       而不是留一句「加上应该会更好」的口头结论。
    """

    def __init__(self, cat_fields, dense_dim: int, tcfg: TwoTowerConfig):
        super().__init__()
        self.bank = EmbeddingBank(cat_fields, dense_dim,
                                  emb_dim=tcfg.emb_dim, dropout=tcfg.dropout)
        in_dim = self.bank.n_total_fields * tcfg.emb_dim + dense_dim
        self.mlp = MLP(in_dim, tuple(tcfg.tower_hidden), tcfg.dropout)
        self.proj = nn.Linear(self.mlp.out_dim, tcfg.tower_dim)
        self.use_bias = tcfg.use_bias
        if self.use_bias:
            # 零初始化：t=0 时偏置项不贡献任何分数，训练从纯内积起步
            self.bias_head = nn.Linear(self.mlp.out_dim, 1)
            nn.init.zeros_(self.bias_head.weight)
            nn.init.zeros_(self.bias_head.bias)

    def forward(self, cat: torch.Tensor,
                dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embs = self.bank(cat, dense)
        B = cat.shape[0]
        z = torch.cat([e.reshape(B, -1) for e in embs] + [dense], dim=1)
        h = self.mlp(z)
        # ⚠️ 必须 L2 归一化。否则内积会退化成「谁的向量模长大谁赢」，
        #    而模长和用户的活跃度正相关 —— 模型会学会「把热门商品的向量
        #    整体放大」这种平凡解，而不是学相关性。
        v = F.normalize(self.proj(h), dim=-1)
        b = self.bias_head(h) if self.use_bias else torch.zeros(
            B, 1, device=v.device, dtype=v.dtype)
        return v, b


class TwoTower(nn.Module):
    """双塔召回。分数 = <h_u, h_i> / τ。"""

    def __init__(self, user_cat_fields, user_dense_dim: int,
                 item_cat_fields, item_dense_dim: int,
                 tcfg: TwoTowerConfig | None = None):
        super().__init__()
        self.tcfg = tcfg or TwoTowerConfig()
        self.user_tower = Tower(user_cat_fields, user_dense_dim, self.tcfg)
        self.item_tower = Tower(item_cat_fields, item_dense_dim, self.tcfg)
        # 可学习的温度，初始为配置值。log 参数化保证恒正。
        self.log_tau = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(self.tcfg.temperature)))))
        # 全局偏置 μ：点击率本身就不高，模型需要先学到「平均水平」，
        # 再去学分用户/分素材的偏移。没有它的话这两件事会互相抵消。
        self.global_bias = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp_min(1e-3)

    # ------------------------------------------------------------------
    # ⚠️ 三个打分入口的**唯一**区别只是要不要算 bias，数学上必须完全一致。
    #    拆开写是因为三处的批量形状不同（成对 / 用户单侧 / 素材单侧），
    #    但一旦哪一处忘了加 bias，训练和评测就会用两套分数 ——
    #    这种错会让 Recall 莫名其妙地低，而且不报任何错。
    #    ``test_score_entry_points_agree`` 就是钉这一点的。
    # ------------------------------------------------------------------
    def encode_users(self, cat, dense) -> torch.Tensor:
        """在线路径：用户向量。带 bias 时是 ``(向量, 偏置)`` 两元组。"""
        return self.user_tower(cat, dense)

    def encode_items(self, cat, dense) -> torch.Tensor:
        """离线路径：全量素材跑一次，结果存进向量检索库。

        ⚠️ 带 bias 时 ``bi`` 也要一起落库 —— 检索库只做内积，返回
           top-N（N 远大于 K）之后再用 ``bi`` 重排。这是标准的
           「召回-重排」两步，代价可以忽略。
        """
        return self.item_tower(cat, dense)

    def forward(self, u_cat, u_dense, i_cat, i_dense) -> torch.Tensor:
        """返回 (B,) 的成对分数。

        ``score = <h_u, h_i>/τ + b_u + b_i + μ``

        交互项只允许内积（+ 温度），偏置项是三个标量 —— 它们
        **不依赖对方那侧的输入**，所以素材侧仍然是离线可预计算的。
        这正是「可预计算」这条约束允许的最大表达力。
        """
        hu, bu = self.encode_users(u_cat, u_dense)
        hi, bi = self.encode_items(i_cat, i_dense)
        s = (hu * hi).sum(dim=-1, keepdim=True) / self.temperature
        return (s + bu + bi + self.global_bias).squeeze(-1)


def sampled_softmax_loss(u_cat, u_dense, i_cat_pos, i_dense_pos,
                         i_cat_neg, i_dense_neg, model: TwoTower,
                         temperature: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, dict]:
    """采样 softmax：正样本 1 个 + 负样本 N 个，做 N+1 分类。

    为什么不用 BCE：BCE 把每个负样本独立看待，学到的只是「正样本分数
    比这一个负样本高」。softmax 是**竞争性**的：它要求正样本在同一个
    softmax 里胜过所有负样本。召回场景下用户只会看到 Top-K，
    这一点至关重要 —— 排序质量取决于相对次序，而不是绝对分数。

    ⚠️ 负样本必须是**同一批请求里其他用户的真实正样本**（in-batch
       negatives），不能随便随机采。随机采的负样本太容易区分，
       模型学到的边界没有意义（trained on easy negatives）。
       这里由调用方传真实曝光过的 pair 进来，而不是随机 id。
    """
    tau = model.temperature if temperature is None else temperature
    hu, bu = model.encode_users(u_cat, u_dense)             # (B, d), (B, 1)
    hi_pos, bi_pos = model.encode_items(i_cat_pos, i_dense_pos)

    # ⚠️ 负样本是 (B, M, ...) 的三维张量，而 EmbeddingBank 只接受 (N, n_fields)
    #    的二维输入（它有一条「列数必须等于字段数」的校验，专门拦索引错位）。
    #    所以要先摊平成 (B·M, ...) 编码，再折回 (B, M, d)。直接传三维会报错，
    #    但更容易踩的是「传了正确形状、列数恰好也等于字段数」的巧合 ——
    #    那会静默地把字段顺序错位。
    B, M = i_cat_neg.shape[0], i_cat_neg.shape[1]
    hi_neg, bi_neg = model.encode_items(
        i_cat_neg.reshape(B * M, -1), i_dense_neg.reshape(B * M, -1))
    hi_neg = hi_neg.reshape(B, M, -1)                       # (B, M, d)
    bi_neg = bi_neg.reshape(B, M, 1)                        # (B, M, 1)

    # einsum 而不是循环：B=1024、M=5、d=64 时循环版慢 20 倍
    pos_logit = (hu * hi_pos).sum(-1, keepdim=True) / tau            # (B, 1)
    neg_logit = torch.einsum("bd,bmd->bm", hu, hi_neg) / tau         # (B, M)
    if model.tcfg.use_bias:
        # 偏置项对**同一行的所有候选**是同一个 bu、同一个 μ，
        # 但 bi 各不相同 —— 热门素材的 bi 高，会让它在 softmax 里
        # 天然占优。这正是我们想要的：主效应不该挤占交互项的容量。
        pos_logit = pos_logit + bu + bi_pos + model.global_bias
        neg_logit = neg_logit + bu + bi_neg.squeeze(-1) + model.global_bias
    logits = torch.cat([pos_logit, neg_logit], dim=1)                # (B, M+1)

    target = torch.zeros(logits.shape[0], dtype=torch.long,
                         device=logits.device)
    loss = F.cross_entropy(logits, target)

    with torch.no_grad():
        rank = (logits > pos_logit).sum(dim=1)
        top1 = float((rank == 0).float().mean())
    return loss, {"top1_acc": top1, "temperature": float(tau)}


def train_two_tower(model: TwoTower, pairs: dict, *, epochs: int = 8,
                    batch_size: int = 1024, lr: float = 3e-3,
                    weight_decay: float = 1e-5, seed: int = 42,
                    grad_clip: float = 1.0, verbose: bool = False):
    """双塔训练循环。

    为什么不复用 ``src/models/train.py`` 的通用循环：那个循环是**逐样本**
    的（每个样本一个标签），而双塔是**逐样本一组候选**（1 正 + N 负做
    softmax）。硬套的话要造一个假的标签张量，反而更绕。

    学习率比排序模型大一个量级（3e-3 vs 1e-3）：召回任务的梯度信号更稠密
    （每个 batch 有 batch_size 个正样本参与竞争，而不是 batch_size × 3% 个），
    而且 L2 归一化之后的向量内积尺度很小，收敛本来就慢。
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    n = pairs["u_cat"].shape[0]
    losses: list[float] = []

    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(n, generator=g)
        ep_loss = []
        for s in range(0, n, batch_size):
            rows = perm[s:s + batch_size]
            batch = {k: v[rows] for k, v in pairs.items()}
            loss, _ = sampled_softmax_loss(**batch, model=model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss.append(float(loss.detach()))
        losses.append(float(sum(ep_loss) / max(1, len(ep_loss))))
        if verbose:
            print(f"    tower epoch {ep + 1:2d}  loss={losses[-1]:.5f}  "
                  f"tau={float(model.temperature):.4f}")
    return {"train_loss": losses, "temperature": float(model.temperature)}


@torch.no_grad()
def recall_at_k(model: TwoTower, u_cat, u_dense, i_cat, i_dense,
                targets: torch.Tensor, k: int = 50,
                chunk: int = 4096) -> float:
    """Recall@K：真实目标素材是否落进了每个用户的 Top-K。

    ⚠️ 分块算。user_batch × n_items 的分数矩阵在 10 万素材时是
       (B, 100000)，一次全算会 OOM。分成小块、每块只保留 Top-K 就够了。

    ⚠️ 打分的公式必须和 ``TwoTower.forward`` **逐项一致**（内积/τ + 三个
       偏置）。少加一项的话，训练时模型在优化一套分数、评测时却在看
       另一套 —— Recall 会平白低一截，而且没有任何报错。

    ⚠️ **必须切到 eval 模式。** ``@torch.no_grad()`` 关掉的只是梯度，
       **不关 dropout**。``train_two_tower`` 结束后模型停在 train 模式，
       直接评测的话塔里的 dropout 每次前向都在随机丢神经元 ——
       同一个模型同一份数据，两次调用能得到不同的 Recall（实测差 0.008），
       而且分数被随机性稀释后整体偏低。这是一个不报错、但让指标
       既偏低又不可复现的坑，和「忘了 model.eval()」是同一类问题。
    """
    was_training = model.training
    model.eval()
    try:
        return _recall_at_k_eval(model, u_cat, u_dense, i_cat, i_dense,
                                 targets, k, chunk)
    finally:
        model.train(was_training)


@torch.no_grad()
def _recall_at_k_eval(model: TwoTower, u_cat, u_dense, i_cat, i_dense,
                      targets: torch.Tensor, k: int, chunk: int) -> float:
    n_items = i_cat.shape[0]
    k = min(k, n_items)
    hits, total = 0, 0

    hi, bi = model.encode_items(i_cat, i_dense)          # (M, d)、(M, 1) 离线算一次
    bi = bi.squeeze(-1)                                  # (M,)
    for s in range(0, u_cat.shape[0], chunk):
        e = min(s + chunk, u_cat.shape[0])
        hu, bu = model.encode_users(u_cat[s:e], u_dense[s:e])   # (b, d)、(b, 1)
        scores = hu @ hi.T                                       # (b, M)
        if model.tcfg.use_bias:
            scores = (scores / model.temperature + bu + bi
                      + model.global_bias)
        else:
            scores = scores / model.temperature
        topk = scores.topk(k, dim=1).indices                     # (b, k)
        tgt = targets[s:e].unsqueeze(1)                          # (b, 1)
        hits += int((topk == tgt).any(dim=1).sum())
        total += e - s
    return hits / max(1, total)
