"""模型层：唯一的一套特征编码器。

**为什么全项目只有一套编码器？**

职责描述里提到「双塔模型」「DeepFM」「ZILN」听起来是三个模型，实际上它们是
**同一个编码器 + 三个不同的头**：

    编码器 EmbeddingBank  →  ├─ FM + DNN → CTR / CTCVR 双头（DeepFM）
                            ├─ 内积     → 召回分数（双塔）
                            └─ MLP      → p / μ / σ（ZILN）

这个统一带来两个好处：

1. **口径自洽**。所有模型看到的特征、嵌入维度、字段顺序完全一致，
   指标对比才是苹果对苹果，而不是「换个模型顺便换了特征」。
2. **消融可信**。LR / FM / DeepFM 用同一份代码、只切开关（``use_fm`` /
   ``use_dnn``），所以 ``AUC(DeepFM) − AUC(FM)`` 这个差值确实只反映
   「DNN 带来的高阶交叉能力」，不掺任何实现差异。

**FM 二阶项为什么必须用 trick？**

朴素写法是双重循环枚举字段对：

    sum_{i<j} <e_i, e_j>     —— O(F² · d)

用恒等式 ``sum_{i<j}<e_i,e_j> = 0.5 * (|Σe|² − Σ|e|²)`` 可以降到 O(F · d)。
F=52、d=8 时是 1300 倍左右的差距，而且字段越多差距越大。

⚠️ ``|Σe|² − Σ|e|²`` 在数学上恒非负，但浮点下可能算出 -1e-7。
   **绝不要开根号** —— sqrt(负数) = NaN，会把整个 batch 的 loss 变成 NaN，
   而且只在某些 batch 上偶发，极难定位。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 大基数 ID 域（user_id / ad_id / advertiser_id / campaign_id）。
# 这些域单独用更小的嵌入维 —— 见 EmbeddingBank 的说明。
LARGE_CARD_ID_FIELDS = {"user_id", "ad_id", "advertiser_id", "campaign_id"}


class EmbeddingBank(nn.Module):
    """把「类别矩阵 + 数值矩阵」编码成一组等长向量。

    设计取舍
    --------
    **所有字段共用同一个 emb_dim**，包括 user_id 这种百万级基数的域。

    为什么不让大 ID 域用更小的维度（配置里的 ``emb_dim_large``）？
    因为 FM 的二阶 trick 要求所有字段的嵌入维相同 —— 维度不齐就没法
    ``(Σe)² − Σe²``，只能退回双重循环。而大 ID 域的嵌入表本来就不大
    （10 万用户 × 8 维 = 80 万参数，3MB），省这点显存完全不值得牺牲
    一个 O(F·d) 的核心算子。配置项保留着，但默认与 emb_dim 相同。

    数值特征的处理
    --------------
    整个数值块用一层线性投影成**一个字段**的嵌入。这样做的好处是：
    数值特征也能参与 FM 的二阶交叉（数值×类别、数值×数值），
    而不是只能走 DNN 那条路。这也是 DeepFM 相比「one-hot + DNN」的
    实质改进之一。

    ⚠️ 类别索引必须由调用方保证落在 [0, cardinality) 内。
       ``nn.Embedding`` 越界不会报错，会读出垃圾内存，
       然后训练到第几百步才以 NaN loss 的形式爆出来。
       测试 ``test_bundle_cat_indices_in_range`` 就是拦这个的。
    """

    def __init__(self, cat_fields: list[tuple[str, int]], dense_dim: int,
                 emb_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.cat_fields = list(cat_fields)
        self.n_fields = len(cat_fields)
        self.emb_dim = emb_dim
        self.dense_dim = int(dense_dim)

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim) for _, card in cat_fields
        ])
        # 一阶（线性）项：每个字段的每个取值一个标量。这就是 FM 里那个
        # ``sum_i w_i x_i``，参数少、且学得到字段内的相对强弱。
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(card, 1) for _, card in cat_fields
        ])
        # 数值块作为一个字段
        self.dense_proj = (nn.Linear(self.dense_dim, emb_dim)
                           if self.dense_dim > 0 else None)
        self.dense_linear = (nn.Linear(self.dense_dim, 1)
                             if self.dense_dim > 0 else None)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for lin in self.linear_embeddings:
            nn.init.zeros_(lin.weight)
        if self.dense_proj is not None:
            # 数值块先压到小方差，避免训练初期 FM 二阶项被数值特征主导
            nn.init.xavier_uniform_(self.dense_proj.weight)
            nn.init.zeros_(self.dense_proj.bias)
        if self.dense_linear is not None:
            nn.init.zeros_(self.dense_linear.weight)
            nn.init.zeros_(self.dense_linear.bias)

    @property
    def n_total_fields(self) -> int:
        """参与 FM / 拼接的字段总数（类别域 + 数值块）。"""
        return self.n_fields + (1 if self.dense_proj is not None else 0)

    def forward(self, cat: torch.Tensor,
                dense: torch.Tensor | None = None) -> list[torch.Tensor]:
        """返回每个字段的嵌入向量列表，每个形状 (B, emb_dim)。"""
        if cat.dim() != 2 or cat.shape[1] != self.n_fields:
            raise ValueError(
                f"类别矩阵形状 {tuple(cat.shape)} 与字段定义不符（应为 (B, {self.n_fields})）")
        if cat.dtype != torch.long:
            cat = cat.long()

        out = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if self.dense_proj is not None:
            if dense is None:
                raise ValueError("bank 定义了数值投影，但 forward 没有传 dense")
            # ⚠️ 形状必须是 (B, emb_dim)，和其他字段**完全一致**。
            #    这里曾经多写了一个 ``.unsqueeze(1)`` 变成 (B, 1, emb_dim)，
            #    后果是 ``torch.stack(embs, dim=1)`` 直接报形状不一致 ——
            #    能报错的还算好的；若 emb_dim 恰好等于 1，它会静默通过，
            #    然后 FM 的二阶项把「批次维」当成「字段维」求和。
            out.append(self.dense_proj(dense))
        return [self.dropout(e) for e in out]


def fm_second_order(embs: list[torch.Tensor]) -> torch.Tensor:
    """FM 二阶项，返回 (B, 1)。用 trick 做到 O(F·d)。

    ``sum_{i<j}<e_i,e_j> = 0.5 * (|Σe_i|² − Σ|e_i|²)``

    ⚠️ 结果恒非负，但浮点可能微负 —— 不要开根号，也不要用 clamp 之外的
       任何「修正」去掩盖它。直接照原样加进 logit 就是正确的。
    """
    E = torch.stack(embs, dim=1)                 # (B, F, d)
    summed = E.sum(dim=1)                        # (B, d)
    sum_sq = (E * E).sum(dim=1)                  # (B, d)
    return 0.5 * (summed * summed - sum_sq).sum(dim=1, keepdim=True)


def first_order(linear_embs: list[nn.Embedding], cat: torch.Tensor,
                dense: torch.Tensor | None,
                dense_linear: nn.Linear | None = None) -> torch.Tensor:
    """FM 的一阶（线性）项，返回 (B, 1)。

    形式就是 ``sum_i w_i x_i``：类别域用一张 (card, 1) 的嵌入表当查表，
    数值域走一层线性。这一项让模型能表达「某个取值本身就偏正/偏负」，
    是 LR 的全部能力所在 —— 关掉 FM 和 DNN 之后剩下的就是这个。
    """
    out = torch.zeros(cat.shape[0], 1, device=cat.device, dtype=torch.float32)
    for i, lin in enumerate(linear_embs):
        out = out + lin(cat[:, i])
    if dense_linear is not None and dense is not None:
        out = out + dense_linear(dense)
    return out


class MLP(nn.Module):
    """标准的前馈塔。ReLU + Dropout，最后一层不加激活。"""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stack_embeddings(embs: list[torch.Tensor]) -> torch.Tensor:
    """把嵌入列表摊平成 (B, F·d)，供 DNN 使用。"""
    B = embs[0].shape[0]
    return torch.cat([e.reshape(B, -1) for e in embs], dim=1)


def logsigmoid_stable(logit: torch.Tensor) -> torch.Tensor:
    """log σ(x)，数值稳定版。

    ⚠️ 不要写 ``torch.log(torch.sigmoid(x))``：x 很负时 sigmoid 下溢到 0，
       log(0) = -inf，反向传播时 0 * inf = NaN。
       ``logsigmoid`` 内部做了 log-sum-exp 变形，不会下溢。
    """
    return F.logsigmoid(logit)


def masked_mean(x: torch.Tensor, mask: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """按掩码求均值，掩码全 0 时返回 0 而不是 NaN。

    ZILN 的对数正态项只在正样本上算，一个 batch 里可能一个正样本都没有
    （尤其是不平衡数据 + 小 batch）。这时候 mask.sum() == 0，
    朴素的 ``(x*mask).sum()/mask.sum()`` 会得到 0/0 = NaN，
    然后整个模型的参数在一瞬间全变成 NaN。
    """
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(eps)


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor,
                 sigma: torch.Tensor) -> torch.Tensor:
    """高斯负对数似然（去掉常数项 ½ln2π 之外的都保留，方便对齐公式）。"""
    return torch.log(sigma) + 0.5 * math.log(2 * math.pi) \
        + (y - mu) ** 2 / (2.0 * sigma ** 2)
