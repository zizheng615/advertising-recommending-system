"""延迟反馈修正：Chapelle 的 DFM（Delayed Feedback Model）。

**问题是什么**

转化有延迟。用户在 10:00 点了广告，18:00 才下单。如果模型在 12:00 训练，
这条样本的标签是 0 —— 但它是**假负样本**，迟早会变成 1。

实测（本仓库生成器，24h 窗口）：假负率约 21%。也就是说五分之一的真实
转化被标成了负样本。模型于是学到「凡是最近几小时的曝光都别信」，
把大量本该出高价的流量打成了低价。

**为什么不能靠「等久一点」解决**

等 168 小时确实能把假负率压到 2%，但业务等不起 —— 广告投放要求小时级
甚至分钟级的反馈闭环。而且越近的样本信息越新鲜，全丢掉反而损失更大。
所以必须在「有偏的短窗口标签」上做修正。

**DFM 怎么修**

引入一个额外的标量头 λ(x)：观察到转化的延迟服从参数为 λ 的指数分布。
注意 λ 是与 CVR 头 p(x) 分开的、自己学出来的。

对一条观测时长为 e 的曝光，真实转化概率是 p，那么：

    P(转化且已观测到) = p · (1 − e^{−λe})
    P(未观测到转化)   = (1 − p) + p · e^{−λe}     ← 里头的 p·e^{−λe}
                                                    就是「转化了但还没被看到」
    L = −y·log[p(1−e^{−λe})] − (1−y)·log[(1−p) + p·e^{−λe}]

推理时**只取 p(x)**，天然就是无偏的 CVR —— 那些假负样本的
``p·e^{−λe}`` 项在训练时就已经被正确地「占用」掉了，不会再被推成 0。

**两个必须夹住的量（都踩过）**

⚠️ ``(λ·e)`` 必须 ``clamp(max=30)``：曝光时长 e 能到 168，λ 训练中可以
   冲到很大，``exp(-λe)`` 会下溢到 0，紧接着 ``log(0) = -inf``，
   反向传播 0×inf = NaN。30 对应 e^{-30} ≈ 1e-13，已经完全够了。

⚠️ ``p`` 和 ``p·e^{−λe}`` 相关的位置都要 ``clamp_min(1e-7)`` 再取 log。
   负样本项的括号 ``(1-p) + p·e^{−λe}`` 在 p≈1 且 λe 很大时会趋近 0，
   不夹就是 ``log(0) = -inf``。

**对照方案**

``elapsed_as_feature``：把 ``elapsed_h`` 直接当特征喂给普通 CVR 模型，
20 行代码，能拿到 DFM 大约 70% 的效果。作为消融对照跑，
用来证明「额外建模 λ 到底值不值这些复杂度」。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.layers import EmbeddingBank, MLP, stack_embeddings

# λ·e 的上限。e^{-30} ≈ 9.4e-14，再大也只是把 exp 推向下溢。
LAMBDA_E_MAX = 30.0
PROB_FLOOR = 1e-7


class DFMModel(nn.Module):
    """在标准 CVR 模型上挂一个延迟率头。"""

    def __init__(self, cat_fields: list[tuple[str, int]], dense_dim: int,
                 cfg: Config, hidden: int = 64):
        super().__init__()
        self.cfg = cfg
        self.bank = EmbeddingBank(cat_fields, dense_dim,
                                  emb_dim=cfg.emb_dim, dropout=cfg.dropout)
        in_dim = self.bank.n_total_fields * cfg.emb_dim + dense_dim
        self.trunk = MLP(in_dim, (hidden, hidden), cfg.dropout)

        h = self.trunk.out_dim
        self.head_cvr = nn.Linear(h, 1)        # p(x)：真正要的那个
        self.head_lambda = nn.Linear(h, 1)     # λ(x)：辅助头，只为训练服务

        nn.init.xavier_uniform_(self.head_cvr.weight)
        nn.init.zeros_(self.head_cvr.bias)
        # λ 初始化为 ln(1/36)：对应「平均延迟约 36 小时」的先验，
        # 和数据的长尾量级一致。从 0 开始的话 e^{-λe} ≈ 1，
        # 训练初期所有负样本都被当成假负，loss 会先涨后跌，很不好调。
        nn.init.zeros_(self.head_lambda.weight)
        nn.init.constant_(self.head_lambda.bias, -3.58)

    def forward(self, cat: torch.Tensor, dense: torch.Tensor,
                elapsed_h: torch.Tensor) -> dict:
        embs = self.bank(cat, dense)
        B = cat.shape[0]
        z = torch.cat([stack_embeddings(embs), dense], dim=1)
        h = self.trunk(z)

        p = torch.sigmoid(self.head_cvr(h)).clamp_min(PROB_FLOOR)
        # softplus 保证 λ > 0；指数分布的率参数必须为正
        lam = F.softplus(self.head_lambda(h)).clamp_min(1e-6)
        lam_e = (lam * elapsed_h.reshape(-1, 1)).clamp(max=LAMBDA_E_MAX)
        return {"p": p, "lambda": lam, "lambda_e": lam_e,
                "not_observed": torch.exp(-lam_e)}

    def loss(self, out: dict, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """DFM 负对数似然。``y`` 是**有偏的短窗口标签**（含假负）。"""
        y = y.reshape(-1, 1).float()
        p = out["p"]
        q = out["not_observed"]             # e^{-λe} = P(延迟超过观测窗口)

        # 真阳性概率：转化了，而且来得及被观测到
        p_obs = (p * (1.0 - q)).clamp_min(PROB_FLOOR)
        # 观测为负的概率：没转化，或者转化了但还没到
        p_neg = ((1.0 - p) + p * q).clamp_min(PROB_FLOOR)

        nll = -(y * torch.log(p_obs) + (1.0 - y) * torch.log(p_neg))
        loss = nll.mean()
        with torch.no_grad():
            mean_lambda_e = float(out["lambda_e"].mean())
        return loss, {"loss": float(loss.detach()),
                      "mean_lambda_e": mean_lambda_e}


class ElapsedFeatureModel(nn.Module):
    """对照方案：把 elapsed 当普通特征，不做任何建模。

    跑这个是为了回答「DFM 那套额外复杂度到底值不值」。
    如果它已经能拿到 DFM 90% 的效果，那就不该上 DFM。
    """

    def __init__(self, cat_fields: list[tuple[str, int]], dense_dim: int,
                 cfg: Config, hidden: int = 64):
        super().__init__()
        self.bank = EmbeddingBank(cat_fields, dense_dim,
                                  emb_dim=cfg.emb_dim, dropout=cfg.dropout)
        in_dim = self.bank.n_total_fields * cfg.emb_dim + dense_dim + 1
        self.trunk = MLP(in_dim, (hidden, hidden), cfg.dropout)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, cat, dense, elapsed_h) -> dict:
        embs = self.bank(cat, dense)
        B = cat.shape[0]
        z = torch.cat([stack_embeddings(embs), dense,
                       elapsed_h.reshape(B, 1)], dim=1)
        logit = self.head(self.trunk(z))
        return {"p": torch.sigmoid(logit), "logit": logit}

    def loss(self, out: dict, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """普通 BCE —— 完全无视「标签可能是假的」这件事。

        这正是它作为对照的意义：让实验数据说话，
        而不是靠「DFM 听起来更对」来下结论。
        """
        loss = F.binary_cross_entropy_with_logits(
            out["logit"], y.reshape(-1, 1).float())
        return loss, {"loss": float(loss.detach())}


def oracle_cvr_from_labels(imp, window: int = 168) -> "object":
    """拿回补标签当 CVR 真值的上界参考（**仅评测**）。

    对应 ``y_conv_w168``：等满 168 小时之后的标签，假负率只剩约 2%。
    它仍然不是真值（还有 2% 假负），但足以当作「延迟问题被解决后
    我们能达到什么水平」的参照线。
    """
    import numpy as np
    return np.asarray(imp[f"y_conv_w{window}"])
