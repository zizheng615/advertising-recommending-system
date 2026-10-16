"""ZILN：两阶段预测「会不会付费」与「会付多少」。

模型做两件事，共用同一个编码器：

    阶段一（分类）：π(x) = P(金额 > 0 | x)          —— 会不会成为付费用户
    阶段二（回归）：ln(金额) | 金额>0  ~  N(μ(x), σ(x))  —— 付多少钱

最后合成的**条件期望**就是 eCPM 公式里那个 LTV：

    Ê[金额 | x] = π(x) · exp(μ(x) + σ(x)²/2)

⚠️ **最后一个公式是本文件最容易写错的地方，而且错了不报错。**

   写成 ``π · exp(μ)`` 得到的是条件**中位数**，不是条件期望。
   对数正态的均值是 ``exp(μ + σ²/2)``，σ=0.7 时这个修正因子是 1.28；
   σ=1.5 时是 3.1。少乘这一项会让 eCPM 系统性低估 30%~200%，
   而 AUC、Gini 这些排序指标**完全看不出来** —— 它们只关心次序。
   最后只有「聚合偏差」这个指标能抓到它，测试里专门盯着这个数。

**为什么两阶段而不是直接回归金额？**

90% 的用户金额是 0。直接用 MSE 回归的话，模型的最优解是「全都预测接近 0」，
梯度被 0 主导，完全学不到付费者内部的金额差异。拆成
「分不分类 + 对数尺度回归」之后，两个子问题各自的信噪比都高得多。

**为什么回归目标取对数？**

金额是右偏的长尾分布（中位数几十块，长尾能到几千）。在对数尺度上它
近似正态 —— 用高斯 NLL 就是「正确的似然」，而不是一个凑合的选择。
测试 ``test_positive_amount_is_conditionally_lognormal`` 验证的就是
生成器确实满足这个假设，否则 ZILN 从根上就是错的。

**为什么 σ 也要建模（而不是固定）？**

不同用户的金额不确定性不一样：新用户的历史信息少，预测区间应该更宽。
σ 学出来之后，PIT 检验（``pit_uniformity_pvalue``）才有意义 ——
它是「对数正态假设是否成立」最直接的检验，同时也是对生成器和损失函数
两边的同时验证。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.layers import EmbeddingBank, MLP, masked_mean, stack_embeddings


class ZILN(nn.Module):
    """零膨胀对数正态。输出 (π, μ, σ) 三个头。"""

    def __init__(self, cat_fields: list[tuple[str, int]], dense_dim: int,
                 cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.bank = EmbeddingBank(cat_fields, dense_dim,
                                  emb_dim=cfg.emb_dim, dropout=cfg.dropout)
        in_dim = self.bank.n_total_fields * cfg.emb_dim + dense_dim
        self.trunk = MLP(in_dim, (cfg.ziln_hidden, cfg.ziln_hidden), cfg.dropout)

        h = self.trunk.out_dim
        self.head_p = nn.Linear(h, 1)
        self.head_mu = nn.Linear(h, 1)
        self.head_log_sigma = nn.Linear(h, 1)

        nn.init.xavier_uniform_(self.head_p.weight)
        nn.init.zeros_(self.head_p.bias)
        nn.init.xavier_uniform_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_log_sigma.weight)
        # σ 的初始值直接对齐真实数据的尺度，省掉几百步的预热
        nn.init.constant_(self.head_log_sigma.bias, 0.0)

    def forward(self, cat: torch.Tensor, dense: torch.Tensor) -> dict:
        embs = self.bank(cat, dense)
        B = cat.shape[0]
        z = torch.cat([stack_embeddings(embs), dense], dim=1)
        h = self.trunk(z)

        logit_p = self.head_p(h)
        mu = self.head_mu(h)
        # ⚠️ σ 必须恒正，且不能塌到 0。
        #    ``softplus`` 保证恒正；clamp 下界防止 (y-μ)²/σ² 除零、
        #    以及 log(σ) → -inf；clamp 上界防止早期 σ 爆炸把
        #    NLL 的第一项（log σ）压过第二项，模型退化成「只预测均值」。
        sigma = F.softplus(self.head_log_sigma(h)).clamp(
            self.cfg.sigma_min, self.cfg.sigma_max)

        return {"logit_p": logit_p, "p": torch.sigmoid(logit_p),
                "mu": mu, "sigma": sigma}

    # ------------------------------------------------------------------
    def loss(self, out: dict, y: torch.Tensor
             ) -> tuple[torch.Tensor, dict]:
        """ZILN 损失 = 分类 BCE + 正样本上的高斯 NLL。

        ``L = BCE(π, 1{y>0})  +  (1/Σ1{y>0}) Σ_{y>0} [ln σ + ½ln2π + (ln y − μ)²/(2σ²)]``

        混合分布的 NLL 自然分解成这两项 —— 这是 ZILN 最漂亮的地方：
        不需要任何手工权重，两个子问题各管各的，量级也自动可比。

        ⚠️ 一个 batch 里可能**一个正样本都没有**（正率 0.3%、batch 512 时
           概率约 20%）。这时第二项的分母是 0，朴素实现会得到 NaN，
           然后整个网络瞬间全变成 NaN，而且只在特定 batch 上偶发。
           ``masked_mean`` 就是干这个的。
        """
        pos = y > 0
        bce = F.binary_cross_entropy_with_logits(out["logit_p"], pos.float())

        if int(pos.sum()) == 0:
            return bce, {"bce": float(bce.detach()), "nll": 0.0,
                         "n_pos": 0}

        log_y = torch.log(y.clamp_min(1e-6))
        nll_terms = (torch.log(out["sigma"])
                     + 0.5 * math.log(2 * math.pi)
                     + (log_y - out["mu"]) ** 2 / (2.0 * out["sigma"] ** 2))
        nll = masked_mean(nll_terms, pos)

        total = bce + nll
        return total, {"bce": float(bce.detach()), "nll": float(nll.detach()),
                       "n_pos": int(pos.sum())}

    # ------------------------------------------------------------------
    @staticmethod
    def expected_value(out: dict) -> torch.Tensor:
        """Ê[y] = π · exp(μ + σ²/2) —— eCPM 要用的就是它。

        ⚠️ 不是 ``π · exp(μ)``。后者是条件中位数，会系统性低估。
        """
        return out["p"] * torch.exp(out["mu"] + out["sigma"] ** 2 / 2.0)


def ziln_predict_arrays(out: dict) -> dict:
    """把模型输出转成 numpy，供指标计算使用。"""
    import numpy as np
    p = out["p"].detach().cpu().numpy().ravel().astype(np.float64)
    mu = out["mu"].detach().cpu().numpy().ravel().astype(np.float64)
    sigma = out["sigma"].detach().cpu().numpy().ravel().astype(np.float64)
    return {"p": p, "mu": mu, "sigma": sigma,
            "expected": p * np.exp(mu + sigma ** 2 / 2.0),
            "median": p * np.exp(mu)}
