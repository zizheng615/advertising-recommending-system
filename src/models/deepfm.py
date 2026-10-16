"""DeepFM 排序模型（ESMM 双头）+ LR / FM 基线。

**为什么基线和主模型写在同一个类里？**

职责描述要求「DeepFM 深度挖掘低阶和高阶非线性交叉特征」。要证明这句话，
需要 ``AUC(DeepFM) > AUC(FM) > AUC(LR)``。但如果 LR / FM / DeepFM 是三个
各自实现的类，这个比较就掺了杂质 —— 特征处理差一点、初始化差一点、
学习率差一点，都能让差值变号，然后你根本分不清是「高阶交叉的价值」
还是「实现细节的差别」。

所以这里只写**一个** ``FactorizationMachineRanker``，用两个开关控制：

    use_fm=False  use_dnn=False   →  LR   （只有一阶项）
    use_fm=True   use_dnn=False   →  FM   （一阶 + 二阶）
    use_fm=True   use_dnn=True    →  DeepFM（一阶 + 二阶 + 高阶）

三者共用同一套嵌入表、同一个优化器、同一份数据、同一个随机种子。
差值是干净的。

**为什么用 ESMM 双头而不是直接预测 CVR？**

CVR 的定义是 P(转化 | 点击)，只有在点击样本上才有标签。直接拿点击样本
训一个 CVR 模型会遇到两个问题：

1. **样本选择偏差（SSB）**：点击样本是「用户愿意点」这个筛选后的子集，
   它的分布和全量曝光分布不一样。模型在点击集上学到的关系，
   推到全量曝光上会系统性偏。
2. **数据量少一个数量级**：CTR 5%，全量 100 万曝光只有 5 万点击，
   其中转化 3000 个。样本越少，越容易过拟合。

ESMM 的做法是在**全量曝光**上同时建两个任务：

    pCTR   = P(点击)           —— 全量样本都有标签
    pCTCVR = P(点击且转化)      —— 全量样本都有标签
    pCVR   = pCTCVR / pCTR     —— 推导出来的，不直接监督

两个任务的标签在全量样本上都可观测，于是既绕开了 SSB，
又让 CVR 那一路拿到了 20 倍的训练信号。这也是本项目里
``pCTCVR ≈ pCTR · pCVR`` 这条恒等式被反复断言的原因。

⚠️ 两个任务**共享输入、不共享参数**（见 ``FactorizationMachineRanker``
   的类文档）。共享主干是个看起来很自然的省事做法，实测会让
   两个任务互相拉扯、并把 pCVR 退化成一个常数偏移。

⚠️ ``pCVR = pCTCVR / pCTR`` 的除法必须夹住分母。pCTR 可以极小，
   不夹的话会得到 inf，然后反向传播把整个网络变成 NaN。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.layers import (EmbeddingBank, MLP, first_order,
                               fm_second_order, masked_mean,
                               stack_embeddings)


@dataclass
class DeepFMConfig:
    """模型结构开关。默认 = 完整 DeepFM。"""

    use_fm: bool = True
    use_dnn: bool = True
    use_esmm: bool = True          # 双头；关掉就退化成单头 CTCVR
    emb_dim: int = 8
    dnn_hidden: tuple = (128, 64)
    dropout: float = 0.2
    cvr_denom_eps: float = 1e-4
    # 点击样本上的 CVR 辅助损失权重。**不能取 0**，理由见 esmm_loss。
    cvr_aux_weight: float = 1.0
    click_weight: float = 1.0
    ctcvr_weight: float = 1.0

    @classmethod
    def as_lr(cls, cfg: Config) -> "DeepFMConfig":
        return cls(use_fm=False, use_dnn=False, emb_dim=cfg.emb_dim,
                   dnn_hidden=cfg.dnn_hidden, dropout=0.0)

    @classmethod
    def as_fm(cls, cfg: Config) -> "DeepFMConfig":
        return cls(use_fm=True, use_dnn=False, emb_dim=cfg.emb_dim,
                   dnn_hidden=cfg.dnn_hidden, dropout=0.0)

    @classmethod
    def as_deepfm(cls, cfg: Config) -> "DeepFMConfig":
        return cls(use_fm=True, use_dnn=True, emb_dim=cfg.emb_dim,
                   dnn_hidden=cfg.dnn_hidden, dropout=cfg.dropout)


class FactorizationMachineRanker(nn.Module):
    """两个**参数独立**的塔：CTR 塔与 CVR 塔，各自一阶 + 二阶 + 高阶。

    为什么是两个塔而不是一个共享主干加两个头？
    ------------------------------------------------
    最初写的是「一个共享编码器 + 两个头」。跑消融时冒出一串互相矛盾的
    现象，最后都指向同一个原因：

      1. 关掉 DNN 之后 ``head_ctr`` / ``head_cvr`` 都退化成 1 维的
         ``nn.Linear``，也就是**一个标量偏置**。于是
         ``logit_cvr − logit_ctr`` 是个常数 —— pCVR 只是 pCTR 的单调变换，
         两个任务的排序完全一样。测出来的 ``auc_cvr|click`` 在 0.55~0.62
         之间，看着像「学到了点东西」，其实是 CTR 排序的投影。
      2. 共享主干意味着两个任务在**抢同一组参数**。CTCVR 正样本在
         1:5 下采样后的训练集里只占 0.7%，单条正样本的损失是
         ``−log(0.007) ≈ 4.9``，而负样本只有 0.007 —— 正负梯度差 700 倍。
         几百条正样本就能把主干 logit 整体往下拽，CTR 头（权重 1.0）
         拉不回来。实测 LR 的 ``mean(p_ctr) = 0.0290``，真值 0.0528，
         整体低估 45%；而且各随机种子之间乱跳。

    修法是让两个任务**共享输入、不共享参数**：CTR 塔和 CVR 塔各有自己的
    嵌入表、一阶表、二阶嵌入和高阶 MLP。这正是 ESMM 原文的结构
    （它共享的是 embedding 层，两个塔各自往上长），也顺势把第 2 个
    问题消掉了 —— CTCVR 的梯度从此只能动 CVR 塔，碰不到 CTR 塔。

    代价是参数量和计算量约翻倍，换来的是消融实验里
    「DeepFM 的 CVR 塔 vs FM 的 CVR 塔」变成真正的苹果对苹果。

    三个变体仍然由 ``use_fm`` / ``use_dnn`` 两个开关切出来，
    且参数集合**严格嵌套**（LR ⊂ FM ⊂ DeepFM），差值依旧干净：

        use_fm=False  use_dnn=False   →  LR    （只有一阶项）
        use_fm=True   use_dnn=False   →  FM    （一阶 + 二阶）
        use_fm=True   use_dnn=True    →  DeepFM（一阶 + 二阶 + 高阶）
    """

    def __init__(self, cat_fields: list[tuple[str, int]], dense_dim: int,
                 mcfg: DeepFMConfig | None = None):
        super().__init__()
        self.mcfg = mcfg or DeepFMConfig()
        self.cat_fields = list(cat_fields)
        self.dense_dim = int(dense_dim)

        def _bank() -> EmbeddingBank:
            return EmbeddingBank(self.cat_fields, self.dense_dim,
                                 emb_dim=self.mcfg.emb_dim,
                                 dropout=self.mcfg.dropout)

        self.bank_ctr = _bank()
        self.bank_cvr = _bank()

        if self.mcfg.use_dnn:
            in_dim = (self.bank_ctr.n_total_fields * self.mcfg.emb_dim
                      + self.dense_dim)
            self.dnn_ctr = MLP(in_dim, tuple(self.mcfg.dnn_hidden),
                               self.mcfg.dropout)
            self.dnn_cvr = MLP(in_dim, tuple(self.mcfg.dnn_hidden),
                               self.mcfg.dropout)
            # 每个塔的高阶部分收成一个标量 logit，再**加**到一阶+二阶上。
            self.head_ctr = nn.Linear(self.dnn_ctr.out_dim, 1)
            self.head_cvr = nn.Linear(self.dnn_cvr.out_dim, 1)
        else:
            self.dnn_ctr = self.dnn_cvr = None
            self.head_ctr = self.head_cvr = None

        self._reset_dnn_heads()

    def _reset_dnn_heads(self) -> None:
        """把两个高阶头的权重清零。

        这样 ``DeepFM 在初始化那一刻 ≡ FM``（高阶那一路输出恰好是 0），
        三个变体在 t=0 的预测完全相同 —— 属于「嵌套模型」该有的性质，
        也顺手排除了「DeepFM 赢在初始化更好」这种解释。
        权重虽然是 0，但更新量是 ``上游梯度 × 上一层激活``，
        第一步就会离开 0，不会卡住。
        """
        for head in (self.head_ctr, self.head_cvr):
            if head is None:
                continue
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------
    def task_logit(self, bank: EmbeddingBank, dnn: MLP | None,
                   head: nn.Linear | None,
                   cat: torch.Tensor, dense: torch.Tensor
                   ) -> torch.Tensor:
        """单个塔的 logit：一阶 + (可选)二阶 + (可选)高阶。返回 (B, 1)。"""
        embs = bank(cat, dense)
        out = first_order(bank.linear_embeddings, cat, dense, bank.dense_linear)
        if self.mcfg.use_fm:
            out = out + fm_second_order(embs)
        if dnn is not None:
            z = torch.cat([stack_embeddings(embs), dense], dim=1)
            out = out + head(dnn(z))
        return out

    def forward(self, cat: torch.Tensor, dense: torch.Tensor) -> dict:
        """返回 dict：logit_ctr / logit_cvr / p_ctr / p_cvr / p_ctcvr。

        **用乘法参数化**：``pCTCVR = pCTR × pCVR``，两个塔各自 sigmoid 之后相乘。

        ⚠️ 这里换过一次实现，原因值得记下来。

           原来用「两个独立头 + 相除」：``pCVR = pCTCVR / pCTR``。
           数学上等价，实际上会被两个独立头之间**没有任何约束**这件事搞死 ——
           训练早期 pCTR 和 pCTCVR 是各学各的，pCTCVR > pCTR 的行非常多，
           相除之后全部被 clip 到 1.0。实测 ``mean(p_cvr) = 1.0000``，
           而且这一列还要进 eCPM，等于把出价全打成了最高价。
           更糟的是它**不报错**：CTR 和 CTCVR 两路的 AUC 都是正常的，
           只有「点击样本上的 CVR AUC」这一个指标会掉到 0.5000 ——
           如果没盯这个数，会一路带着一个常数 1 的出价系数上线。

           乘法参数化把 ``pCVR ≤ 1`` 变成**结构性保证**，不需要任何 clip。
        """
        logit_ctr = self.task_logit(self.bank_ctr, self.dnn_ctr,
                                    self.head_ctr, cat, dense)
        logit_cvr = self.task_logit(self.bank_cvr, self.dnn_cvr,
                                    self.head_cvr, cat, dense)

        p_ctr = torch.sigmoid(logit_ctr)
        p_cvr = torch.sigmoid(logit_cvr)
        return {"logit_ctr": logit_ctr, "logit_cvr": logit_cvr,
                "p_ctr": p_ctr, "p_cvr": p_cvr, "p_ctcvr": p_ctr * p_cvr}

    # ------------------------------------------------------------------
    def loss(self, out: dict, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """统一训练接口：``y`` 的第 0 列是点击、第 1 列是转化。

        这个两列约定把「多标签」压成一个张量，好处是通用训练循环
        （``src/models/train.py``）不需要为任何单个模型开特例 ——
        DFM / ZILN / 双塔都走同一个循环，指标才可比。
        """
        if y.dim() != 2 or y.shape[1] < 2:
            raise ValueError(f"ESMM 需要 (B, 2) 的标签，收到 {tuple(y.shape)}")
        return self.esmm_loss(out, y[:, 0:1], y[:, 1:2])

    def esmm_loss(self, out: dict, y_click: torch.Tensor,
                  y_conv: torch.Tensor,
                  click_weight: float | None = None,
                  ctcvr_weight: float | None = None,
                  cvr_weight: float | None = None) -> tuple[torch.Tensor, dict]:
        """ESMM 损失 = BCE(pCTR, y_click) + BCE(pCTCVR, y_conv)
        + **点击样本上的 BCE(pCVR, y_conv)**。

        ⚠️ 第三项（``cvr_weight``）不是可选项，**取 0 会让 pCVR 一路飘到 1**。

           只看前两项时，CVR 塔唯一的约束来自乘积 ``pCTR·pCVR``。
           而 ``1 − σ(a)σ(b)`` 在 ``b`` 变大之后趋于 ``1 − σ(a)``，
           **对 b 的偏导趋于 0** —— 也就是说一旦 pCVR 抬高，
           占绝大多数的负样本就再也提供不了任何向下的梯度，
           只剩下那 0.7% 的 CTCVR 正样本继续把 b 往上推。
           这是一个稳定但完全错误的平衡点。

           实测（mini 档，seed 42）：DeepFM 的 ``mean(p_cvr) = 0.9161``
           （真值 0.0428，高了 21 倍），FM 是 0.7895，LR 直接顶到 1.0000。
           而且它**不报错**：``mean(p_ctcvr)`` 反而看着还行，因为
           pCTR 恰好被压低来补偿。但这一列是要进 eCPM 的 ——
           带着它上线，等于把出价系数整体放大 20 倍。

           同时它还解释了另一个症状：CVR 塔没有平衡点，
           它的 logit 就会来回冲，第一节点的 FM 二阶项被推到大数值，
           epoch 1 的训练 loss 冲到 63（正常是 0.5）。所以这一项
           同时也是**训练稳定性**的解药，不只是校准的解药。

        ⚠️ 为什么这一项**不引入样本选择偏差（SSB）**：

           SSB 说的是「只拿点击样本训练」，那样模型永远看不到
           「曝光了但没点」的那 95% 流量。这里两项是**并存**的 ——
           CTCVR 那一项仍然在全量曝光上算，模型照样在全量分布上学习；
           CVR 辅助项只是额外钉住 pCVR 的**绝对水位**。

        ⚠️ 这一项也**不需要先验校正**：1:5 下采样是按 ``y_click`` 做的，
           点击行（正的和负的）**全部保留**，所以点击子集上
           ``P(转化|点击)`` 就是真实值。实测采样集 0.0600、真值 0.0600。
           与 ``correct_prior`` 只校正 pCTR 是同一个道理。

        ⚠️ 第二个 BCE 的标签是 ``y_conv``（在**全量曝光**上定义：
           曝光→点击→转化），不是「点击样本上的 CVR 标签」。
           它天然是 0/1，且在全量样本上都可观测 —— 这正是 ESMM 绕开
           样本选择偏差的地方。**不要**先把数据过滤成只剩点击样本再算，
           那样就把 ESMM 唯一的价值扔掉了。

        ``click_weight`` 调 CTR 那一路的权重。两个塔的参数是独立的，
        所以它**不再**用于「防止 CTR 任务主导 CVR 任务」——
        它现在只改变 CTR 塔自己的有效学习率（正样本率 5% 对 0.3%，
        两路的梯度量级本来就不在同一个数量级上）。

        ⚠️ ``pCTCVR = σ(a)·σ(b)`` **没有** logit 形式（它的 logit 是
           ``log(σ(a)σ(b) / (1−σ(a)σ(b)))``，不是 ``a+b``）。
           所以不能直接丢给 ``binary_cross_entropy_with_logits``，
           只能自己算 log 概率 —— 但必须用 ``logsumexp`` 的恒等式算，
           不能先算出 p 再 ``log(p)``：

               log p      = −softplus(−a) − softplus(−b)
               log(1 − p) = logsumexp([0, a, b]) − softplus(a) − softplus(b)

           后一个恒等式来自
           ``1 − σ(a)σ(b) = (1 + eᵃ + eᵇ) / ((1+eᵃ)(1+eᵇ))``，
           取对数就是上面那行。p 在 a、b 都很大时会舍入到 1.0
           （float32 在 1 附近的间隔约 1e-7），``log(1-p) = log(0) = -inf``，
           反向传播 0×inf = NaN。恒等式在 log 空间里从头算到尾，
           永远不会经过「先算 1−p」这一步。
        """
        m = self.mcfg
        return esmm_loss_manual(
            out["logit_ctr"], out["logit_cvr"], y_click, y_conv,
            click_weight=m.click_weight if click_weight is None else click_weight,
            ctcvr_weight=(m.ctcvr_weight if ctcvr_weight is None
                          else ctcvr_weight),
            cvr_weight=m.cvr_aux_weight if cvr_weight is None else cvr_weight)


def esmm_loss_manual(logit_ctr: torch.Tensor, logit_cvr: torch.Tensor,
                     y_click: torch.Tensor, y_conv: torch.Tensor,
                     click_weight: float = 1.0, ctcvr_weight: float = 1.0,
                     cvr_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """ESMM 三项损失的**纯函数**版本。

    拆成模块级函数是为了让测试能直接喂极端 logit 进去 —— 不用先造一个
    模型和数据管线，就能验证「a=b=40 时损失有限」这种数值性质。
    ``FactorizationMachineRanker.esmm_loss`` 只是它的一层包装。
    """
    yc = y_click.reshape(-1, 1).float()
    yv = y_conv.reshape(-1, 1).float()

    loss_ctr = F.binary_cross_entropy_with_logits(logit_ctr, yc,
                                                  reduction="mean")

    log_p = (-F.softplus(-logit_ctr)) + (-F.softplus(-logit_cvr))
    zeros = torch.zeros_like(logit_ctr)
    log_1mp = (torch.logsumexp(
        torch.cat([zeros, logit_ctr, logit_cvr], dim=1), dim=1, keepdim=True)
        - F.softplus(logit_ctr) - F.softplus(logit_cvr))
    loss_ctcvr = -(yv * log_p + (1.0 - yv) * log_1mp).mean()

    # 点击样本上的 CVR 辅助损失：钉住 pCVR 的绝对水位。
    # masked_mean 在 mask 全 0（小 batch 里一个点击都没有）时返回 0 而不是 NaN。
    loss_cvr = masked_mean(
        F.binary_cross_entropy_with_logits(logit_cvr, yv, reduction="none"), yc)

    total = (click_weight * loss_ctr + ctcvr_weight * loss_ctcvr
             + cvr_weight * loss_cvr)
    return total, {"loss_ctr": float(loss_ctr.detach()),
                   "loss_ctcvr": float(loss_ctcvr.detach()),
                   "loss_cvr": float(loss_cvr.detach())}


def correct_prior(out: dict, beta_ctr: float) -> dict:
    """把负下采样训练出的概率还原到真实先验。

    下采样把「负样本」按 β 的比例抽稀，于是**几率（odds）被整体放大 1/β 倍**：

        odds_sample = odds_true / β   ⟹   logit_true = logit_sample + ln β

    ⚠️ **只需要校正 pCTR 一个头，pCVR 不用校正 —— 这一点很容易想反。**

       前提是「按点击标签下采样」，于是：

       - **pCTR** 的正样本就是点击，全保留 → odds 被放大 1/β 倍 → 必须校正。
       - **pCVR** = P(转化 | 点击)。它的正样本（点了且转化）全保留，
         负样本（点了没转化）也全保留（它们都是点击样本），
         **一点都没被抽稀** → 校正它反而会引入 1/β 倍的错误。

       实测（mini 档，β_ctr = 0.262）：采样集里 P(转化|点击) = 0.0600，
       真实值 0.0600，一模一样；而 pCTR 从 0.1667 校正回 0.0500。

       如果当初图省事**按转化标签**下采样，情况正好反过来 ——
       那时「点过但没转化」的样本作为 CTR 的正样本被抽掉了，
       ``ln β`` 不再能还原 pCTR，实测会把 0.061 校正成 0.013（真值 0.050）。
       所以下采样对哪一列做，是个必须想清楚的决定，不是随手一写。

    pCTCVR = pCTR_corrected × pCVR，与乘法参数化保持一致。
    """
    lc = np.asarray(out["logit_ctr"], dtype=np.float64).ravel() \
        + math.log(max(beta_ctr, 1e-12))
    p_ctr = _sigmoid_np(lc)
    p_cvr = np.asarray(out["p_cvr"], dtype=np.float64).ravel()
    return {"logit_ctr": lc, "p_ctr": p_ctr, "p_cvr": p_cvr,
            "p_ctcvr": p_ctr * p_cvr}


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。

    ⚠️ 不能写成 ``1/(1+np.exp(-x))``。x 很负时 ``np.exp(-x)`` 会上溢到 inf
       （1300 行数据里就足够触发），得到 0 的同时在 stderr 打一条
       ``RuntimeWarning: overflow encountered in exp`` —— 结果碰巧是对的，
       但警报会把真正的数值问题淹没掉。按符号分支就完全不会溢出。
    """
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def make_ranker(cat_fields, dense_dim, cfg: Config,
                variant: str = "deepfm") -> FactorizationMachineRanker:
    """按名字造模型。``variant ∈ {lr, fm, deepfm}``。

    这是消融实验的唯一入口 —— 三个变体的差异**只**来自这两个开关。
    """
    variants = {
        "lr": DeepFMConfig.as_lr,
        "fm": DeepFMConfig.as_fm,
        "deepfm": DeepFMConfig.as_deepfm,
    }
    if variant not in variants:
        raise ValueError(f"未知变体 {variant!r}，可选 {sorted(variants)}")
    return FactorizationMachineRanker(cat_fields, dense_dim, variants[variant](cfg))
