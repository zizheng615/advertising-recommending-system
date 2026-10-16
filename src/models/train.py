"""训练循环：所有模型共用的一套。

统一在这里，是为了让不同模型之间的指标差异真的来自模型本身，
而不是「谁的学习率调得好一点」。

三条工程约定
------------
1. **早停看验证集 AUC，不看 loss。** 模型完全可以 loss 继续降、
   AUC 同时降（概率整体更自信但次序更差）。早停要看业务指标。
2. **梯度裁剪 1.0。** 嵌入表的梯度在极端稀疏的字段上会爆，
   一次爆炸就足以把整个嵌入表污染掉。
3. **AdamW 的 weight_decay 不加到 embedding 和 bias 上。**
   对嵌入做 L2 衰减等于把没出现过的取值往 0 拽，
   而这些取值本来就应该保持初始值。这是标准做法，但很容易漏。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.metrics.ranking import safe_auc


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    valid_auc: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_auc: float = 0.0

    def as_dict(self) -> dict:
        return {"train_loss": self.train_loss, "valid_auc": self.valid_auc,
                "valid_loss": self.valid_loss, "best_epoch": self.best_epoch,
                "best_auc": self.best_auc}


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """把参数分组：嵌入表和 bias 不做 weight decay，其余正常。

    ⚠️ 这条不做的话，嵌入表里**没被训练到的取值**（长尾用户、冷门素材）
       会被持续往 0 拉。它们本来应该保持初始的随机小值，
       被拉到 0 之后所有长尾 ID 的向量都一样了，等于把长尾流量
       全部映射到同一个点 —— 而这恰恰是本项目要解决的那个痛点。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or "embedding" in name.lower() or "emb" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_batches(n: int, batch_size: int, seed: int, shuffle: bool = True):
    """产出一批批的行索引。"""
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


# 标签张量的规范名字。**全部**要摘掉，理由见 slice_batch。
LABEL_KEYS = ("y", "y_click", "y_conv")


def _tensor_keys(tensors: dict) -> list[str]:
    """只对张量做行切片 —— 元信息（字段定义之类）原样带过去。"""
    return [k for k, v in tensors.items() if isinstance(v, torch.Tensor)]


def slice_batch(tensors: dict, rows: torch.Tensor,
                label_keys: tuple = LABEL_KEYS) -> tuple[dict, dict]:
    """切一个 batch，并把标签张量摘出来。

    ⚠️ **必须把「所有」标签都摘掉**再 ``model(**batch)``，不是只摘要用的那个。
       张量字典里通常同时躺着 ``y`` / ``y_click`` / ``y_conv`` 三个标签，
       只摘 ``label_key`` 的话，剩下两个会被当作 forward 的关键字参数传进去，
       直接 ``unexpected keyword argument``。更阴的情况是模型恰好有同名参数 ——
       那就静默地喂错了东西，loss 照样在降。

       所以这里按**规范名集合**摘，而不是按「这次要用哪个」摘。
    """
    batch, labels = {}, {}
    tkeys = set(_tensor_keys(tensors))
    for k, v in tensors.items():
        if k in tkeys:
            if k in label_keys:
                labels[k] = v[rows]
            else:
                batch[k] = v[rows]
        else:
            batch[k] = v
    return batch, labels


def train_model(
    model: nn.Module,
    tensors: dict,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_keys: tuple = LABEL_KEYS,
    label_key: str = "y",
    auc_key: str = "y",
    prob_key: str = "p",
    grad_clip: float = 1.0,
    patience: int = 2,
    seed: int = 42,
    verbose: bool = False,
) -> TrainHistory:
    """通用训练循环。

    ``tensors`` 是「名字 → 张量」的字典，第 0 维都是样本维。
    ``model`` 需要提供 ``loss(out, y) -> (loss, info)``，
    并且 ``forward(**batch)`` 能跑通。

    ``label_key`` 指定喂给 ``model.loss`` 的标签；``auc_key`` / ``prob_key``
    指定早停用的标签与预测（ESMM 双头时二者会不同：CTR 早停看
    ``y_click``/``p_ctr``，CTCVR 早停看 ``y_conv``/``p_ctcvr``）。
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(build_param_groups(model, weight_decay), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    hist = TrainHistory()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    label_keys = tuple(label_keys)

    for ep in range(epochs):
        model.train()
        losses = []
        for bidx in make_batches(len(train_idx), batch_size, seed * 1000 + ep):
            rows = torch.as_tensor(train_idx[bidx], dtype=torch.long)
            batch, labels = slice_batch(tensors, rows, label_keys)
            out = model(**batch)
            loss, _ = model.loss(out, labels[label_key])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪：稀疏大基数嵌入表的梯度很容易爆，一次爆炸能污染整张表
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()

        v_auc, v_loss = evaluate_model(
            model, tensors, valid_idx, batch_size=batch_size, seed=seed,
            label_keys=label_keys, label_key=label_key,
            auc_key=auc_key, prob_key=prob_key)
        hist.train_loss.append(float(np.mean(losses)))
        hist.valid_auc.append(v_auc)
        hist.valid_loss.append(v_loss)

        if verbose:
            print(f"    epoch {ep + 1:2d}  loss={hist.train_loss[-1]:.5f}  "
                  f"valid_auc={v_auc:.5f}  valid_loss={v_loss:.5f}")

        # ⚠️ 早停判据用 AUC，不用 loss：loss 还能继续降而 AUC 已经在跌，
        #    那说明模型只是变得更自信，排序能力反而退化了。
        if v_auc > hist.best_auc + 1e-5:
            hist.best_auc, hist.best_epoch = v_auc, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def evaluate_model(model: nn.Module, tensors: dict, idx: np.ndarray,
                   *, batch_size: int = 4096, seed: int = 42,
                   label_keys: tuple = LABEL_KEYS, label_key: str = "y",
                   auc_key: str = "y", prob_key: str = "p"
                   ) -> tuple[float, float]:
    """返回 (AUC, loss)。"""
    if len(idx) == 0:
        return 0.5, float("nan")
    model.eval()
    label_keys = tuple(label_keys)
    probs, labels, losses = [], [], []

    for bidx in make_batches(len(idx), batch_size, seed, shuffle=False):
        rows = torch.as_tensor(idx[bidx], dtype=torch.long)
        batch, lab = slice_batch(tensors, rows, label_keys)
        out = model(**batch)
        loss, _ = model.loss(out, lab[label_key])
        losses.append(float(loss))
        if prob_key not in out:
            raise KeyError(f"模型输出里没有 {prob_key!r}，无法算 AUC")
        probs.append(out[prob_key].detach().cpu().numpy().ravel())
        labels.append(lab[auc_key].detach().cpu().numpy().ravel())

    p = np.concatenate(probs)
    y = np.concatenate(labels)
    return safe_auc(y, p), float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, tensors: dict, idx: np.ndarray,
            *, batch_size: int = 4096, keys: tuple[str, ...] = ("p",),
            seed: int = 42) -> dict[str, np.ndarray]:
    """批量推理，返回 {输出名: numpy 数组}。

    只对张量切行 —— 元信息（``_fields`` 之类）直接传给 forward 之外的地方，
    不参与索引。这里选择**不传**任何非张量键给模型，避免污染签名。
    """
    model.eval()
    tkeys = set(_tensor_keys(tensors))
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for bidx in make_batches(len(idx), batch_size, seed, shuffle=False):
        rows = torch.as_tensor(idx[bidx], dtype=torch.long)
        batch = {k: v[rows] for k, v in tensors.items()
                 if k in tkeys and k not in LABEL_KEYS}
        out = model(**batch)
        for k in keys:
            buckets[k].append(out[k].detach().cpu().numpy().ravel())
    for k, v in buckets.items():
        if not v or sum(len(x) for x in v) != len(idx):
            raise ValueError(f"推理输出 {k!r} 的行数与请求的索引数不符")
    return {k: np.concatenate(v) for k, v in buckets.items()}
