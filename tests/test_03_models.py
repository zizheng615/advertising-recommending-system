"""职责点 2：双塔召回 + DeepFM 排序 + ESMM 双头。

这个文件分两半，性质完全不同：

**上半场 —— 恒等式与数值稳定性（秒级）**
    FM 的 O(F·d) trick 和朴素双循环对不对得上、ESMM 的 logsumexp 损失
    和手写 BCE 一不一样、``pCTCVR = pCTR × pCVR`` 成不成立。
    这些是**精确断言**（tol ≤ 1e-5），跑得飞快，坏了立刻知道。

**下半场 —— 消融实验（分钟级）**
    ``AUC(DeepFM) > AUC(FM) > AUC(LR)``。这条才是职责描述里
    「深度挖掘低阶和高阶非线性交叉特征」的真正证明，也是整个项目里
    最难写稳的一条测试 —— 见 ``test_ablation_ordering_ctr`` 的长注释。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.metrics.ranking import expected_calibration_error, safe_auc
from src.models.deepfm import (DeepFMConfig, FactorizationMachineRanker,
                               correct_prior, esmm_loss_manual)
from src.models.layers import fm_second_order
from src.models.train import predict

MAX_ROW = 20_000        # 恒等式类测试只需要很小的样本量


# ============================================================ 数值 / 结构
def test_fm_second_order_matches_naive_double_loop(bundle):
    """FM 的 O(F·d) trick 必须和 O(F²·d) 的朴素双循环**逐元素相等**。

    ``sum_{i<j}<e_i,e_j> = 0.5*(|Σe|² − Σ|e|²)`` 是整个 FM 的算力关键：
    56 个字段时朴素写法要算 1540 次内积，trick 只要 56 次向量加法。
    但它是**改写过**的公式，所以必须拿朴素写法当标尺钉死。

    ⚠️ 容差 1e-5 而不是 1e-7：trick 里有 ``(Σe)² − Σe²`` 这种
       「两个大数相减」的步骤，float32 下相对误差会被放大。
       实测在 emb=8、56 字段、batch=64 时最大绝对误差约 1e-6。
    """
    torch.manual_seed(0)
    B, F, d = 64, 12, 8
    embs = [torch.randn(B, d) for _ in range(F)]

    fast = fm_second_order(embs)                       # (B, 1)

    # 朴素双循环：老老实实枚举所有 i<j 的内积
    naive = torch.zeros(B, 1)
    for i in range(F):
        for j in range(i + 1, F):
            naive += (embs[i] * embs[j]).sum(dim=1, keepdim=True)

    assert torch.allclose(fast, naive, atol=1e-5), (
        f"FM trick 与朴素双循环最大偏差 "
        f"{(fast - naive).abs().max().item():.3e}")


def test_fm_second_order_never_produces_nan(bundle):
    """trick 的中间量在浮点下可能微负，但**绝不允许开根号**。

    朴素实现里有人会写成 ``sqrt(|Σe|² − Σe²)`` 去「修正」负号，
    那会在某些 batch 上偶发 NaN —— 而且只在部分 batch 上出现，
    极难复现。这里用极端输入（大模长、全零）压一遍。
    """
    torch.manual_seed(1)
    cases = [
        [torch.randn(32, 8) * 1e3 for _ in range(10)],       # 大模长
        [torch.zeros(32, 8) for _ in range(10)],             # 全零
        [torch.randn(1, 8) for _ in range(10)],              # batch=1
    ]
    for i, embs in enumerate(cases):
        out = fm_second_order(embs)
        assert torch.isfinite(out).all(), f"第 {i} 组极端输入产生了 NaN/inf"
        assert out.shape == (embs[0].shape[0], 1)


def test_esmm_loss_matches_manual_bce(bundle):
    """ESMM 的 logsumexp 损失必须等于「先算 p 再算 BCE」的手写版本。

    主实现走的是恒等式
        log p      = −softplus(−a) − softplus(−b)
        log(1 − p) = logsumexp([0,a,b]) − softplus(a) − softplus(b)
    绕开「先算 1−p」。这条测试就是拿朴素版本（用 float64 算，
    避免它自己先溢出）当标尺，确认恒等式没写错。
    """
    torch.manual_seed(2)
    B = 4096
    a = torch.randn(B, 1) * 3.0
    b = torch.randn(B, 1) * 3.0
    y_click = (torch.rand(B, 1) < torch.sigmoid(a)).float()
    y_conv = (torch.rand(B, 1) < torch.sigmoid(b)).float()

    got, info = esmm_loss_manual(a, b, y_click, y_conv,
                                 click_weight=1.0, ctcvr_weight=1.0,
                                 cvr_weight=1.0)

    # 朴素版本：float64 下老老实实算 p 和 1−p
    p = torch.sigmoid(a.double()) * torch.sigmoid(b.double())
    eps = 1e-12
    ctr = -(y_click.double() * torch.log(torch.sigmoid(a.double()) + eps)
            + (1 - y_click.double()) * torch.log(1 - torch.sigmoid(a.double()) + eps))
    ctcvr = -(y_conv.double() * torch.log(p + eps)
              + (1 - y_conv.double()) * torch.log(1 - p + eps))
    m = y_click.double()
    cvr = -((y_conv.double() * torch.log(torch.sigmoid(b.double()) + eps)
             + (1 - y_conv.double())
             * torch.log(1 - torch.sigmoid(b.double()) + eps))
            * m).sum() / m.sum().clamp_min(1.0)
    want = ctr.mean() + ctcvr.mean() + cvr

    assert abs(float(got) - float(want)) < 1e-4, (
        f"ESMM 损失 {float(got):.6f} 与手写 BCE {float(want):.6f} 不一致")
    assert set(info) == {"loss_ctr", "loss_ctcvr", "loss_cvr"}


def test_esmm_loss_is_finite_on_extreme_logits(bundle):
    """极端 logit 下损失必须有限 —— 这是恒等式存在的**唯一理由**。

    朴素写法 ``log(1 − σ(a)σ(b))`` 在 a=b=40 时 σ 舍入到 1.0，
    ``log(0) = -inf``，反传时 0×inf = NaN，整个网络瞬间报废。
    """
    a = torch.tensor([[40.0], [-40.0], [0.0], [200.0], [-200.0]],
                     requires_grad=True)
    b = torch.tensor([[40.0], [40.0], [-40.0], [200.0], [-200.0]],
                     requires_grad=True)
    y_click = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])
    y_conv = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])

    loss, info = esmm_loss_manual(a, b, y_click, y_conv)
    assert torch.isfinite(loss), f"极端输入下 loss = {loss}"
    assert all(np.isfinite(v) for v in info.values()), f"分量里有非有限值：{info}"

    loss.backward()
    for name, t in [("a", a), ("b", b)]:
        assert torch.isfinite(t.grad).all(), f"{name} 的梯度里有 NaN/inf"


def test_deepfm_equals_fm_when_high_order_head_is_zeroed(bundle):
    """DeepFM 在初始化那一刻**恰好等于** FM —— 三个变体是严格嵌套的。

    ``head_ctr`` / ``head_cvr`` 零初始化，于是高阶那一路输出恒为 0，
    logit 只剩「一阶 + 二阶」。这条恒等式的作用是排掉一个解释：
    「DeepFM 赢是因为它初始化得更好」。不是 —— 它在 t=0 和 FM 一模一样。

    做法：直接拿 DeepFM 自己的 ``bank_ctr`` 手算一阶+二阶，和 forward
    的结果比。不能拿「FM 模型」和「DeepFM 模型」比，因为两个模型的
    嵌入表是各自独立初始化的，值本来就不同。
    """
    from src.models.layers import first_order

    torch.manual_seed(3)
    from src.features.pipeline import FeatureBundle          # noqa: F401
    fields = [("a", 5), ("b", 7), ("c", 3)]
    dense_dim = 4
    mcfg = DeepFMConfig.as_deepfm(DeepFMConfig())
    mcfg.emb_dim, mcfg.dnn_hidden, mcfg.dropout = 8, (16, 8), 0.0
    model = FactorizationMachineRanker(fields, dense_dim, mcfg)
    model.eval()

    cat = torch.randint(0, 3, (32, len(fields)))
    dense = torch.randn(32, dense_dim)

    with torch.no_grad():
        got = model(cat, dense)["logit_ctr"]
        bank = model.bank_ctr
        embs = bank(cat, dense)
        want = first_order(bank.linear_embeddings, cat, dense,
                           bank.dense_linear) + fm_second_order(embs)

    assert torch.allclose(got, want, atol=1e-6), (
        f"DeepFM 在零初始化下没有退化成 FM，最大偏差 "
        f"{(got - want).abs().max().item():.3e}")


def test_pctcvr_is_exactly_pctr_times_pcvr(model_and_batch):
    """``pCTCVR = pCTR × pCVR`` 必须是**结构性等式**，不是「约等于」。

    这两个头如果有任何一行违反它（早期版本用「两个独立头 + 相除」就会），
    下游 eCPM = pCTR × pCVR × LTV 就和排序用的分数对不上了。
    容差给到 1e-6 是因为它本来就该是精确的。
    """
    out = model_and_batch
    lhs = out["p_ctcvr"]
    rhs = out["p_ctr"] * out["p_cvr"]
    assert torch.allclose(lhs, rhs, atol=1e-6), (
        f"pCTCVR 与 pCTR×pCVR 最大偏差 {(lhs - rhs).abs().max().item():.3e}")


def test_pcvr_never_exceeds_one(bundle, model_and_batch):
    """pCVR 必须天然 ≤ 1（乘法参数化给的保证），不需要任何 clip。

    早期版本用 ``pCVR = pCTCVR / pCTR`` 再 clip 到 1，
    实测 ``mean(p_cvr) = 1.0000`` —— 整列都被 clip 满了，
    而 CTR/CTCVR 两路的 AUC 看起来完全正常。这个 bug 会一路带到出价。
    """
    p = model_and_batch["p_cvr"]
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0 + 1e-6, (
        f"pCVR 落在 [{float(p.min()):.6f}, {float(p.max()):.6f}]，越界了")


# ============================================================ 消融实验
def test_ablation_ordering_ctr(rankers, cfg, bundle, ranker_tensors,
                               impressions, split):
    """**本文件最重要的一条**：DeepFM > FM > LR，且优势必须可测。

    职责描述写的是「DeepFM 深度挖掘低阶和高阶非线性交叉特征」。
    要把这句话变成可验证的命题，需要真值里**真的存在**只有 DNN
    能表达的成分 —— 三阶张量 ``T3[u_act, a_pop, ctx_slot]`` 和
    四阶 ``T4[..., page_kind]``。FM 的模型形式里根本没有三阶项，
    所以它在结构上不可能拟合这一部分；LR 连二阶都没有。

    ⚠️ 这条测试折腾了很久才写稳，中间踩的坑都值得记下来 ——
       因为它们每一个都让结果看起来像「DeepFM 这个模型不行」：

       1. **真值的成分比例**。最初 alpha_2=1.2 而 alpha_3=1.0，
          信号几乎全在 FM 的主场（二阶）里，单看二阶项 CTR AUC 就有
          0.80、单看三阶只有 0.67。FM 逼近上界，DeepFM 无处施展，
          差值在 0 附近随机震荡。→ 调成 alpha_2=0.45 / alpha_3=2.4 /
          alpha_4=2.2，单成分 AUC 变成 0.62 对 0.90。
       2. **高阶项对模型不可观测**。``a_pop``（素材热度分档）原本
          压根不在曝光特征表上，于是那份三阶信号对所有模型都是噪声。
          → 把四个**输入键**加进特征（加的是输入，不是 T3 的值，
          所以不是泄漏）。
       3. **数值块没标准化**。``seg_x_creative_cnt`` 的 std 有 6921，
          同块其他列只有 0.03。FM 的二阶项是双线性型，初始化后直接
          输出 ±3000，epoch 1 的 loss 冲到 165。LR 毫发无伤，
          于是看起来像「FM 有 bug」。→ 见 ``schema.LOG1P_FEATURES``。
       4. **早停看错了指标**。CTCVR 在验证集上只有约 140 个正样本，
          AUC 标准误 ≈0.02，用它选 epoch 等于随机抽一个。
          → 改成看 CTR 的验证 AUC（正样本 3300+，标准误 ≈0.005）。

       修完之后，**16 epoch、三个随机种子**跑出来的干净对照如下
       （``valid ∪ test`` 上的 CTR AUC）：

       | seed | LR | FM | DeepFM | FM−LR | DeepFM−FM |
       |---|---|---|---|---|---|
       | 42   | 0.8220 | 0.8732 | 0.8893 | +0.0512 | +0.0161 |
       | 7    | 0.8623 | 0.9093 | 0.9198 | +0.0470 | +0.0105 |
       | 2024 | 0.8474 | 0.8678 | 0.8949 | +0.0204 | +0.0271 |

       门槛取 ``+0.015`` / ``+0.005``：相对三个种子里**最差**的那次
       （+0.0204 / +0.0105）仍留了 26% / 52% 的余量。
       测试本身是确定性的（固定种子 + 单线程 torch），所以这里要防的
       不是「随机波动」，而是「有人改了特征或模型结构之后优势被
       悄悄吃掉」—— 门槛要卡在「真的坏了」和「只是换了个种子」之间。

       ⚠️ 曾经踩过的坑：把门槛定成 +0.02 又只有 8 个 epoch 时，
          seed 2024 的 FM−LR 恰好是 +0.0292，看着还有余量；
          但 `best_ep` 在所有变体上都等于最后一个 epoch —— 早停从未
          触发，说明模型根本没训完，那些数字全是欠训练的产物。
          提到 16 epoch 之后 seed 2024 掉到 +0.0204，恰好卡在旧门槛上。
          **门槛必须建立在收敛之后的测量上，否则它保护的是噪声。**
    """
    ev = np.concatenate([split.valid, split.test])
    y_click = impressions["y_click"].to_numpy()[ev]

    auc = {}
    for v in ("lr", "fm", "deepfm"):
        out = predict(rankers[v][0], ranker_tensors, ev,
                      keys=("logit_ctr", "p_cvr"))
        auc[v] = safe_auc(y_click, out["logit_ctr"])

    assert auc["fm"] - auc["lr"] > 0.015, (
        f"二阶交叉没有带来提升：AUC(LR)={auc['lr']:.4f} "
        f"AUC(FM)={auc['fm']:.4f}")
    assert auc["deepfm"] - auc["fm"] > 0.005, (
        f"高阶交叉没有带来提升：AUC(FM)={auc['fm']:.4f} "
        f"AUC(DeepFM)={auc['deepfm']:.4f} —— 先查真值的 alpha_3/alpha_4 "
        f"是不是被调小了，再查高阶输入键还在不在特征表里")
    assert auc["deepfm"] > auc["fm"] > auc["lr"], (
        f"消融顺序不为 LR < FM < DeepFM：{auc}")


def test_ablation_ranking_on_cvr(rankers, cfg, bundle, ranker_tensors,
                                 impressions, split):
    """CVR 塔上同样是 DeepFM > FM。

    断言只覆盖 DeepFM 对 FM，**不包含 LR**：CVR 的辅助监督在训练集里
    只有约 600 个正样本，FM 那种「容量大但没有深度」的模型会明显过拟合，
    排序能力在种子之间会掉到 LR 以下（实测 seed 42/2024 如此，seed 7 相反）。
    而 DeepFM 三个种子全部为正 —— 这是真值里那份高阶结构决定的，
    不是训练噪声。把 LR 也写进断言只会得到一个 flaky 的测试。
    """
    ev = np.concatenate([split.valid, split.test])
    y_click = impressions["y_click"].to_numpy()[ev]
    clicked = y_click == 1
    y_conv = impressions["y_conv_oracle"].to_numpy()[ev]

    auc = {}
    for v in ("lr", "fm", "deepfm"):
        out = predict(rankers[v][0], ranker_tensors, ev, keys=("p_cvr",))
        auc[v] = safe_auc(y_conv[clicked], out["p_cvr"][clicked])

    assert auc["deepfm"] > auc["fm"], (
        f"CVR 塔上 DeepFM 没有打赢 FM：FM={auc['fm']:.4f} "
        f"DeepFM={auc['deepfm']:.4f}")


def test_prior_correction_restores_ctr_level(rankers, cfg, ranker_tensors,
                                             impressions, split, sampled_train):
    """先验校正必须把 pCTR 的**绝对水位**拉回真实基率。

    ⚠️ 这条抓的是「漏掉 ``logit += ln β``」这个 bug。
       1:5 下采样把 CTR 从 5% 抬到 16.7%，不校正的话推理时会一直
       用 16.7% 去算 eCPM —— 出价系统性高估约 3.3 倍，
       而且**所有排序指标（AUC）都完全正常**，只有水位是错的。

    容差 25% 相对误差：mini 档测试集的真实 CTR 只有 5% 上下，
    模型本身还有一点校准误差，钉到 5% 以内会变成在测噪声。
    """
    _, beta = sampled_train
    ev = np.concatenate([split.valid, split.test])
    y_click = impressions["y_click"].to_numpy()[ev]

    out = predict(rankers["deepfm"][0], ranker_tensors, ev,
                  keys=("logit_ctr", "p_cvr"))
    raw = 1.0 / (1.0 + np.exp(-out["logit_ctr"]))
    corr = correct_prior(out, beta)["p_ctr"]

    true_rate = y_click.mean()
    assert abs(raw.mean() - true_rate) / true_rate > 0.5, (
        f"未校正的 pCTR 均值 {raw.mean():.4f} 居然接近真值 {true_rate:.4f} —— "
        f"下采样可能没生效，这条测试就失去意义了")
    assert abs(corr.mean() - true_rate) / true_rate < 0.25, (
        f"校正后 pCTR 均值 {corr.mean():.4f}，真值 {true_rate:.4f}，偏差过大")


def test_pctcvr_mean_matches_base_rate(rankers, ranker_tensors, impressions,
                                       split, sampled_train):
    """校正后的 ``mean(pCTCVR)`` 必须接近真实的转化率。

    这一条同时验证三件事：CVR 塔没有飘（曾经飘到 mean(p_cvr)=0.92）、
    先验校正只作用在 CTR 上、以及乘法参数化的口径自洽。
    """
    _, beta = sampled_train
    ev = np.concatenate([split.valid, split.test])
    y_conv = impressions["y_conv"].to_numpy()[ev]

    out = predict(rankers["deepfm"][0], ranker_tensors, ev,
                  keys=("logit_ctr", "p_cvr"))
    c = correct_prior(out, beta)

    ratio = c["p_ctcvr"].mean() / y_conv.mean()
    assert 0.6 < ratio < 1.6, (
        f"mean(pCTCVR)={c['p_ctcvr'].mean():.6f}，真实转化率={y_conv.mean():.6f}，"
        f"相差 {ratio:.2f} 倍 —— 先查 CVR 塔是不是飘了"
        f"（mean(p_cvr)={c['p_cvr'].mean():.4f}）")


def test_ctr_calibration_ece(rankers, ranker_tensors, impressions, split,
                             sampled_train):
    """CTR 的期望校准误差必须足够小。

    AUC 只度量排序，一个把所有人都预测成 0.9 的模型也能有 AUC。
    而 eCPM 是要**拿数值去乘**的，所以水位必须准。ECE < 0.02
    相当于「分箱后的平均偏差不超过 2 个百分点」，对 5% 基率来说很严。
    """
    _, beta = sampled_train
    ev = np.concatenate([split.valid, split.test])
    y_click = impressions["y_click"].to_numpy()[ev]

    # ``correct_prior`` 同时要 logit_ctr 和 p_cvr（它按 ESMM 的乘法口径
    # 返回 pCTCVR = pCTR_corrected × pCVR），所以两个键都得取。
    # 只取 logit_ctr 会在函数里以 KeyError 炸掉 —— 报错点在 deepfm.py，
    # 但根因在这里。
    out = predict(rankers["deepfm"][0], ranker_tensors, ev,
                  keys=("logit_ctr", "p_cvr"))
    p = correct_prior(out, beta)["p_ctr"]
    ece = expected_calibration_error(y_click, p)
    assert ece < 0.02, f"CTR 的 ECE = {ece:.4f}，概率水位不可信"


def test_ranker_survives_extreme_category_values(bundle, ranker_tensors):
    """类别索引取到上界、数值块取极端值时，forward 不能出 NaN。

    ``nn.Embedding`` 越界不报错、读垃圾内存，训练几百步后才以 NaN 现身。
    这里主动把所有索引推到 card-1、数值推到 ±50σ 压一遍。
    """
    mcfg = DeepFMConfig.as_deepfm(DeepFMConfig())
    mcfg.emb_dim, mcfg.dnn_hidden, mcfg.dropout = 8, (16, 8), 0.0
    model = FactorizationMachineRanker(
        bundle.cat_fields, bundle.imp_dense.shape[1], mcfg)
    model.eval()

    cat = torch.tensor(
        [[card - 1 for _, card in bundle.cat_fields]], dtype=torch.long)
    cat = cat.repeat(8, 1)
    dense = torch.full((8, bundle.imp_dense.shape[1]), 50.0)
    dense[1::2] = -50.0

    with torch.no_grad():
        out = model(cat, dense)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"极端输入下 {k} 出现 NaN/inf"

    y = torch.tensor([[1.0, 1.0]] * 4 + [[0.0, 0.0]] * 4)
    loss, _ = model.loss(out, y)
    assert torch.isfinite(loss), f"极端输入下 loss = {loss}"
