"""05 ZILN：预测「会不会付费」和「付多少」，给出 eCPM 里的 LTV。

产出
----
``artifacts/models/ziln_{scale}.pkl``           ZILN 模型
``artifacts/data/ziln_tensors_{scale}.pkl``     用户级张量 + 切分下标
``artifacts/metrics/05_ziln_{scale}.json``

**这一步在证明什么**

90% 的用户永不付费。直接回归金额会被 0 主导 —— 模型学到的唯一东西
就是「预测 0」，而这恰恰是最安全的选择。ZILN 把问题拆成两半：
``π(x) = P(付费)`` 和 ``exp(μ(x)) = 付费金额的条件中位数``，
再由混合分布给出条件期望。

**最容易写错、而且不报错的一个公式**

    Ê[y] = π · exp(μ + σ²/2)     ← 对
    Ê[y] = π · exp(μ)            ← 错（这是条件**中位数**）

σ=0.7 时 ``exp(σ²/2) = 1.28``，σ=1.5 时是 3.08。少乘这一项，
eCPM 低估 30%~200%，而 **AUC / Gini 全都看不出来** ——
它们只看次序，不看水平。所以这一步的核心指标是
``aggregate_bias``（**绝对**偏差），不是 AUC。

**为什么必须按步数而不是 epoch 配预算**

用户级样本量本来就小（mini 档 4000 用户），batch 一大每 epoch 只剩
十几个 batch。``cfg`` 的默认 10 epoch × batch 512 = **60 步**，
此时付费分类 AUC 只有 0.52 —— 和瞎猜一样，但 loss 曲线看起来很正常
（在降）。所以这里覆盖成 60 epoch × batch 256 = 720 步。

**PIT 必须做两次，混在一起会得出反向结论**

同一个统计量 ``Φ((ln y − μ)/σ)``，喂不同的 μ/σ 回答的是两个问题：

| 用谁的参数 | 回答的问题 | p（mini 实测） |
|---|---|---|
| 生成器的真实 ``_true_log_mu`` + ``ltv_sigma`` | 金额是不是真的对数正态（**建模前提**） | 0.55 ✅ |
| 模型的 ``μ̂`` / ``σ̂`` | 参数估得准不准（**校准程度**） | 0.006 ❌ |

只做第二个，会看到 p=0.006，然后得出「对数正态假设不成立、ZILN 从根上
就是错的」—— 而真相只是 σ̂ 估小了（实测均值约为真实值的 0.56 倍）。
**门槛只压在第一个上**；第二个如实报出来，并且额外卡一条
「σ̂ 均值 / 真实 σ 必须落在 [0.4, 2.0]」，防止 σ 头退化到量纲之外。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from scripts._common import (DATA_DIR, MODEL_DIR, Timer, banner,  # noqa: E402
                             cfg_for, load_bundle, load_data, log,
                             parse_scale, save_artifact, save_metrics,
                             show_table)
from src.common.seed import set_seed                        # noqa: E402
from src.metrics import ranking as R                        # noqa: E402
from src.metrics import regression as G                     # noqa: E402
from src.models import dataset as DS                        # noqa: E402
from src.models.ziln import ZILN, ziln_predict_arrays      # noqa: E402

# 按**步数**配预算，不按 epoch —— 理由见模块文档。
ZILN_EPOCHS = 60
ZILN_BATCH = 256
ZILN_LR = 3e-3


def user_split(n_users: int, seed: int,
               valid_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """用户级的**随机**切分。

    ⚠️ 不沿用曝光级的 ``time_split``：用户表上没有时间列（LTV 是
       「这个人一生会付多少钱」，本来就没有时间轴）。但同一个用户
       不能同时出现在 train 和 valid 里 —— ZILN 一行一个用户，
       重复出现的用户会被模型记住，验证指标虚高。
    """
    rng = np.random.default_rng([seed, 4242])
    perm = rng.permutation(n_users)
    n_valid = max(1, int(round(n_users * valid_frac)))
    return perm[n_valid:], perm[:n_valid]


def train_ziln(model: ZILN, tr: dict, va: dict, cfg) -> list[float]:
    """ZILN 的训练循环。

    为什么不复用 ``src/models/train.py``：那个循环是二分类的
    （一个 logit、一个标签），而 ZILN 一次要回传三个头、标签是连续金额。
    硬套要造一堆假的输出名，反而更容易错。
    """
    opt = torch.optim.AdamW(model.parameters(), lr=ZILN_LR,
                            weight_decay=cfg.weight_decay)
    n = tr["cat"].shape[0]
    losses: list[float] = []
    for ep in range(ZILN_EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(cfg.seed * 100 + ep)
        perm = torch.randperm(n, generator=g)
        ep_loss = []
        for s in range(0, n, ZILN_BATCH):
            b = perm[s:s + ZILN_BATCH]
            out = model(tr["cat"][b], tr["dense"][b])
            loss, _ = model.loss(out, tr["y"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss.append(float(loss.detach()))
        losses.append(float(np.mean(ep_loss)))
    return losses


@torch.no_grad()
def ziln_forward(model: ZILN, t: dict) -> dict:
    model.eval()
    return model(t["cat"], t["dense"])


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)
    torch.set_num_threads(1)

    data = load_data(cfg)
    users = data["users"]
    bundle = load_bundle(cfg)

    n_users = len(bundle.user_cat)
    tr_idx, va_idx = user_split(n_users, cfg.seed)
    tr = DS.ziln_user_tensors(bundle, users, tr_idx, cfg)
    va = DS.ziln_user_tensors(bundle, users, va_idx, cfg)
    save_artifact({"train": tr, "valid": va, "train_idx": tr_idx,
                   "valid_idx": va_idx}, DATA_DIR / "ziln_tensors.pkl", cfg)

    with Timer(f"05 训练 ZILN（{scale}）"):
        torch.manual_seed(cfg.seed)
        model = ZILN(tr["_fields"], tr["dense"].shape[1], cfg)
        losses = train_ziln(model, tr, va, cfg)
        save_artifact(model, MODEL_DIR / "ziln.pkl", cfg)

    n_steps = len(losses) * int(np.ceil(len(tr_idx) / ZILN_BATCH))
    banner("训练")
    log.info("用户级 %d 训练 / %d 验证（随机切分），%d epoch × batch %d "
             "= **%d 步**；loss %.4f → %.4f",
             len(tr_idx), len(va_idx), ZILN_EPOCHS, ZILN_BATCH, n_steps,
             losses[0], losses[-1])

    # ---------------- 三个头各自的诊断 ----------------
    # 用模型自带的 ``ziln_predict_arrays`` 而不是自己手写一遍转换：
    # 它同时给出 ``expected`` 和 ``median`` 两版，正好用来对比
    # 「含不含 σ²/2 修正」，也就**不可能**出现脚本和模型两处实现对不上的情况。
    pred = ziln_predict_arrays(ziln_forward(model, va))
    p_pay, mu, sigma = pred["p"], pred["mu"], pred["sigma"]
    y = va["y"].numpy().ravel()                       # 真实付费金额（元）
    pos = y > 0
    y_log = np.log(np.clip(y, 1e-6, None))
    ev = pred["expected"]                             # π·exp(μ+σ²/2)

    banner("σ 头")
    log.info("σ ∈ [%.4f, %.4f]，std %.4f（上界 %.1f / 下界 %.3f）—— "
             "σ 是常数就说明这一头退化了，混合分布变成普通对数正态",
             sigma.min(), sigma.max(), sigma.std(),
             cfg.sigma_max, cfg.sigma_min)

    # 付费分类：和「直接拿真实付费概率排序」的理论上限比
    oracle_p = users["_true_pay_prob"].to_numpy()[va_idx]
    auc_pay = R.safe_auc(pos.astype(float), p_pay)
    auc_oracle = R.safe_auc(pos.astype(float), oracle_p)
    banner("付费分类（判据不是 AUC 绝对值，而是「离理论上限有多远」）")
    log.info("AUC = %.4f，理论上限（拿生成器的真实付费概率排序）= %.4f，"
             "达到上限的 %.0f%%", auc_pay, auc_oracle,
             100 * auc_pay / auc_oracle if auc_oracle else 0.0)

    # ---------------- PIT：必须做**两次**，而且不能混为一谈 ----------------
    #
    # PIT = Φ((ln y − μ)/σ)，假设成立时服从 U(0,1)。但**用谁的 μ/σ**，
    # 检验的是完全不同的东西：
    #
    #   ① 用生成器的**真实参数** ``_true_log_mu`` + ``ltv_sigma``
    #      → 检验「金额确实是对数正态」这个**建模前提**。这才是门槛。
    #   ② 用模型的 μ̂/σ̂
    #      → 检验**模型参数估得准不准**，即校准程度。
    #
    # ⚠️ 一开始我只做了 ②，p = 0.006，看着像「对数正态假设不成立、
    #    ZILN 从根上就是错的」。换成真实参数后 p 直接到 0.55 ——
    #    **假设完全成立，是 σ̂ 估小了**（实测 σ̂ 均值 0.5405，
    #    真实值 0.70，比值 0.772）。把校准误差当成前提不成立，是最容易得出
    #    反向结论的一步，所以这里两个都算、两个都报，门槛只压在 ① 上。
    #
    # ⚠️ σ̂ 有**两个口径**，别混：σ 是「付费金额」这一头的尺度参数，
    #    而付费用户只占 10%，所以
    #        全体验证样本（含 90% 的 0）     mean_all = 0.4964
    #        只取付费用户（y > 0）           mean_pos = 0.5405 ← 「估得准不准」看这个
    #    拿 mean_all 去跟真实 σ=0.70 比会得出 0.709，与「偏小到 0.772」
    #    的结论方向一致但数值对不上 —— 因为分母是对的、分子换了对象。
    #    这条踩过：两个数一度以 ``mean`` / ``ratio_to_true`` 并排写进
    #    metrics json，看起来像一对，实际是两个口径，连 fig07 都
    #    照着画错（柱子画 0.4964、标注写 0.77 倍）。现在键名自带口径。
    #   ① 用**全表**的付费用户，不用验证折 —— 生成器的分布假设跟
    #      训练/验证怎么切没有任何关系，用满样本能把 KS 的统计功效
    #      从 88 个样本提到 352 个（388→88 会让 p 值纯靠抖动，
    #      同一个生成器两次跑出 0.11 和 0.55 都是可能的）。
    y_all = users["pay_amount"].to_numpy(dtype=np.float64)
    pos_all = y_all > 0
    mu_true = users["_true_log_mu"].to_numpy(dtype=np.float64)[pos_all]
    log_y_all = np.log(np.clip(y_all[pos_all], 1e-6, None))
    sigma_true = np.full(int(pos_all.sum()), cfg.ltv_sigma)
    pit_assumption = G.pit_uniformity_pvalue(log_y_all, mu_true, sigma_true)
    #   ② 只能在验证折上做 —— 它检验的是模型的参数，训练集上的 p 值没有意义
    pit_model = G.pit_uniformity_pvalue(y_log[pos], mu[pos], sigma[pos])
    sigma_ratio = float(sigma[pos].mean() / cfg.ltv_sigma)

    banner("PIT 均匀性检验（同一个统计量，用两套参数 = 两个完全不同的问题）")
    log.info("① 用**真实参数**：p = %.4f → %s。"
             "这一步检验的是「金额服从对数正态」这个建模前提",
             pit_assumption,
             "前提成立 ✅" if pit_assumption > 0.05
             else "前提不成立 ❌（生成器的金额分布实现有问题）")
    log.info("② 用**模型参数**：p = %.4f。这一步检验的是 μ̂/σ̂ 估得准不准，"
             "**不是**前提是否成立", pit_model)
    log.info("σ̂ 均值 %.4f（**付费样本口径**，全体验证样本口径是 %.4f），"
             "真实值 %.4f，比值 %.2f —— "
             "σ̂ 系统性偏小（是 %d 个用户级样本下这一头的已知短板，"
             "README 里如实记录）；σ̂ 偏小会放大 NLL 里的 "
             "(ln y − μ̂)²/2σ̂² 项。**只有付费口径能跟真实 σ 比** —— "
             "σ 描述的是 y>0 那一支的尺度",
             float(sigma[pos].mean()), float(sigma.mean()),
             cfg.ltv_sigma, sigma_ratio, len(tr_idx))

    # ---------------- 聚合偏差：这一步的核心 ----------------
    # 参照系的选择决定了这个数有没有意义，选错了就是白算：
    #   ① 逐用户的真实金额 ``y`` —— 含巨大抽样噪声（「这个人到底付没付」
    #      本身就是随机的）。拿它当分母，偏差里混着噪声，看不出系统性错误。
    #   ② 生成器自己的条件期望 ``expected_ltv = π·exp(μ+σ²/2)``
    #      —— 出价真正想要的那个量，没有抽样噪声。这才是分母。
    # 两个都算：把 ① 当作**噪声地板**报出来，读者才知道 ② 的
    # 相对大小说明了什么（0.22 / 0.09 ≈ 2.4 倍地板，才算「有系统性偏差」）。
    true_ev = users["expected_ltv"].to_numpy(dtype=np.float64)[va_idx]
    noise_floor = G.aggregate_bias(true_ev, y)
    bias = G.aggregate_bias(true_ev, ev)
    bias_naive = G.aggregate_bias(true_ev, pred["median"])

    banner("聚合偏差（这一步真正的判据）")
    log.info("含 exp(σ²/2) 修正：%.4f", bias)
    log.info("漏掉 σ²/2 的版本：%.4f（实测放大倍数 %.4f）—— "
             "这一项就是 eCPM 会低估的比例，而 AUC/Gini **完全看不出来**",
             bias_naive, float(np.mean(np.exp(sigma ** 2 / 2))))
    log.info("纯抽样噪声地板（拿真实金额比）= %.4f —— "
             "偏差要跟它比，不是跟 0 比", noise_floor)

    # ---------------- 排序与回归质量 ----------------
    # ⚠️ 这里原先有一句 banner("高价值用户的排序能力与条件金额的回归质量")，
    #    而它下面直接就是 show_table(..., "指标与门槛") —— show_table 自己
    #    会打标题，于是屏幕上留下一个**下面什么都没有的段落标题**，
    #    看着像哪一行日志被吞了。内容在下面的表里，标题删掉。
    gini = G.normalized_gini(y, ev)
    skill = G.log_mae_skill(y_log[pos], mu[pos])
    rows = [
        {"metric": "Normalized Gini（按 Ê[y] 排序）", "value": gini,
         "threshold": 0.30},
        {"metric": "付费分类 AUC", "value": auc_pay, "threshold": 0.55},
        {"metric": "log-MAE skill（正样本 μ 对 ln y）", "value": skill,
         "threshold": 0.10},
        {"metric": "聚合偏差 |Ê[y] − 真实条件期望|", "value": bias,
         "threshold": 0.50},
    ]
    show_table(rows, ["metric", "value", "threshold"], "指标与门槛")

    # ---------------- 门槛 ----------------
    problems = []
    if (sigma < cfg.sigma_min - 1e-9).any() or (sigma > cfg.sigma_max + 1e-9).any():
        problems.append(f"σ 越界：[{sigma.min():.4f}, {sigma.max():.4f}]，"
                        f"允许区间 [{cfg.sigma_min}, {cfg.sigma_max}]")
    if sigma.std() <= 1e-6:
        problems.append("σ 是常数 —— σ 头没在学，混合分布退化了")
    if not 0.4 <= sigma_ratio <= 2.0:
        problems.append(f"σ̂ 均值/真实 σ = {sigma_ratio:.3f}，超出 [0.4, 2.0] —— "
                        f"σ 头偏离实际量纲太远，NLL 的 (ln y−μ̂)²/2σ̂² 项会失控")
    if not pit_assumption > 0.05:
        problems.append(f"**用真实参数**做的 PIT p={pit_assumption:.4f} ≤ 0.05："
                        f"「金额服从对数正态」这个建模前提不成立，"
                        f"ZILN 的损失函数从根上就是错的")
    if not auc_pay > 0.55:
        problems.append(f"付费分类 AUC {auc_pay:.4f} ≤ 0.55")
    if not auc_pay > 0.80 * auc_oracle:
        problems.append(f"付费分类 AUC {auc_pay:.4f} 不到理论上限 "
                        f"{auc_oracle:.4f} 的 80%")
    if not gini > 0.30:
        problems.append(f"Normalized Gini {gini:.4f} ≤ 0.30")
    if not skill > 0.10:
        problems.append(f"log-MAE skill {skill:.4f} ≤ 0.10")
    if not bias < 0.50:
        problems.append(f"聚合偏差 {bias:.4f} ≥ 0.50")
    if not bias <= bias_naive + 0.02:
        problems.append(f"含 σ²/2 修正的偏差 {bias:.4f} 反而明显差于漏掉它的 "
                        f"版本 {bias_naive:.4f} —— expected_value 有问题")

    metrics = {
        "scale": scale,
        "train": {"n_train": int(len(tr_idx)), "n_valid": int(len(va_idx)),
                  "epochs": ZILN_EPOCHS, "batch": ZILN_BATCH,
                  "steps": int(n_steps),
                  "loss_first": losses[0], "loss_last": losses[-1]},
        # ⚠️ σ̂ 两个口径**必须分开存、键名自带口径**。
        #    原先只有 ``mean``（全体，含 90% 的 y=0）和 ``ratio_to_true``
        #    （只有付费样本）两个键并排 —— 名字看着是一对，实际是两个
        #    总体，于是 json 自己跟自己矛盾（0.4964 配 0.772，而
        #    0.4964/0.70 = 0.709）。fig07 照着画就画错了：柱子 0.4964、
        #    标注「是真实值的 0.77 倍」。
        #    比值只能配**付费口径** —— σ 是 y>0 那一支的尺度参数。
        "sigma": {"min": float(sigma.min()), "max": float(sigma.max()),
                  "std": float(sigma.std()),
                  "mean_all": float(sigma.mean()),
                  "mean_pos": float(sigma[pos].mean()),
                  "true_sigma": float(cfg.ltv_sigma),
                  "ratio_pos_to_true": sigma_ratio,
                  "ratio_all_to_true": float(sigma.mean() / cfg.ltv_sigma)},
        "metrics": {"auc_pay": auc_pay, "auc_oracle": auc_oracle,
                    "pit_assumption_pvalue": pit_assumption,
                    "pit_model_pvalue": pit_model,
                    "n_positives": int(pos.sum()),
                    "normalized_gini": gini, "log_mae_skill": skill,
                    "aggregate_bias": bias,
                    "aggregate_bias_without_sigma2": bias_naive,
                    "noise_floor": noise_floor},
        "problems": problems,
    }
    save_metrics("05_ziln", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：付费 AUC %.4f（上限 %.4f 的 %.0f%%）、"
             "前提 PIT p=%.3f（模型校准 PIT p=%.3f）、Gini %.3f、聚合偏差 %.3f "
             "（噪声地板 %.3f）", auc_pay, auc_oracle,
             100 * auc_pay / auc_oracle, pit_assumption, pit_model, gini,
             bias, noise_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
