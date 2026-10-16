"""03 排序模型消融 + 先验校正 + 延迟反馈去偏（DFM）。

产出
----
``artifacts/models/ranker_{variant}_{scale}.pkl``   LR / FM / DeepFM 三个变体
``artifacts/models/dfm_{scale}.pkl``                DFM 去偏模型
``artifacts/data/ranker_tensors_{scale}.pkl``       排序模型的输入张量
``artifacts/metrics/03_ranker_{scale}.json``

**这一步在证明什么**

职责描述里「DeepFM 深度挖掘低阶与高阶非线性交叉特征」这句话，只有在
真值里**真的存在** FM 表达不了的成分时才是可验证的命题。生成器为此
埋了三阶张量 ``T3[u_act, a_pop, ctx_slot]`` 和四阶 ``T4[..., page_kind]``
（FM 的模型形式里压根没有三阶项），以及「分层 × 素材」的稀疏高秩残差。
所以这里断言的不是「模型跑通了」，而是 **LR < FM < DeepFM 的排序**。

**1:5 负下采样之后必须做先验校正**

下采样把正率从 5% 抬到约 24%，模型学到的概率是**采样后**的先验。
推理时不做 ``logit += ln(β)``（β = 负样本保留率），eCPM 会系统性高估
约 4 倍 —— 而且**不报任何错**，只是预算悄悄花飞。

**这个文件里还有一条独立的主线：延迟反馈**

线上拿到的 ``y_conv`` 是「24h 窗口 + 数据末尾截断」后的标签，
比真值低 25% 左右（见 01 步的假负表）。DFM 用一个额外的标量延迟率头
``λ_d`` 把「还没转」和「不会转」分开，从而还原出无偏的 ``p(x)``。
判据是 **``mean(p̂)`` 对 oracle 的偏差**，不是 AUC ——
AUC 只关心排序，去偏要修的是**水平**，两者是两件事。

⚠️ **这个判据有口径，量错了会得出「DFM 偏了 5 倍」的假结论。**
DFM 的模型形式直接包含「点击」这一步，它估的是**曝光级**的
``P(转化)``，所以参照必须是**全曝光**上的 ``y_conv_w168`` 均值
（实测 ≈ 0.0030）—— 这正是本文件用的口径，也是**唯一**正确的口径。

同一个 ``y_conv_w168``，若改成在**点击样本**上取均值是 0.0575，
两者差 **19 倍**。拿后者去量 DFM，会得到「偏了 5 倍」这种惊悚数字，
而它其实只是**口径错了**：DFM 从来没打算估点击条件的转化率。
（DeepFM 的 ``p_cvr`` 头**才是**点击条件的 —— 它的损失算在点击行上，
所以它的参照就该用点击样本。两个头长得像，口径不一样，见 08。）

为此下面除了曝光级口径，还**并排报一份点击级口径**，让这个 19 倍的
差自己显形，而不是留给人去踩。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from scripts._common import (DATA_DIR, MODEL_DIR, Timer, banner,  # noqa: E402
                             cfg_for, load_bundle, load_data, load_split,
                             log, parse_scale, save_artifact, save_metrics,
                             show_table)
from src.common.seed import set_seed                        # noqa: E402
from src.debias.dfm import DFMModel, ElapsedFeatureModel    # noqa: E402
from src.metrics.ranking import (expected_calibration_error,  # noqa: E402
                                 safe_auc)
from src.models import dataset as DS                        # noqa: E402
from src.models.deepfm import (DeepFMConfig,                # noqa: E402
                               FactorizationMachineRanker)
from src.models.sampling import (downsample_negatives,      # noqa: E402
                                 prior_shift, smote, smote_report)
from src.models.train import predict, train_model           # noqa: E402

# 消融三个变体**共用**这套超参 —— 任何一处不一样，
# 差值就不再只反映模型结构，实验也就失去意义。
EPOCHS = 16
BATCH = 512
LR = 1e-3
EMB = 8
HIDDEN = (64, 32)

# 去偏那一组独立一套：**在完整训练折上训**，不做负下采样。
#
# ⚠️ 这一点必须写清楚，否则去偏实验会给出一个完全错误的结论。
#    1:5 负下采样把训练集正率抬到约 4 倍，模型学出的 p̂ 是**采样后先验**下
#    的概率，本身就比真实值高约 4 倍（要另外做先验校正还原）。
#    而延迟造成的低估只有 24% —— 两者叠在一起，测到的是「+63% 的高估」，
#    和「假负让 p̂ 偏低」这件事完全相反。
#    「类别不平衡」和「标签有偏」是两个独立的问题，必须分开测。
#
# ⚠️ 而且这一组**必须关掉早停**（patience 设成一个大数）。
#    早停的判据是「对 y_conv 的 AUC」，那是个**排序**指标；而去偏要修的
#    是概率的**水平**，两者不同步。实测开着早停时 DFM 在第 4 个 epoch
#    就被掐掉，偏差 −26.5%，而 vanilla 一路训到第 6 个 epoch 拿到 −33.5%
#    —— 比出来只有 21% 的改善，看着像「DFM 没用」，其实是**训练被截断了**。
#    固定跑满 12 个 epoch 之后是 −25.3% 对 −35.9%（改善 30%），
#    而且 10 / 12 / 16 三个长度下结论一致，不是调出来的。
DFM_EPOCHS = 12
DFM_BATCH = 1024
NO_EARLY_STOP = 99

_VARIANT = {"lr": DeepFMConfig.as_lr, "fm": DeepFMConfig.as_fm,
            "deepfm": DeepFMConfig.as_deepfm}


def train_ranker(variant: str, bundle, tensors, train_rows, valid_idx, cfg):
    """训练一个消融变体。

    早停看 **CTR** 的验证 AUC，不看 CTCVR：mini 档验证集里 CTR 正样本
    约 3300 个（AUC 标准误 ≈0.005），CTCVR 正样本只有约 140 个
    （标准误 ≈0.02）。用后者选 epoch 等于在 epoch 之间随机抽一个。
    """
    torch.manual_seed(cfg.seed)
    mcfg = _VARIANT[variant](cfg)
    mcfg.emb_dim, mcfg.dnn_hidden = EMB, HIDDEN
    model = FactorizationMachineRanker(
        bundle.cat_fields, bundle.imp_dense.shape[1], mcfg)
    hist = train_model(model, tensors, train_rows, valid_idx, epochs=EPOCHS,
                       batch_size=BATCH, lr=LR, weight_decay=cfg.weight_decay,
                       auc_key="y_click", prob_key="p_ctr", seed=cfg.seed)
    return model, hist


def cvr_auc(p_cvr: np.ndarray, imp, ev: np.ndarray) -> float:
    """CVR 塔的排序能力。

    ⚠️ 分母是**点击**而不是曝光：CVR 的条件事件是「点击后转化」。
       拿全曝光去算 AUC 会把「谁是高点击人群」混进来，
       数字好看但和 CVR 无关。
    """
    clicked = imp["y_click"].to_numpy()[ev] == 1
    y = imp["y_conv_oracle"].to_numpy()[ev][clicked]
    return safe_auc(y, p_cvr[clicked])


def smote_ablation(bundle, imp, split, cfg) -> dict:
    """SMOTE 的消融对照 —— 证明「下采样 + 先验校正」为什么不输给它。

    职责描述里写了 SMOTE。这里**完整实现**它，然后让数据说话：
    1:5 下采样已经把正率抬到 24%，此时 SMOTE 过采样是在一个**已经不
    不平衡**的分布上再造合成样本，既没有收益，还会因为合成样本落在
    原样本的凸组合上而**压缩特征空间的多样性**。

    ⚠️ SMOTE **绝不能跨时间切分做**。在 train∪valid∪test 上找近邻，
       会造出「未来样本的近邻」并落进训练集 —— 这是时序泄漏。
       所以这里只在 train 折内、用固定种子。
    """
    train = split.train
    y = imp["y_click"].to_numpy()[train]
    keep = downsample_negatives(y, neg_pos_ratio=cfg.neg_pos_ratio,
                                seed=cfg.seed)
    rows = train[keep]
    y_sub = y[keep]

    # 数值块用 float32，SMOTE 的 k-NN 在 192 维上开销不小，
    # 抽 6 万个样本足够回答「有没有收益」这个问题
    X = bundle.imp_dense[rows].astype(np.float32)
    n_max = min(60_000, len(y_sub))
    sub = np.arange(n_max)
    # ⚠️ ``smote`` 返回的**只有合成样本**（全是正类），不是合并后的集合。
    #    直接拿它算「合成后的正率」会得到 1.0 这个荒唐的数 ——
    #    必须自己拼回去再统计。
    _, y_syn = smote(X[sub], y_sub[sub], k_neighbors=5, seed=cfg.seed)
    rep = smote_report(y_sub[sub], np.concatenate([y_sub[sub], y_syn]))
    rep["n_synthetic"] = int(len(y_syn))
    return {
        "n_train_after_downsample": int(len(y_sub)),
        "pos_rate_after_downsample": float(y_sub.mean()),
        "smote": rep,
        "sampled_rows": int(n_max),
    }


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)
    torch.set_num_threads(1)        # 逐位可复现，见 conftest 的说明

    data = load_data(cfg)
    imp = data["impressions"]
    split = load_split(cfg)
    bundle = load_bundle(cfg)

    with Timer(f"03 构造排序张量（{scale}）"):
        # 带 elapsed_h 建一次就够：DFM / elapsed 对照要用它，
        # 而排序模型不认识这个键（forward 签名里没有），
        # 所以给它的是一个**滤掉 elapsed_h 的视图**——
        # 只复制字典、不复制张量，79 万行下省掉的是一份完整的内存。
        full = DS.ranker_tensors(bundle, imp, cfg, with_elapsed=True)
        save_artifact(full, DATA_DIR / "ranker_tensors.pkl", cfg)
        tensors = {k: v for k, v in full.items() if k != "elapsed_h"}

    # 1:5 负下采样 —— **只在 train 折内做**，β 也相对 train 折算
    y_train = imp["y_click"].to_numpy()[split.train]
    keep = downsample_negatives(y_train, neg_pos_ratio=cfg.neg_pos_ratio,
                                seed=cfg.seed)
    train_rows = split.train[keep]
    beta = prior_shift(y_train, keep)

    ev = np.concatenate([split.valid, split.test])
    y_click_ev = imp["y_click"].to_numpy()[ev]

    banner("1:5 负下采样")
    log.info("正率 %.4f → %.4f，β（负样本保留率）= %.4f",
             y_train.mean(), y_train[keep].mean(), beta)
    log.info("∴ 推理时必须 logit += ln(β) = %+.4f；"
             "漏掉会让 eCPM 高估约 %.1f 倍，且不报任何错",
             np.log(beta), 1.0 / beta)

    # ---------------- 消融 ----------------
    results, models = [], {}
    for v in ("lr", "fm", "deepfm"):
        with Timer(f"03 训练 {v.upper()}（{scale}）"):
            model, hist = train_ranker(v, bundle, tensors, train_rows,
                                       split.valid, cfg)
        models[v] = model
        save_artifact(model, MODEL_DIR / f"ranker_{v}.pkl", cfg)

        out = predict(model, tensors, ev,
                      keys=("logit_ctr", "p_ctr", "p_cvr", "p_ctcvr"))
        # 先验校正：logit += ln(β)，再把校正后的概率 sigmoid 回来。
        # 这才是能直接乘进 eCPM 的那个数。
        p_ctr_fix = 1.0 / (1.0 + np.exp(-(out["logit_ctr"] + np.log(beta))))
        results.append({
            "variant": v,
            "epochs_run": len(hist.train_loss),
            "best_valid_auc": hist.best_auc,
            "auc_ctr": safe_auc(y_click_ev, out["logit_ctr"]),
            "auc_cvr": cvr_auc(out["p_cvr"], imp, ev),
            "auc_ctcvr": safe_auc(imp["y_conv_oracle"].to_numpy()[ev],
                                  out["p_ctcvr"]),
            "mean_p_ctr_raw": float(np.mean(out["p_ctr"])),
            "mean_p_ctr_corrected": float(np.mean(p_ctr_fix)),
            "ece_ctr": expected_calibration_error(y_click_ev, p_ctr_fix),
        })

    show_table(results, ["variant", "epochs_run", "best_valid_auc", "auc_ctr",
                         "auc_cvr", "auc_ctcvr", "mean_p_ctr_raw",
                         "mean_p_ctr_corrected", "ece_ctr"],
               "消融：LR < FM < DeepFM")

    abl = {r["variant"]: r for r in results}
    banner("先验校正")
    log.info("校正前平均 pCTR = %.4f，校正后 = %.4f（真实基率 %.4f）",
             abl["deepfm"]["mean_p_ctr_raw"],
             abl["deepfm"]["mean_p_ctr_corrected"], y_click_ev.mean())

    with Timer(f"03 SMOTE 消融对照（{scale}）"):
        sm = smote_ablation(bundle, imp, split, cfg)
    banner("SMOTE 消融对照")
    log.info("下采样后正率已经是 %.4f —— 此时『不平衡』问题已经不存在了",
             sm["pos_rate_after_downsample"])
    log.info("SMOTE 把它推到 %.4f（合成 %d 条），"
             "但合成样本落在原样本的凸组合上，多样性是**降低**的；"
             "所以主路径是下采样 + 先验校正，SMOTE 定位为消融对照",
             sm["smote"]["pos_rate_after"], sm["smote"]["n_synthetic"])

    # ---------------- 延迟反馈去偏 ----------------
    #
    # ⚠️ 这一组必须是**受控对照**：同一个架构、同一份数据、同一个种子，
    #    **只有损失函数（以及模型能看到的输入列）不同**。所以三个模型
    #    全部用 ``ElapsedFeatureModel`` / ``DFMModel`` 这条线，
    #    而不是拿 DeepFM 的 ``p_cvr`` 来比 —— 后者的偏差里混着
    #    「ESMM 的 CVR 头本来就没做概率校准」这件事，和延迟无关，
    #    拿它当对照组测出来的是另一回事。
    debias_rows = []
    with Timer(f"03 DFM 延迟反馈去偏（{scale}）"):
        mcfg = cfg.scaled(emb_dim=EMB)
        n_cat, n_dense = bundle.cat_fields, bundle.imp_dense.shape[1]
        # vanilla 的唯一区别：永远看不到「已经过了多久」这条信息，
        # 因此完全无法察觉标签是假的。把 elapsed_h 整列置零即可。
        zero_full = dict(full)
        zero_full["elapsed_h"] = torch.zeros_like(full["elapsed_h"])

        # 三个模型的数据、种子、超参**完全一致**，只有损失不同。
        # 训练折用**完整**的，不下采样（理由见文件顶部的 DFM_EPOCHS 注释）。
        dfm_kw = dict(epochs=DFM_EPOCHS, batch_size=DFM_BATCH, lr=LR,
                      weight_decay=cfg.weight_decay, label_key="y_conv",
                      auc_key="y_conv", prob_key="p", seed=cfg.seed,
                      patience=NO_EARLY_STOP)

        # DFM：额外挂一个标量延迟率头 λ(x)，把「还没转」和「不会转」分开。
        # 训练用的是**有偏的** y_conv —— 那正是线上唯一拿得到的那列标签。
        dfm_m = DFMModel(n_cat, n_dense, mcfg)
        torch.manual_seed(cfg.seed)
        dfm_hist = train_model(dfm_m, full, split.train, split.valid, **dfm_kw)
        save_artifact(dfm_m, MODEL_DIR / "dfm.pkl", cfg)

        # 廉价方案：把 elapsed 当普通特征直接喂进去，不做任何建模。
        # 跑它是为了回答「DFM 那套额外复杂度到底值不值」——
        # 如果这个廿行的版本能拿到 DFM 九成的效果，就不该上 DFM。
        elapsed_m = ElapsedFeatureModel(n_cat, n_dense, mcfg)
        torch.manual_seed(cfg.seed)
        train_model(elapsed_m, full, split.train, split.valid, **dfm_kw)

        vanilla_m = ElapsedFeatureModel(n_cat, n_dense, mcfg)
        torch.manual_seed(cfg.seed)
        train_model(vanilla_m, zero_full, split.train, split.valid, **dfm_kw)

        # 参照真值用 **168h 回补标签**（假负只剩 2%），而不是无限窗口的
        # oracle —— 后者是「上帝视角」，现实中永远拿不到，
        # 拿它当参照会把门槛定得比业务上可能的水平更严。
        y_ref = imp["y_conv_w168"].to_numpy()
        ref_rate = float(y_ref[ev].mean())
        online_rate = float(imp["y_conv"].to_numpy()[ev].mean())

        # elapsed 在训练折与评测折上的取值范围 —— 这两个区间几乎不重叠，
        # 而「elapsed 当普通特征」那个廉价方案的全部问题都出在这里。
        el = imp["elapsed_h"].to_numpy()
        el_tr = (float(el[split.train].min()), float(el[split.train].max()))
        el_ev = (float(el[ev].min()), float(el[ev].max()))
        extrapolated = float((el[ev] < el_tr[0]).mean())

        preds = {
            "vanilla（elapsed 恒 0，普通 BCE）": predict(
                vanilla_m, zero_full, ev, keys=("p",))["p"],
            "elapsed-as-feature（廿行廉价版）": predict(
                elapsed_m, full, ev, keys=("p",))["p"],
            "DFM（本方案）": predict(dfm_m, full, ev, keys=("p",))["p"],
        }
        y168_ev = y_ref[ev]
        clk_ev = imp["y_click"].to_numpy()[ev] == 1
        click_rate = float(y168_ev[clk_ev].mean())
        for name, p in preds.items():
            debias_rows.append({
                "model": name,
                "mean_p": float(p.mean()),
                "bias_vs_w168": float(p.mean()) / ref_rate - 1.0,
                "auc_vs_w168": safe_auc(y168_ev, p),
                # 点击级口径 —— **只作对照**，不是判据。DFM 估的是曝光级
                # 概率，拿它比点击条件的基率就是口径错了（见模块 docstring）。
                "mean_p_clicks": float(p[clk_ev].mean()),
                "bias_clicks": float(p[clk_ev].mean()) / click_rate - 1.0,
            })

    banner(f"延迟反馈去偏（全集 p̂ 对 168h 回补基率 {ref_rate:.5f}）")
    log.info("线上 y_conv 基率 %.5f，比 168h 回补低 %.1f%% —— 这就是假负",
             online_rate, (1 - online_rate / ref_rate) * 100)
    show_table(debias_rows, ["model", "mean_p", "bias_vs_w168",
                             "auc_vs_w168"], "受控对照：只有损失函数不同")
    show_table(debias_rows, ["model", "mean_p_clicks", "bias_clicks"],
               f"同一批预测，换成**点击级**口径（基率 {click_rate:.4f}）—— "
               f"只作对照，不是判据")
    log.info("⚠️ 两个口径的基率差 %.1f 倍（曝光级 %.5f vs 点击级 %.4f），"
             "拿错了口径就会得出「DFM 偏了 5 倍」的假结论。DFM 的模型形式"
             "包含点击这一步，它的参照是**曝光级**那一列 —— 也就是上面"
             "那个表。点击级这一份是**故意并排**放出来的反例。",
             click_rate / ref_rate, ref_rate, click_rate)
    log.info("⚠️ 「elapsed 当普通特征」那一行偏差高达 %+.0f%%，不是笔误，"
             "是**外推**：训练折的 elapsed 落在 [%.0f, %.0f]h，"
             "而评测折全部在 [%.0f, %.0f]h —— **零重叠**，"
             "评测集 %.0f%% 的行落在训练时从未见过的区间里。"
             "elapsed 在训练窗口内与 day 高度共线，模型把「日级市场漂移」"
             "学成了「elapsed 效应」，再外推出去就爆了。"
             "这正是「加一列特征」和「建一个延迟模型」的区别。",
             debias_rows[1]["bias_vs_w168"] * 100, el_tr[0], el_tr[1],
             el_ev[0], el_ev[1], extrapolated * 100)

    vanilla_bias = abs(debias_rows[0]["bias_vs_w168"])
    dfm_bias = abs(debias_rows[2]["bias_vs_w168"])

    # ---------------- 门槛 ----------------
    problems = []
    if not abl["fm"]["auc_ctr"] - abl["lr"]["auc_ctr"] > 0.015:
        problems.append(f"FM 没打赢 LR：{abl['fm']['auc_ctr']:.4f} vs "
                        f"{abl['lr']['auc_ctr']:.4f}")
    if not abl["deepfm"]["auc_ctr"] - abl["fm"]["auc_ctr"] > 0.005:
        problems.append(f"DeepFM 没打赢 FM：{abl['deepfm']['auc_ctr']:.4f} vs "
                        f"{abl['fm']['auc_ctr']:.4f} —— 先查真值的 "
                        f"alpha_3/alpha_4 是不是被调小了")
    if not abl["deepfm"]["auc_cvr"] > abl["fm"]["auc_cvr"]:
        problems.append("CVR 塔上 DeepFM 没打赢 FM")
    if not abs(abl["deepfm"]["mean_p_ctr_corrected"] - y_click_ev.mean()) < 0.01:
        problems.append(f"先验校正后 pCTR "
                        f"{abl['deepfm']['mean_p_ctr_corrected']:.4f} 偏离实际 "
                        f"{y_click_ev.mean():.4f}")
    if debias_rows[0]["bias_vs_w168"] > -0.10:
        problems.append(f"vanilla 的 p̂ 只低 {debias_rows[0]['bias_vs_w168']:.1%}，"
                        f"假负不够强 → DFM 的动机不成立（查延迟参数）")
    if not dfm_bias < vanilla_bias * 0.75:
        problems.append(f"DFM 的偏差 {dfm_bias:.1%} 没有明显好于 vanilla 的 "
                        f"{vanilla_bias:.1%}")

    metrics = {
        "scale": scale,
        "downsample": {"beta": beta, "prior_logit_shift": float(np.log(beta)),
                       "pos_rate_before": float(y_train.mean()),
                       "pos_rate_after": float(y_train[keep].mean()),
                       "n_train_rows": int(len(train_rows))},
        "ablation": results,
        "smote": sm,
        "debias": {"ref_rate_w168": ref_rate, "online_rate": online_rate,
                   "false_negative_frac": 1 - online_rate / ref_rate,
                   "elapsed_range_train": el_tr,
                   "elapsed_range_eval": el_ev,
                   "extrapolated_frac": extrapolated,
                   "models": debias_rows,
                   "dfm_epochs": int(len(dfm_hist.train_loss))},
        "problems": problems,
    }
    save_metrics("03_ranker", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：LR(%.4f) < FM(%.4f) < DeepFM(%.4f)；"
             "DFM 把 p̂ 偏差从 %.1f%% 压到 %.1f%%",
             abl["lr"]["auc_ctr"], abl["fm"]["auc_ctr"],
             abl["deepfm"]["auc_ctr"], vanilla_bias * 100, dfm_bias * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
