"""01 生成模拟数据并体检。

产出
----
``artifacts/data/raw_{scale}.pkl``     users / items / impressions
``artifacts/data/split_{scale}.pkl``   按时间切的 train/valid/test 行索引
``artifacts/metrics/01_data_{scale}.json``

**为什么这里是全项目的地基**

后面每一环的结论都建立在「这份数据里真的有可学的信号」之上：
DeepFM 要打赢 FM，得先有高阶交叉；DFM 要去偏，得先有延迟；
ZILN 要算准期望金额，得先有零膨胀。这些性质**必须在这里被验证**，
否则后面所有「模型 A 比模型 B 好 0.02」都是在噪声里比大小。

三个必须体检的量
----------------
1. **基率**：CTR ≈ ``target_ctr``、CVR|click ≈ ``target_cvr``。
   这是校准器（二分法调截距）的结果，偏了说明校准没收敛。
2. **假负率随窗口单调下降**：6h > 24h > 72h > 168h。
   延迟是三分量混合（立即 / 同会话 / 长尾），窗口越长补回的越多。
   这条曲线不单调就说明延迟抽样写错了 —— 而 DFM 的整个动机
   就建立在「窗口截断造成假负」之上。
3. **隐变量没有泄漏到特征列**：``_true_*`` 开头的列绝不能出现在
   模型输入里，且没有任何一列与真值 logit 的相关系数接近 1。
   这是本项目最容易「做出 0.99 AUC」的坑。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from scripts._common import (DATA_DIR, Timer, banner, cfg_for, log,  # noqa: E402
                             parse_scale, save_artifact, save_metrics,
                             show_table)
from src.common.seed import set_seed                        # noqa: E402
from src.data.generator import DataGenerator                # noqa: E402
from src.data.splits import label_stats, time_split         # noqa: E402


def false_negative_table(imp: pd.DataFrame) -> list[dict]:
    """每个观测窗口的假负率。

    假负 = 「这个人其实会转化，但在窗口内还没转」。
    分母用 ``y_conv_oracle``（无限窗口的真值），所以这是**定义上**的
    假负，不含模型误差 —— 衡量的是「数据本身有多脏」。
    """
    n_true = int(imp["y_conv_oracle"].sum())
    rows = []
    for w in (6, 24, 72, 168):
        pos = int(imp[f"y_conv_w{w}"].sum())
        rows.append({
            "window_h": w,
            "positives": pos,
            "fn_rate": (n_true - pos) / n_true,
            "kept": pos / n_true,
        })
    # 线上真实标签：24h 窗口 + 数据末尾截断
    pos = int(imp["y_conv"].sum())
    rows.append({"window_h": "online",
                 "positives": pos,
                 "fn_rate": (n_true - pos) / n_true,
                 "kept": pos / n_true})
    return rows


def leakage_check(imp: pd.DataFrame) -> dict:
    """隐变量泄漏检查。

    ⚠️ 判据不是「特征里有没有 ``_true_`` 列」—— 那些列**本来就该在这里**，
       它们是模拟数据的地面真值，是后面所有评测（oracle 标签、AUC gap、
       PIT 检验）的唯一依据。真正要防的是「它们被喂进模型」，
       而模型的输入是**特征流水线**产出的 bundle，不是这张原始表。
       所以「原始表里有 ``_true_``」是**信息**，「bundle 里有」才是**故障**
       ——后者在 02 步和 ``tests/test_02_features.py`` 里卡。

       这里能抓的是另一半：某个**可观测**列被写成了 latent 的无噪声副本。
       真发生这种事时，模型的 AUC 会直接飞到 0.95 以上，
       整条链路看着漂亮、实际什么也没证明。

    ⚠️ 阈值是「**接近 1**」而不是「显著」。1 万行时 r=0.05 的 p 值就已经
       小于 0.001 了，拿显著性去卡，任何有微弱真实信号的列都会被误报。
    """
    true_cols = [c for c in imp.columns if c.startswith("_true_")]
    skip = set(true_cols) | {"elapsed_h"}
    feat_cols = [c for c in imp.columns
                 if c not in skip and not c.startswith("y_")]
    logit = imp["_true_logit_ctr"].to_numpy(dtype=np.float64)
    worst, worst_col = 0.0, ""
    for c in feat_cols:
        v = imp[c]
        if not pd.api.types.is_numeric_dtype(v):
            continue
        x = v.to_numpy(dtype=np.float64)
        if np.std(x) == 0 or np.std(logit) == 0:
            continue
        r = abs(float(np.corrcoef(x, logit)[0, 1]))
        if r > worst:
            worst, worst_col = r, c
    return {
        "oracle_columns": true_cols,
        "n_features_scanned": len(feat_cols),
        "max_abs_corr_with_true_logit": worst,
        "max_corr_column": worst_col,
        "leaked": bool(worst > 0.95),
    }


def delay_mixture(imp: pd.DataFrame) -> dict:
    """延迟的分布形态 —— DFM 的指数延迟假设是否合理，看这个。"""
    d = imp.loc[imp["_true_y_conv"] == 1, "_true_delay_h"].to_numpy()
    return {
        "n_conversions": int(d.size),
        "median_h": float(np.median(d)),
        "p90_h": float(np.percentile(d, 90)),
        "p99_h": float(np.percentile(d, 99)),
        "frac_within_1h": float(np.mean(d <= 1.0)),
        "frac_beyond_24h": float(np.mean(d > 24.0)),
    }


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)

    with Timer(f"01 生成数据（{scale}）"):
        data = DataGenerator(cfg).generate()
        users, items, imp = data["users"], data["items"], data["impressions"]
        split = time_split(imp, cfg)
        save_artifact(data, DATA_DIR / "raw.pkl", cfg)
        save_artifact(split, DATA_DIR / "split.pkl", cfg)

    # 校准质量：全表的基率（CVR 的分母是**点击**，和全项目口径一致）
    #
    # ⚠️ 这里必须用 ``y_conv_oracle`` 对目标，**不能**用线上那列 ``y_conv``。
    #    校准器（``_bisect_intercept``）调的是**真值**基率，而 ``y_conv``
    #    是「24h 窗口 + 数据末尾截断」后的有偏标签，它天然比真值低 ——
    #    低的那个比例**就是**假负率。拿它去对目标，会把「延迟反馈这个
    #    核心难点」误报成「基率没校准」，然后去乱调生成器。
    n_click = int(imp["y_click"].sum())
    ctr = float(imp["y_click"].mean())
    cvr = float(imp["y_conv_oracle"].sum()) / n_click if n_click else 0.0
    cvr_online = float(imp["y_conv"].sum()) / n_click if n_click else 0.0

    fn_rows = false_negative_table(imp)
    leak = leakage_check(imp)
    delay = delay_mixture(imp)

    splits = [{"split": name, **label_stats(imp, idx)}
              for name, idx in (("train", split.train), ("valid", split.valid),
                                ("test", split.test))]
    banner("基率校准")
    log.info("CTR = %.4f（目标 %.4f，偏差 %+.1f%%）", ctr, cfg.target_ctr,
             (ctr / cfg.target_ctr - 1) * 100)
    log.info("CVR|click（oracle）= %.4f（目标 %.4f，偏差 %+.1f%%）", cvr,
             cfg.target_cvr, (cvr / cfg.target_cvr - 1) * 100)
    log.info("CVR|click（线上 y_conv）= %.4f，比 oracle 低 %.1f%%"
             " —— 这个缺口就是假负样本，是 DFM 要修的东西",
             cvr_online, (1 - cvr_online / cvr) * 100 if cvr else 0.0)

    show_table(fn_rows, ["window_h", "positives", "fn_rate", "kept"],
               "假负率随观测窗口的变化（分母 = oracle 正样本）")
    show_table(splits, ["split", "n", "n_click", "n_conv", "ctr", "cvr"],
               "时间切分")
    banner("延迟分布")
    log.info("%s", delay)
    banner("泄漏检查")
    log.info("地面真值列 %d 个（**应当存在**，是评测依据，不进模型）：%s",
             len(leak["oracle_columns"]), ", ".join(leak["oracle_columns"]))
    log.info("扫描 %d 个可观测列，与真值 logit 的最大 |corr| = %.4f（列 %s）",
             leak["n_features_scanned"], leak["max_abs_corr_with_true_logit"],
             leak["max_corr_column"])

    # ---- 体检结论：不满足就**直接失败**，不要把脏数据带进下游 ----
    problems = []
    if not abs(ctr - cfg.target_ctr) < 0.005:
        problems.append(f"CTR {ctr:.4f} 偏离目标 {cfg.target_ctr}")
    if not abs(cvr - cfg.target_cvr) < 0.010:
        problems.append(f"CVR(oracle) {cvr:.4f} 偏离目标 {cfg.target_cvr}")
    fns = [r["fn_rate"] for r in fn_rows[:4]]
    if not all(a > b for a, b in zip(fns, fns[1:])):
        problems.append(f"假负率不随窗口单调下降：{[round(x, 4) for x in fns]}")
    if leak["leaked"]:
        problems.append(f"可观测列疑似 latent 副本：{leak['max_corr_column']} "
                        f"|corr|={leak['max_abs_corr_with_true_logit']:.3f}")

    metrics = {
        "scale": scale,
        "n_users": int(len(users)),
        "n_items": int(len(items)),
        "n_impressions": int(len(imp)),
        "ctr": ctr, "cvr": cvr, "cvr_online": cvr_online,
        "false_negative_gap": (1 - cvr_online / cvr) if cvr else 0.0,
        "target_ctr": cfg.target_ctr, "target_cvr": cfg.target_cvr,
        "false_negative": fn_rows,
        "delay": delay,
        "leakage": {"oracle_columns": leak["oracle_columns"],
                    "n_features_scanned": leak["n_features_scanned"],
                    "max_abs_corr_with_true_logit":
                    leak["max_abs_corr_with_true_logit"],
                    "max_corr_column": leak["max_corr_column"],
                    "leaked": leak["leaked"]},
        "splits": splits,
        "problems": problems,
    }
    save_metrics("01_data", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：基率已校准、假负随窗口单调、无隐变量泄漏")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
