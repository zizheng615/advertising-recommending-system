"""08 真实环境出价模拟：把训练好的模型接进拍卖，智能出价 vs 固定出价。

产出
----
``artifacts/metrics/08_bidding_sim_{scale}.json``

**和 06 的区别 —— 这才是「真实环境」四个字的分量**

06 用的是 ``_synthetic_values``：真值加一个**人为设定的**对数正态噪声
（``noise_sigma=0.35``）当预测，模型误差是假装的。那一步检验的是控制环。

这一步把**真正训出来的模型**接进去：DeepFM 的 pCTR、DeepFM CVR 头 / DFM
的 pCVR、ZILN 的 E[金额]，按在线口径乘成 eCPM。进拍卖的不再是
「真值 + 白噪声」，而是**带真实误差结构的预测** —— 误差和特征相关、
有些人群被系统性高估、有些被低估。这正是出价系统上线后面对的东西。

**四个必须做对、做错又不报错的地方**

1. **先验校正只作用于 pCTR，不作用于 pCVR**。负下采样保留全部点击、
   只丢曝光负样本（``downsample_negatives(y_click)``），所以**点击率**
   从 5% 变成 17%，而**点击后的转化率**完全没变。两个头的口径因此不同：
   * pCTR 需要 ``logit += ln β``（β=0.2559 → −1.3629）；
   * pCVR 的损失本来就在**点击样本**上算（ESMM 的第三项），条件事件
     就是「点击」，下采样没有改变它 —— **再校正一次就是错的**。
   脚本把「顺手也给 pCVR 加一次」当成消融跑出来（见 ``pCVR 先验校正消融``）。
2. **每个头有自己的口径，混了口径不会报错，只会让系数静默偏掉**：
   * pCTR → 目标 ``y_click``（分母是**曝光**）
   * DeepFM 的 pCVR → 目标 ``y_conv_w168``，但**只在点击上取均值**
     （分母是**点击**）
   * DFM 的 p → 它直接估 ``P(click)·P(conv|click)``，所以目标要在
     **全部曝光**上取 —— 同一个 ``y_conv_w168``，两种取法差约 20 倍
3. **LTV 的口径**：ZILN 的 ``E[y] = π·exp(μ + σ²/2)``，不是 ``π·exp(μ)``，
   也不能用「这个人实际付了多少钱」（90% 是 0，出价会全变成 0）。
4. **预算必须是外生的**。第一版按 06 的老办法用**预测**的无约束花费反解
   ``budget = u_pred / 3.6``，结果预算本身随预测水平变 —— 于是「校准前后
   有什么变化」这个问题被污染了：价值涨 30%，而预算同时涨了 50%，两者
   根本分不开。更糟的是小预算把 B5 的 α* 压到 ``alpha_min`` 的下界上，
   上帝视角先被裁了一刀。现在改成 ``budget = u_true / 3.6``（用**真实**
   价值反解），三次仿真共用同一个预算 —— 预算由广告主定，不由模型定，
   这才是真实情况。

**预注册的预期（先说，再验 —— 不然容易看到什么就信什么）**

> **预算固定时，把三个头同时乘一个常数，对「按价值比例出价」的买手
> 是恒等变换。**

理由：``build_policies`` 用 ``solve_alpha`` 反解水位，使
``E[花费(α·v)] = plan/n``。v 整体乘 c ⇒ α 精确地缩到 α/c ⇒
``出价 = (c·v)·(α/c) = v·α``，逐请求相同，于是消耗轨迹、水位轨迹、
赢下的曝光全都相同。

**这条恒等性有一个已知的例外，而且只有一个**：``budget_gate_scale``
（预算硬闸门）把「这一批的**期望**扣费」压在剩余预算之内，而
``expected_spend_cpm`` 对出价是**超线性**的 —— 出价整体乘 c，期望扣费
乘的比 c 多，于是闸门对大出价的压缩更狠。**它是整条链路里唯一不和
c 交换的部件**，所以脚本把它单独拎出来量：

| 量 | 期望 | 说明 |
|---|---|---|
| 水位之比 ``α_raw / α_lvl`` | 精确等于 c（±2%）| 检验 solve_alpha 的协变性 |
| B4 价值 \|差\| | 闸门没触发时 < 0.5% | 恒等性 |
| B4 价值**符号** | 闸门触发时**校准侧不劣** | 闸门惩罚虚高出价，方向可预注册 |

再加上水位是否触到 ``[alpha_min, alpha_max]`` 的边界（触了就说明控制器
饱和，此时水平校准**真的**要紧），一共四个判据，脚本逐条报出实测值。

**这条恒等性推出一个反直觉的结论，也正是这一步最有价值的一条**

> **逐头把边际均值校准对，并不能让乘积无偏。**

实测：校准后四个头的相对误差都从 −8.5%/−46.9%/−29.1%/−22.0% 收到
−1.7%/−11.6%/−13.1%/+1.1%，但 ``E[pCTR·pCVR·LTV]`` 的聚合偏差仍有
50% 量级（见输出）。原因是 ``E[XYZ]`` 依赖**联合分布**，不是三个边际
期望的乘积 —— 三个头各自对上均值，乘积照样偏。

而且按上面的恒等性，**这 50% 的乘积偏差对出价毫无影响**。所以正确的
说法是：水平校准的价值在于**对外可解释**（水位数值、风控阈值、跨渠道
比价对得上），**不在于**让出价更准。让出价更准的只有一件事 —— **排序**。

**那 B4/B5 只有一半不到是什么问题？——排序，不是水平。**
B5（上帝视角）用真值出价、排序完美；B4 用的是预测。差距全部来自
``pred_value`` 与 ``true_value`` 的**排序不完美**，而这**不是任何一个
常数能修的**（常数的效果被上面的恒等性抵掉了）。所以这里如实报出，并把
「重训模型 / 把乘积的排序修好」列为后续工作 —— 而不是把门槛调到实测值
以下来「通过」。

⚠️ 但「排序不好」要用**对的那个指标**来说。脚本原先报的是全量
Spearman，实测它跟拍卖结果**反向**：esmm 的 Spearman 更高（0.4880 >
0.3826），B4/B5 却更低（0.2498 < 0.4620）。这不是噪声 —— 按比例出价对
v 的**单调变换免疫**，而 Spearman 对单调变换**完全免疫**，它测的恰好是
拍卖看不见的那部分。现在改报三个同向的指标（弹性 γ、紧预算截获率、
外加留作反例的全量 Spearman），推导见 ``rank_diag``。

**两个 q 的构造方式（顺带做一次真实的模型选择）**

| 变体 | pred_q | 说明 |
|---|---|---|
| ``esmm`` | pCTR × pCVR（两个头相乘） | 线上主路径，ESMM 结构 |
| ``dfm`` | DFM 直接输出的 P(转化) | 一条链路估到底，跳过相乘 |

两者都是合法估计，谁在拍卖里更值钱只能靠仿真回答 —— 这正是这一步存在的
意义。它们的**水平**差 2 倍，而按上面的恒等性，这不构成不公平。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from scipy.stats import spearmanr                            # noqa: E402

from scripts._common import (MODEL_DIR, Timer, _scaled,      # noqa: E402
                             banner, cfg_for, load_bundle, load_data,
                             load_metrics, load_pickle, load_ranker_tensors,
                             load_split, log, parse_scale, save_metrics,
                             show_table)
from src.bidding.auction import MarketConfig, expected_spend_cpm  # noqa: E402
from src.bidding.bidder import build_policies, ecpm          # noqa: E402
from src.bidding.controller import diurnal_weights           # noqa: E402
from src.bidding.sim_day import make_stream, run_all_policies  # noqa: E402
from src.common.seed import set_seed                        # noqa: E402
from src.metrics import regression as G                     # noqa: E402
from src.models import dataset as DS                        # noqa: E402
from src.models.train import predict                        # noqa: E402
from src.models.ziln import ziln_predict_arrays             # noqa: E402

N_REQUESTS = 200_000
SIM_SEED = 2024
BUDGET_TENSION = 3.6          # 与 06 一致：约束必须真正 binding


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --------------------------------------------------------------------------
# 头的定义（口径只在这里写一次）
# --------------------------------------------------------------------------


def head_specs() -> list[dict]:
    """四个头，以及**各自正确的口径**。

    ``on_clicks`` 是这里唯一真正要紧的字段：它决定校准系数拿谁当分母，
    也决定报表里那一列「实际均值」是什么。写错的后果不是报错，而是
    系数静默偏掉 —— 同一个 ``y_conv_w168``，在全部曝光上平均是 0.0030，
    在点击上平均是 0.0575，**差 19 倍**。
    """
    return [
        {"key": "p_ctr", "name": "pCTR", "src": "DeepFM，已做先验校正",
         "target": "y_click", "on_clicks": False, "is_prob": True,
         "target_name": "y_click（曝光）"},
        {"key": "p_cvr", "name": "pCVR", "src": "DeepFM 的 CVR 头",
         "target": "y_conv_obs", "on_clicks": True, "is_prob": True,
         "target_name": "y_conv_w168（点击）"},
        {"key": "p_cvr_dfm", "name": "P(转化)", "src": "DFM，一条链路估到底",
         "target": "y_conv_obs", "on_clicks": False, "is_prob": True,
         "target_name": "y_conv_w168（曝光）"},
        {"key": "ltv", "name": "E[金额]", "src": "ZILN，π·exp(μ+σ²/2)",
         "target": "pay", "on_clicks": False, "is_prob": False,
         "target_name": "pay_amount（曝光）"},
    ]


def _spec(key: str) -> dict:
    return next(s for s in head_specs() if s["key"] == key)


def _mask(pool: dict, spec: dict):
    return (pool["y_click"] > 0) if spec["on_clicks"] else slice(None)


def _corrected(pool: dict, spec: dict, coef: dict | None) -> np.ndarray:
    v = pool[spec["key"]]
    if coef is None:
        return v
    v = v * coef[spec["key"]]
    return np.clip(v, 0.0, 1.0) if spec["is_prob"] else np.maximum(v, 0.0)


# --------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------


def load_models(cfg) -> dict:
    def need(name: str, script: str) -> Path:
        p = _scaled(MODEL_DIR / name, cfg)
        if not p.exists():
            raise SystemExit(f"缺少 {p.name} —— 先运行 {script}")
        return p

    return {
        "deepfm": load_pickle(need("ranker_deepfm.pkl",
                                   "scripts/03_train_ranker.py")),
        "dfm": load_pickle(need("dfm.pkl", "scripts/03_train_ranker.py")),
        "ziln": load_pickle(need("ziln.pkl", "scripts/05_ziln.py")),
        "beta": float(load_metrics("03_ranker", cfg)["downsample"]["beta"]),
    }


def predict_heads(cfg, models: dict, idx: np.ndarray, bundle, imp, users) -> dict:
    """在一个折上算出四个头的预测，外加评标用的真值。

    **测试折和验证折走的是同一条代码路径** —— 两处各写一遍的话，
    校准系数会在某个看不见的地方对不上口径，而且不会有任何报错。
    """
    full = load_ranker_tensors(cfg)
    # DeepFM 的 bank 里没有 ``elapsed_h`` 这个域；而 ``predict`` 会把字典里
    # **所有**非标签张量当关键字参数传进 forward，多一列就是 TypeError。
    thin = {k: v for k, v in full.items() if k != "elapsed_h"}

    dm = predict(models["deepfm"], thin, idx, keys=("logit_ctr", "p_cvr"))
    dfm = predict(models["dfm"], full, idx, keys=("p",))

    # ⚠️ 先验校正只加在 pCTR 上，理由见模块 docstring 第 1 条。
    p_ctr = sigmoid(dm["logit_ctr"] + np.log(models["beta"]))

    n_users = len(bundle.user_cat)
    zt = DS.ziln_user_tensors(bundle, users,
                              np.arange(n_users, dtype=np.int64), cfg)
    models["ziln"].eval()
    with torch.no_grad():
        zp = ziln_predict_arrays(models["ziln"](zt["cat"], zt["dense"]))
    uid = imp["user_id"].to_numpy()[idx]

    return {
        "p_ctr": p_ctr,
        "p_cvr": dm["p_cvr"],                # DeepFM 的 CVR 头（点击条件）
        "p_cvr_dfm": dfm["p"],               # DFM 的 P(转化)（曝光级）
        "ltv": zp["expected"][uid],
        "logit_cvr_raw": dm["p_cvr"],        # 仅用于先验校正消融
        # --- 现实中可观测的校准目标 ---
        "y_click": imp["y_click"].to_numpy()[idx],
        "y_conv_obs": imp["y_conv_w168"].to_numpy()[idx],   # 168h 回填
        "pay": users["pay_amount"].to_numpy(dtype=np.float64)[uid],
        # --- 只在模拟里有的真值，**只用于打分**，绝不进任何策略 ---
        "true_q": (sigmoid(imp["_true_logit_ctr"].to_numpy()[idx])
                   * sigmoid(imp["_true_logit_cvr"].to_numpy()[idx])),
        "true_ltv": users["expected_ltv"].to_numpy(dtype=np.float64)[uid],
    }


def fit_level_correction(valid: dict) -> dict:
    """在**验证折**上拟合每个头的水平系数。

    最朴素的「让均值对上」：``c = 目标均值 / 预测均值``。它不动排序
    （AUC / Gini / Spearman 一个都不改），只把水平搬正。

    目标全部用**现实中拿得到的量**：
      * pCTR → ``y_click``（曝光回执，立刻可得）
      * pCVR → ``y_conv_w168``（168h 回填，项目里公认的现实参照；
        **不用** ``_true_*`` —— 那在线上永远拿不到）
      * LTV  → ``pay_amount``（充值流水，最迟 T+1 可得）

    ⚠️ 只在**验证折**上拟合。在测试折上拟合 = 拿答案调参，报出来的
       改善全是假的。验证折在前、测试折在后，所以残差里必然含一段
       **跨期漂移** —— 实测残差 1%~12%，就是这么来的。
    """
    coef = {}
    for spec in head_specs():
        m = _mask(valid, spec)
        coef[spec["key"]] = (float(valid[spec["target"]][m].mean())
                             / float(valid[spec["key"]][m].mean()))
    return coef


def head_table(pool: dict, coef: dict | None) -> list[dict]:
    rows = []
    for spec in head_specs():
        m = _mask(pool, spec)
        pred = _corrected(pool, spec, coef)
        act = float(pool[spec["target"]][m].mean())
        p = float(pred[m].mean())
        rows.append({"头": spec["name"], "来源": spec["src"],
                     "口径": spec["target_name"], "预测均值": p,
                     "实际均值": act, "相对误差": p / act - 1.0})
    return rows


# --------------------------------------------------------------------------
# 变体与仿真
# --------------------------------------------------------------------------


def build_q(variant: str, pool: dict, coef: dict | None) -> np.ndarray:
    """按变体把各头拼成 ``q = P(转化)``。"""
    if variant == "esmm":
        return (_corrected(pool, _spec("p_ctr"), coef)
                * _corrected(pool, _spec("p_cvr"), coef))
    if variant == "dfm":
        return _corrected(pool, _spec("p_cvr_dfm"), coef)
    raise ValueError(f"未知变体 {variant!r}")


def value_source_of(pool: dict, q: np.ndarray, ltv: np.ndarray):
    """把预测池包成 ``make_stream`` 的 ``value_source``。

    池子是**固定**的测试折样本（每个真实的 (用户, 素材, 上下文) 一行），
    而 ``make_stream`` 每次要 ``n`` 个请求的价值，所以这里做**有放回**的
    重采样 —— 相当于把测试折这段时间的流量重复投放。用重采样而不是取模
    循环，是为了让每次抽到的构成有随机性，不跟池子的原始行序绑定。
    三次仿真用同一个 ``SIM_SEED``，所以抽到的请求序列**逐条相同** ——
    这是配对比较，策略间差异里不会混进流量抽样噪声。

    真值和预测走**同一个下标** —— 三个口径必须配对，否则「预测买到的」
    和「真值打分的」就不是同一批曝光了。
    """
    n_pool = pool["n_pool"]
    tq, tl, rl = pool["true_q"], pool["true_ltv"], pool["pay"]

    def source(n: int, rng: np.random.Generator):
        i = rng.integers(0, n_pool, size=n)
        return tq[i], tl[i], rl[i], q[i], ltv[i]

    return source


def run_sim(cfg, market, traffic, pool: dict, q, ltv,
            budget: float | None = None) -> dict:
    stream = make_stream(cfg, market, n_requests=N_REQUESTS, seed=SIM_SEED,
                         traffic=traffic,
                         value_source=value_source_of(pool, q, ltv))
    if budget is None:
        # 用**真实**价值反解预算：外生、三次仿真共用、且保证约束 binding。
        # 用预测反解会让预算跟着模型水平走（见模块 docstring 第 4 条）。
        u_true = float(expected_spend_cpm(stream.true_value_cpm,
                                          market).sum() * cfg.cpm_to_impression)
        budget = round(u_true / BUDGET_TENSION, -1)
    pols = build_policies(cfg, market,
                          pred_value_cpm=stream.pred_value_cpm,
                          true_value_cpm=stream.true_value_cpm,
                          budget=budget, traffic=traffic,
                          plan_weights=stream.plan_weights)
    res = {r.name: r for r in run_all_policies(stream, pols, cfg, market,
                                               budget=budget, seed=SIM_SEED)}
    return {"res": res, "budget": float(budget),
            "pred_value_cpm": stream.pred_value_cpm,
            "true_value_cpm": stream.true_value_cpm}


def saturates(r, cfg) -> bool:
    """B4 的水位是否贴到了下界/上界 —— 恒等性成立与否的**唯一**判据。

    留一点余量：反比缩放后落在边界附近不能算饱和，要真正贴住才算。
    """
    a = r.alpha_path
    lo, hi = cfg.alpha_min, cfg.alpha_max
    return bool((a <= lo * (1 + 1e-3)).any() or (a >= hi * (1 - 1e-3)).any())


def rank_diag(pred_v: np.ndarray, true_v: np.ndarray,
              buy_frac: float) -> dict:
    """排序质量的三个指标，以及**为什么全量 Spearman 在这里是错的**。

    实测（mini 档，两个变体）：

    | 指标 | esmm | dfm | 谁该赢 |
    |---|---|---|---|
    | 全量 Spearman | 0.4880 | 0.3826 | dfm ← **反向** |
    | 弹性 γ | 0.4834 | 0.7943 | dfm ✅ |
    | 截获率 @1.2% | 0.2951 | 0.3930 | dfm ✅ |

    全量 Spearman 反向不是噪声，而是它**测错了东西**：出价是
    ``b = α·v``，**按比例**而不是按名次。把 v 做任何单调变换，α 会恰好
    反向补偿（恒等性判据①证明的就是这件事），逐请求出价一个字都不变 ——
    而 Spearman 只排名次，**对单调变换完全免疫**。它测的正是拍卖
    **看不见**的那部分。拿它给「拍卖只在乎排序」作证会得出反的结论。

    真正决定买到什么的是预测的**离散程度**：γ 是 ``log(真值)`` 对
    ``log(预测)`` 的回归斜率。γ = 1 说明预测的高低比例跟真值一致；
    γ < 1 说明预测**过度发散** —— 头部出价被抬到远超真值的水平，
    预算全砸在那几个虚高的出价上。esmm 的 v 是**三个各自带噪的概率
    相乘**，噪声跟着相乘放大，所以 γ 只有 0.48；dfm 一条链路估到底，
    γ = 0.79。分位比值曲线（预测分十档，档内 真值均值/预测均值）里
    看得最直白：esmm 从 1.9e7 一路降到 3.7，dfm 从 221 降到 2.8 ——
    两万倍的离散差距。

    截获率是同一件事的直读版：预算只买得起池子里 ``buy_frac`` 那一段
    （实测 B4 的 win_rate ≈ 1.2%），所以**只在这一段上算**「按预测取前
    f 拿到的真值 / 按真值取前 f 本可拿到的」。⚠️ 它只在**紧预算**下
    有效，f 放到 5% 以上两个变体就分不开了（实测 cap@5%：esmm 0.6065
    > dfm 0.5829，已经反向）—— 预算越紧，排序越值钱，这本身就是
    智能出价要解决的问题的另一种说法。
    """
    pos = (pred_v > 0) & (true_v > 0)          # log-log 只在正数上做
    g, _ = np.polyfit(np.log(pred_v[pos]), np.log(true_v[pos]), 1)
    k = max(1, int(round(true_v.size * buy_frac)))
    got = true_v[np.argsort(-pred_v)[:k]].sum()
    ideal = true_v[np.argsort(-true_v)[:k]].sum()
    return {"spearman_all": float(spearmanr(pred_v, true_v).statistic),
            "loglog_elasticity": float(g),
            "capture_at_buy_frac": float(got / ideal),
            "buy_frac": float(k / true_v.size)}


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)
    torch.set_num_threads(1)

    models = load_models(cfg)
    data = load_data(cfg)
    imp, users, bundle = data["impressions"], data["users"], load_bundle(cfg)
    split = load_split(cfg)

    with Timer("08 推理四个头（验证折 + 测试折）"):
        va = predict_heads(cfg, models, split.valid, bundle, imp, users)
        te = predict_heads(cfg, models, split.test, bundle, imp, users)
    pool = dict(te, n_pool=int(split.test.size))

    cols = ["头", "来源", "口径", "预测均值", "实际均值", "相对误差"]
    banner(f"接入真实模型（测试折 {pool['n_pool']} 条曝光）")
    log.info("先验校正 β = %.4f → pCTR 的 logit += ln β = %+.4f。"
             "**pCVR 不做这一步** —— 它的损失在点击样本上算，"
             "条件事件没被下采样动过", models["beta"], np.log(models["beta"]))
    raw_rows = head_table(pool, None)
    show_table(raw_rows, cols, "① 未校准")

    coef = fit_level_correction(va)
    log.info("② 验证折上拟合的水平系数（只搬水平，排序一点没动）：%s",
             "，".join(f"{s['name']} ×{coef[s['key']]:.3f}"
                       for s in head_specs()))
    cal_rows = head_table(pool, coef)
    show_table(cal_rows, cols, "② 校准后")
    t = {r["头"]: abs(r["相对误差"]) for r in cal_rows}

    # ---------------- 先验校正的消融：顺手也给 pCVR 加一次会怎样 ----------------
    clk = pool["y_click"] > 0
    cvr = pool["p_cvr"]
    cvr_wrong = sigmoid(np.log(cvr / (1 - cvr)) + np.log(models["beta"]))
    act_cvr = float(pool["y_conv_obs"][clk].mean())
    log.info("③ 消融：如果顺手把先验校正也加到 pCVR 上，pCVR 会从 %.4f "
             "变成 %.4f（相对误差 %+.1f%% → %+.1f%%）—— 因为 pCVR 的"
             "条件事件是**点击**，而下采样只动了曝光负样本。"
             "这是个对称性陷阱：两个头长得一样，口径不一样",
             float(cvr[clk].mean()), float(cvr_wrong[clk].mean()),
             (float(cvr[clk].mean()) / act_cvr - 1) * 100,
             (float(cvr_wrong[clk].mean()) / act_cvr - 1) * 100)

    log.info("⚠️ 概率头的**排序**其实都不差（DFM 对 168h 回填标签的 AUC "
             "= 0.918），但**水平各偏一档**。AUC / Gini 对这一类错误完全"
             "失明 —— 它们只看次序不看水平，而 eCPM 是三个头相乘。"
             "所以水平偏差只能靠**聚合偏差**（一个绝对量）来抓。")

    market = MarketConfig.from_mean(cfg.market_cpm_mean, cfg.market_sigma_m)
    traffic = diurnal_weights(cfg.t_slots)

    # ---------------- 三遍仿真，共用同一个外生预算 ----------------
    runs: dict[str, dict] = {}
    budget = None
    for tag, variant, co in (("esmm_raw", "esmm", None),
                             ("esmm_level", "esmm", coef),
                             ("dfm_raw", "dfm", None)):
        q = build_q(variant, pool, co)
        ltv = _corrected(pool, _spec("ltv"), co)
        with Timer(f"08 {tag} 出价模拟（{scale}）"):
            runs[tag] = run_sim(cfg, market, traffic, pool, q, ltv, budget)
        budget = runs[tag]["budget"]          # 第一次算出来，后面沿用
        runs[tag]["variant"] = variant
        runs[tag]["calibrated"] = co is not None
        # eCPM 的**乘积**聚合偏差：拿真实 q·LTV 当参照
        runs[tag]["ecpm_bias"] = G.aggregate_bias(
            pool["true_q"] * pool["true_ltv"] * 1000,
            ecpm(q, np.ones_like(q), ltv))
        # 排序质量的三个指标。买入比例取该变体 B4 的实测 win_rate ——
        # 预算真正买到的那一段才是排序要负责的那一段。
        runs[tag].update(rank_diag(q * ltv, pool["true_q"] * pool["true_ltv"],
                                   runs[tag]["res"]["B4_lambda_pid"].win_rate))

    for tag in runs:
        r = runs[tag]
        banner(f"{tag}：B0–B5（预算 {r['budget']:.0f} 元，三次仿真共用；"
               f"排序：弹性 γ = {r['loglog_elasticity']:.4f}、"
               f"截获率@{r['buy_frac']:.1%} = {r['capture_at_buy_frac']:.4f}、"
               f"全量 Spearman = {r['spearman_all']:.4f}）")
        show_table([{"策略": v.name, "花费": v.spend,
                     "预算误差": v.overspend_frac, "期望价值": v.expected_value,
                     "ROI": v.roi, "pacing": v.pacing_rmse,
                     "水位峰值": float(v.alpha_path.max()),
                     "闸门命中": v.gate_hits}
                    for v in r["res"].values()],
                   ["策略", "花费", "预算误差", "期望价值", "ROI", "pacing",
                    "水位峰值", "闸门命中"], "智能出价 vs 固定出价")

    # ---------------- 恒等性：四个判据，逐条量 ----------------
    raw, lvl = runs["esmm_raw"], runs["esmm_level"]
    b4_raw = raw["res"]["B4_lambda_pid"]
    b4_lvl = lvl["res"]["B4_lambda_pid"]
    sat_raw, sat_lvl = saturates(b4_raw, cfg), saturates(b4_lvl, cfg)
    gated = b4_raw.gate_hits > 0 or b4_lvl.gate_hits > 0
    gap = abs(b4_lvl.expected_value / b4_raw.expected_value - 1.0)
    # 三个头的水平系数连乘 = 预测价值的整体缩放倍数 c。
    # 恒等性说 α 必须精确地按 1/c 缩，所以 α_raw/α_lvl 应当等于 c。
    level_factor = coef["p_ctr"] * coef["p_cvr"] * coef["ltv"]
    alpha_ratio = float(b4_raw.alpha_path.max() / b4_lvl.alpha_path.max())
    alpha_dev = abs(alpha_ratio / level_factor - 1.0)

    banner("预注册的预期：预算固定时，整体缩放价值对按比例出价的买手是不是恒等")
    log.info("价值整体缩放倍数 c = pCTR %.3f × pCVR %.3f × LTV %.3f = %.4f",
             coef["p_ctr"], coef["p_cvr"], coef["ltv"], level_factor)
    log.info("判据① 水位之比 α_raw/α_lvl = %.4f / %.4f = %.4f，"
             "相对 c 只差 %.3f%% —— **solve_alpha 的协变性成立**："
             "v 乘 c，α 就精确缩到 α/c，出价 (c·v)·(α/c) 逐请求不变",
             b4_raw.alpha_path.max(), b4_lvl.alpha_path.max(), alpha_ratio,
             alpha_dev * 100)
    log.info("判据② 水位是否触边界：未校准 %s，校准后 %s（带 [%.2f, %.2f]）",
             "触了" if sat_raw else "没触", "触了" if sat_lvl else "没触",
             cfg.alpha_min, cfg.alpha_max)
    log.info("判据③ 硬闸门命中次数：未校准 %d，校准后 %d。"
             "闸门把一批的**期望**扣费压在剩余预算内，而期望扣费对出价"
             "**超线性** —— 出价虚高 c 倍，闸门压得比 1/c 更狠。"
             "所以它是整条链路里唯一不和缩放交换的部件",
             b4_raw.gate_hits, b4_lvl.gate_hits)
    log.info("判据④ B4 价值 %.1f → %.1f（%+.3f%%）",
             b4_raw.expected_value, b4_lvl.expected_value,
             (b4_lvl.expected_value / b4_raw.expected_value - 1) * 100)
    if gated:
        log.info("→ **闸门触发了，所以恒等性在这里破功**，差值 %.3f%% 就是"
                 "闸门的代价，而且是**校准侧占优**（%.1f > %.1f）—— "
                 "虚高的出价被闸门压得更狠，这正是方向可预注册的那一半。"
                 "如果没有闸门（纯比例出价 + 外生预算），校准前后会完全"
                 "相同：**水平的价值全在闸门和对外可解释性上，不在排序上**",
                 gap * 100, b4_lvl.expected_value, b4_raw.expected_value)
    else:
        log.info("→ **闸门没触发**，所以应当逐请求恒等。实测差 %.3f%%",
                 gap * 100)

    # ---------------- 逐头校准 ≠ 乘积校准 ----------------
    banner("逐头校准 ≠ 乘积校准（这一步最有价值的一条）")
    log.info("四个头的相对误差：%s → %s",
             "，".join(f"{r['头']} {r['相对误差'] * 100:+.1f}%"
                       for r in raw_rows),
             "，".join(f"{r['头']} {r['相对误差'] * 100:+.1f}%" for r in cal_rows))
    log.info("但 **eCPM = pCTR·pCVR·LTV 的乘积聚合偏差**只有 %.4f → %.4f。"
             "三个头各自对上均值，乘积照样偏 —— 因为 E[XYZ] 依赖"
             "**联合分布**，不等于三个边际期望的乘积。想修乘积就得直接"
             "校准乘积本身（拿观测收入拟一个标量），而不是校准它的因子。"
             "而按上面的恒等性，这 %.0f%% 的乘积偏差对出价**毫无影响** —— "
             "水平校准的价值在**对外可解释**（水位数值、风控阈值、"
             "跨渠道比价），不在让出价更准。让出价更准的只有排序",
             raw["ecpm_bias"], lvl["ecpm_bias"], lvl["ecpm_bias"] * 100)

    # ---------------- 主结论 ----------------
    banner("主结论")
    for tag in ("esmm_raw", "esmm_level", "dfm_raw"):
        r = runs[tag]
        b0, b4 = r["res"]["B0_fixed"], r["res"]["B4_lambda_pid"]
        r["lift_vs_fixed"] = b4.expected_value / b0.expected_value - 1.0
        r["ratio_b4_b5"] = (b4.expected_value
                            / r["res"]["B5_oracle"].expected_value)
        r["roi_gain"] = b4.roi / b0.roi - 1.0
        log.info("[%-11s] B4 相对 B0（人工固定出价）：价值 %+.1f%%、ROI %+.1f%%；"
                 "B4/B5（上帝视角上界）= %.4f", tag, r["lift_vs_fixed"] * 100,
                 r["roi_gain"] * 100, r["ratio_b4_b5"])
    best = max(runs, key=lambda k: runs[k]["ratio_b4_b5"])
    worst = min(runs, key=lambda k: runs[k]["ratio_b4_b5"])
    log.info("两个 q 的构造方式里 **%s** 更值钱（B4/B5 = %.4f；%s 是 %.4f）。"
             "两者的**水平**差 2 倍，按上面的恒等性不构成不公平",
             best, runs[best]["ratio_b4_b5"], worst,
             runs[worst]["ratio_b4_b5"])
    log.info("⚠️ 「拍卖只在乎排序」这句话**说对了一半，用错了指标**。赢家 %s "
             "的全量 Spearman 反而**更低**（%.4f vs %s 的 %.4f），B4/B5 却高 "
             "%.4f —— 因为这个指标恰好测的是拍卖**看不见**的东西（推导见 "
             "rank_diag）。按比例出价对 v 的单调变换免疫，所以决定成败的是"
             "预测的**离散程度**而不是名次：弹性 γ %s %.4f vs %s %.4f"
             "（越接近 1 越可信），紧预算截获率 %s %.4f@%.1f%% vs %s "
             "%.4f@%.1f%%。三个指标同向，全量 Spearman 独自反向 —— "
             "谁是错的指标很清楚",
             best, runs[best]["spearman_all"],
             worst, runs[worst]["spearman_all"],
             runs[best]["ratio_b4_b5"] - runs[worst]["ratio_b4_b5"],
             best, runs[best]["loglog_elasticity"],
             worst, runs[worst]["loglog_elasticity"],
             best, runs[best]["capture_at_buy_frac"], runs[best]["buy_frac"] * 100,
             worst, runs[worst]["capture_at_buy_frac"],
             runs[worst]["buy_frac"] * 100)
    log.info("⚠️ 最好也只有 %.2f，**远低于** 06 里纯合成数据的 0.95。差的那"
             "部分是**真实模型误差的代价**，不是控制环的问题：B5 用真值"
             "出价、排序完美，B4 用的是预测。**这是排序问题，不是水平问题**"
             " —— 任何一个常数都修不了它。所以这里如实报出，并把它列为"
             "后续工作（重训模型 / 把乘积的排序修好），而不是把门槛降到"
             "实测值以下来「通过」", runs[best]["ratio_b4_b5"])

    # ---------------- 门槛 ----------------
    problems = []
    for tag, r in runs.items():
        res = r["res"]
        b0, b1 = res["B0_fixed"], res["B1_greedy"]
        b4, b5 = res["B4_lambda_pid"], res["B5_oracle"]
        # 超支红线。**分开卡**：0.5% 是给要上线的那些策略的
        # （B0 人工基线 + B2/B3/B4 智能出价 + B5 上界）。
        # B1 贪心是**故意不带 pacing 的对照**，它的存在就是为了把预算
        # 打光；它的超支来自「闸门按**期望**扣费封顶、而单 slot 的实际
        # 扣费有 4~5% 的尾巴」，出价越虚高尾巴越厚。给它 1.5%，
        # 并且要求它仍然低于 B4（那才是它要证明的东西）。
        for x in res.values():
            lim = 0.015 if x.name == "B1_greedy" else 0.005
            if x.overspend_frac > lim:
                problems.append(f"[{tag}] {x.name} 超支 "
                                f"{x.overspend_frac:.4%} > {lim:.1%}")
        if b4.budget_error >= 0.02:
            problems.append(f"[{tag}] B4 预算误差 {b4.budget_error:.2%} ≥ 2%")
        if b4.pacing_rmse >= 0.03:
            problems.append(f"[{tag}] B4 pacing {b4.pacing_rmse:.4f} ≥ 0.03")
        # 职责点 3 的核心主张：接真模型之后仍然要打赢人工固定出价
        if not b4.expected_value > b0.expected_value * 1.05:
            problems.append(f"[{tag}] B4 {b4.expected_value:.1f} 没超过 B0 "
                            f"{b0.expected_value:.1f} 的 105%")
        if not b4.roi > b0.roi:
            problems.append(f"[{tag}] B4 ROI {b4.roi:.3f} ≤ B0 {b0.roi:.3f}")
        if not b1.expected_value < b4.expected_value:
            problems.append(f"[{tag}] 贪心不低于 B4 —— 时段价值差异没起作用")
        if not b4.expected_value < b5.expected_value:
            problems.append(f"[{tag}] B4 达到甚至超过 B5 —— "
                            f"上界失效，真值可能泄漏进了策略")
        # 外生预算 + 上帝视角：B5 的 α* 是解析反解的，必须落在带内且花得准
        if saturates(b5, cfg):
            problems.append(f"[{tag}] B5 的 α* 触到了水位带边界 "
                            f"(峰值 {b5.alpha_path.max():.4f}) —— 上帝视角"
                            f"先被裁了一刀，上界不干净")
        if b5.budget_error >= 0.02:
            problems.append(f"[{tag}] B5 预算误差 {b5.budget_error:.2%} ≥ 2%")

    # ---- 恒等性的四个判据 ----
    # ① 水位之比必须等于整体缩放倍数 c（solve_alpha 的协变性）
    if not alpha_dev < 0.02:
        problems.append(f"水位之比 α_raw/α_lvl = {alpha_ratio:.4f} 与整体缩放"
                        f"倍数 c = {level_factor:.4f} 差 {alpha_dev:.2%} > 2% —— "
                        f"solve_alpha 的协变性不成立")
    # ② 水位不能触到带边界（触了就饱和，水平校准真的会改变出价）
    if sat_lvl:
        problems.append(f"校准后的 B4 水位贴住 [{cfg.alpha_min}, "
                        f"{cfg.alpha_max}] 边界 —— 控制器饱和，恒等性前提不成立")
    # ③④ 价值的差值与方向：没触发闸门时必须恒等；触发了则必须校准侧不劣
    if gated:
        if not b4_lvl.expected_value >= b4_raw.expected_value:
            problems.append(f"闸门触发时校准侧反而更差 "
                            f"({b4_lvl.expected_value:.1f} < "
                            f"{b4_raw.expected_value:.1f}) —— 与「闸门惩罚"
                            f"虚高出价」的方向相反，机制说不通")
    elif not gap < 5e-3:
        problems.append(f"闸门没触发，B4 价值却差了 {gap:.4%} > 0.5% —— "
                        f"与「预算固定时按价值比例出价对全局缩放恒等」矛盾")
    # 校准的效果：每个头至少要把误差**减半**。
    # 不用「< 5%」这种绝对门槛 —— 系数在验证折上拟合、在测试折上使用，
    # 中间隔着一段跨期漂移，残差本来就不可能为零。
    for r0, r1 in zip(raw_rows, cal_rows):
        if not abs(r1["相对误差"]) < 0.5 * abs(r0["相对误差"]):
            problems.append(f"校准没能把 {r1['头']} 的误差减半"
                            f"（{r0['相对误差']:.1%} → {r1['相对误差']:.1%}）")
    # 乘积偏差：如实报出，但要求校准**方向**是对的
    if not lvl["ecpm_bias"] < raw["ecpm_bias"]:
        problems.append(f"校准把 eCPM 的乘积偏差弄大了"
                        f"（{raw['ecpm_bias']:.4f} → {lvl['ecpm_bias']:.4f}）")
    # B4 相对 B0 的提升是职责点 3 的对外主张，两个变体都得成立
    for tag in ("esmm_raw", "dfm_raw"):
        if not runs[tag]["lift_vs_fixed"] > 0.20:
            problems.append(f"[{tag}] B4 相对 B0 只提升 "
                            f"{runs[tag]['lift_vs_fixed']:.1%} < 20%")
    # 诚实下界：带着真实模型误差，也要拿到上帝视角的四成
    if not runs[best]["ratio_b4_b5"] > 0.40:
        problems.append(f"最好的变体 B4/B5 = {runs[best]['ratio_b4_b5']:.4f} "
                        f"≤ 0.40 —— 预测排序差到出价失去意义了")

    def r_json(x) -> dict:
        return {"spend": x.spend, "overspend_frac": x.overspend_frac,
                "expected_value": x.expected_value, "roi": x.roi,
                "win_rate": x.win_rate, "pacing_rmse": x.pacing_rmse,
                "gate_hits": x.gate_hits,
                "alpha_first": float(x.alpha_path[0]),
                "alpha_max_seen": float(x.alpha_path.max())}

    metrics = {
        "scale": scale,
        "setup": {"n_test": pool["n_pool"], "n_requests": N_REQUESTS,
                  "n_valid": int(split.valid.size), "beta": models["beta"],
                  "prior_logit_shift": float(np.log(models["beta"])),
                  "alpha_band": [cfg.alpha_min, cfg.alpha_max],
                  "budget": budget, "budget_tension": BUDGET_TENSION,
                  "budget_source": "真实价值的无约束花费（外生，三次仿真共用）",
                  "level_correction": coef},
        "heads_uncalibrated": raw_rows,
        "heads_calibrated": cal_rows,
        "prior_correction_ablation": {
            "target": act_cvr,
            "p_cvr_raw_mean_on_clicks": float(cvr[clk].mean()),
            "p_cvr_with_prior_correction": float(cvr_wrong[clk].mean()),
        },
        "identity_check": {
            "level_factor_c": level_factor,
            "alpha_ratio_raw_over_level": alpha_ratio,
            "alpha_ratio_rel_dev": alpha_dev,
            "saturated_raw": sat_raw, "saturated_calibrated": sat_lvl,
            "gate_hits_raw": b4_raw.gate_hits,
            "gate_hits_calibrated": b4_lvl.gate_hits,
            "gate_fired": gated,
            "b4_value_raw": b4_raw.expected_value,
            "b4_value_calibrated": b4_lvl.expected_value,
            "rel_gap": gap,
            "verdict": ("闸门触发：恒等性被硬闸门打破，校准侧占优"
                        if gated else "闸门未触发：逐请求恒等"),
        },
        "product_vs_marginal": {"ecpm_bias_raw": raw["ecpm_bias"],
                                "ecpm_bias_calibrated": lvl["ecpm_bias"]},
        "runs": {tag: {"variant": r["variant"], "calibrated": r["calibrated"],
                       # 排序质量的三个指标。全量 Spearman 是**反例**，
                       # 留着是为了让「它不该被当成排序质量」有据可查。
                       "spearman_pred_true": r["spearman_all"],
                       "loglog_elasticity": r["loglog_elasticity"],
                       "capture_at_buy_frac": r["capture_at_buy_frac"],
                       "buy_frac": r["buy_frac"],
                       "ecpm_aggregate_bias": r["ecpm_bias"],
                       "lift_vs_fixed_bid": r["lift_vs_fixed"],
                       "ratio_b4_to_oracle": r["ratio_b4_b5"],
                       "policies": {k: r_json(v) for k, v in r["res"].items()}}
                 for tag, r in runs.items()},
        "best_variant": best,
        "problems": problems,
    }
    save_metrics("08_bidding_sim", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    # 恒等性分两层，结论必须分开说 —— 混成一句就会在「闸门触发」那一支
    # 里报出一个字面上为真的假结论。判据①（α 的协变性）**永远**成立，
    # 它是 solve_alpha 的代数性质；价值级的恒等则要看闸门有没有插手。
    verdict = ("闸门触发 → 价值级恒等被硬闸门打破，"
               "差值是闸门的代价（方向可预注册）"
               if gated else "闸门未触发 → 价值级恒等成立")
    log.info("体检通过：无超支，最好变体 %s 的 B4 相对 B0 价值 %+.0f%%、"
             "ROI %+.0f%%；B4/B5 = %.4f（真实模型误差的代价已量化）；"
             "整体缩放的恒等性 —— solve_alpha 的协变性成立"
             "（α 之比与 c 差 %.2f%%），价值级：%s",
             best, runs[best]["lift_vs_fixed"] * 100,
             runs[best]["roi_gain"] * 100, runs[best]["ratio_b4_b5"],
             alpha_dev * 100, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
