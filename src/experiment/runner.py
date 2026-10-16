"""实验编排：造数据 → 分流 → 检验 → 报告。

**AA 干跑（dry run）是这个模块存在的第一理由**

AB 实验的第一步**不是**跑 AB，是跑 AA：两组跑**完全相同**的策略。
此时 H0 为真，所以理论上不该出现显著差异。

但这里有个必须讲清楚的认知：**AA 的单次 p 值必然服从 U(0,1)**，
也就是说它本来就有 5% 的概率「显著」。所以

* **不能**断言「跑一次 AA，p > 0.05」—— 那是个 5% 概率失败的断言，
  写进 CI 就是每 20 次红一次，最后被人加 `flaky` 标记，
  反而把真正的分流 bug 一起掩盖掉；
* 正确的断言是**分布**：跑 ``aa_replications`` 次独立 AA（每次换
  ``exp_id``，重新分流），断言 p 值服从 U(0,1)（KS 检验 p > 0.05），
  且**假阳性率落在 [0.01, 0.12]**（理论值 0.05）。

KS 检验本身也是个统计检验，所以它自己也有 5% 的失败率；
本项目的取法是「KS 的 p > 0.05」+「FPR 落在区间内」**同时**成立，
两个条件一起把「系统性偏差」和「单次抖动」分开 —— 前者会让 KS 的 p
接近 0，后者最多让其中一个条件偶尔擦边。

**AA 到底能查出什么**：分流链路的**内生性偏差**。常见来源有
哈希有偏（某类 ID 集中进一组）、缓存串组、某组的日志丢事件、
两组跑在同一台机器上互相抢资源。这些都会让 AA 出现系统性的
非零差异，而且**它们在 AB 里是看不出来的** —— AB 里混淆在真实的
处理效应里，看起来就像「效果不错」。

**业务周期**：一天的实验会被昼夜和星期效应绑架（周末的 CVR 天然不同），
所以报告按天给出逐日提升，并单独检查工作日/周末两个分层，
用「两个分层都显著且同号」来证明效果不是某一个时段撑起来的。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.experiment import stats as ST
from src.experiment.cuped import cuped_adjust, leakage_warning
from src.experiment.hash_split import split_users

# --------------------------------------------------------------------------
# 数据生成的三个旋钮（改动前先读注释，它们各自决定一个下游结论）
# --------------------------------------------------------------------------

# 用户转化倾向的对数正态 σ。它直接决定**设计效应 deff**：
# 调大 → 用户之间差别更大 → deff 更大 → 同样点击量能测出的提升更小。
# 实测（``_probe17``，2 万用户 × 6 次）：σ=0.3 → deff 3.6，
# σ=0.45 → 4.7，σ=0.6 → 6.6。
CVR_PROPENSITY_SIGMA = 0.6

# CVR 对用户价值的敏感度（``p_cvr ∝ (0.8 + slope·value)·u``）。
#
# ⚠️ 这个斜率是 DGP 里**最容易把 deff 搞爆**的旋钮，而且它比
#    ``CVR_PROPENSITY_SIGMA`` 猛得多。取 1/30（「高价值用户转化率是
#    普通用户的 4 倍」）时，重点击用户同时是高转化用户，两个长尾相乘，
#    实测 deff ≈ 56；即使把 σ 压到 0.3，deff 仍有 44。
#    真实广告实验里 CVR 的设计效应是个位数，所以这里取 1/600 ——
#    「高价值用户转化略高」这层含义保住了，方差不再由极少数人主导，
#    实测 deff ≈ 6.4（范围 4.0~8.3）。
VALUE_TO_CVR_SLOPE = 1.0 / 600.0

# 实验前的人均转化次数（乘以用户倾向）。它决定 CUPED 协变量的强度：
# 太小 → 大多数人是 0 → ρ 太低 → CUPED 触发不了 min_rho 门槛。
# 0.8 对应约一半用户实验前有转化，实测 ρ≈0.55（方差下降约 31%）。
PRE_CONV_RATE = 0.8


# --------------------------------------------------------------------------
# 模拟的用户级实验数据
# --------------------------------------------------------------------------


@dataclass
class ExperimentData:
    """用户级实验数据。

    漏斗是 ``曝光 → 点击 → 转化``，和项目其余部分的口径一致：
    **主指标 CVR 的分母是点击**（对应 ``Config.target_cvr = 0.06``），
    CTR 的分母才是曝光。

    **协变量 ``pre_revenue`` 取自实验之前** —— 这是 CUPED 唯一不能让步的前提。

    逐日数组 ``daily_*`` 保留下来是为了「跨越完整业务周期」的验证：
    只跑一天的话，周末和工作日的差异会被当成处理效应。
    """

    user_id: np.ndarray
    impressions: np.ndarray        # 实验期曝光
    clicks: np.ndarray
    conversions: np.ndarray
    revenue: np.ndarray            # 实验期收入，元
    spend: np.ndarray              # 实验期花费，元
    pre_revenue: np.ndarray        # **实验前**收入（CUPED 协变量）
    pre_conversions: np.ndarray
    daily_impressions: np.ndarray  # (n_users, n_days)
    daily_clicks: np.ndarray
    daily_conversions: np.ndarray
    day_is_weekend: np.ndarray     # (n_days,) bool

    def __len__(self) -> int:
        return int(self.user_id.size)

    @property
    def n_days(self) -> int:
        return int(self.daily_conversions.shape[1])

    def stratum_mask(self, weekend: bool) -> np.ndarray:
        """按「工作日 / 周末」把逐日数组汇成用户级聚合。

        返回 ``(impressions, clicks, conversions)`` 三个用户级数组,
        给分层检验用。
        """
        cols = self.day_is_weekend if weekend else ~self.day_is_weekend
        return (self.daily_impressions[:, cols].sum(axis=1),
                self.daily_clicks[:, cols].sum(axis=1),
                self.daily_conversions[:, cols].sum(axis=1))


def make_experiment_data(cfg, *, n_users: int, seed: int, days: int = 28,
                         cvr_lift: float = 0.0, revenue_lift: float = 0.0,
                         variant: np.ndarray | None = None,
                         lift_fraction: float = 1.0) -> ExperimentData:
    """造一份用户级实验数据。

    ``variant`` 给出每个用户属于哪一组 —— **必须在造数据之前就分流**，
    因为处理效应只施加在变体 > 0 的用户上。
    传 ``None`` 加上非零 ``lift`` 会直接报错：否则效应被静默丢弃，
    实验看起来「没有提升」，而这是最容易被误读成「功能没用」的一种失败。

    ``lift_fraction`` 模拟**处理只覆盖一部分流量**（新模型只对 30%
    的请求生效），真实 ATE 因此是 ``lift × lift_fraction`` ——
    用它来验证检验估出来的效应量对不对。

    **两个基率锚在配置上**：聚合 CTR ≈ ``cfg.target_ctr``、
    聚合 CVR（= 转化/点击）≈ ``cfg.target_cvr``。
    这件事比看上去重要——功效分析的基线、README 里的对照、
    以及「这条链路的口径是否自洽」全都建立在它上面。

    **用户价值是双峰的**（有付费习惯 / 没有），这一点必须保留：
    它决定了收入指标的方差结构，而 CUPED 的收益恰好来自
    「同一个人在实验前后都是高价用户」这个相关性。
    换成一条光滑分布，CUPED 的方差下降会假得离谱。
    """
    if n_users <= 0:
        raise ValueError("用户数必须为正")
    if days <= 0:
        raise ValueError("实验天数必须为正")
    if (cvr_lift or revenue_lift) and variant is None:
        raise ValueError(
            "要施加处理效应就必须先分流并把 variant 传进来，"
            "否则效应会被静默丢弃（表现为「实验无提升」）")
    rng = np.random.default_rng([seed, 7717])
    uid = np.arange(n_users, dtype=np.int64)

    # --- 用户级的潜在属性（实验前后都成立）---
    heavy = rng.random(n_users) < 0.10
    base_value = np.where(heavy,
                          np.exp(rng.normal(np.log(60.0), 0.9, n_users)),
                          np.exp(rng.normal(np.log(5.0), 1.0, n_users)))

    treated = (np.zeros(n_users, dtype=bool) if variant is None
               else np.asarray(variant).reshape(-1) > 0)
    if treated.size != n_users:
        raise ValueError("variant 长度与用户数不符")
    effective = treated & (rng.random(n_users) < lift_fraction)

    # --- 稳定的用户倾向 ---
    #
    # ``u_cvr`` 是用户**自身的转化倾向**（均值为 1 的对数正态）。
    # 它承担两件事：
    #   1. 提供 CVR 的**用户间异质性** —— 这正是设计效应 deff 的来源，
    #      也是比率指标不能退化成二项检验的原因；
    #   2. 同时驱动**实验前**的转化，于是 ``pre_revenue`` 天然成为有效的
    #      CUPED 协变量 ——「过去爱转化的人将来也爱转化」本来就是
    #      一个稳定的用户属性，CUPED 的全部收益都建立在它上面。
    #
    # ⚠️ 它**刻意不与点击量挂钩**。把 CVR 直接写成 ``base_value`` 的增函数
    #    （``0.6 + base_value/30``）时，重点是「高价值用户既点得多又转化得高」，
    #    两个长尾相乘，实测 deff 冲到 45 —— 那不是真实实验的样子
    #    （真实 CVR 的 deff 通常是个位数），而是把全部方差交给了
    #    极少数极端用户，功效分析会因此得出「需要 50 万用户」这种
    #    误导性的结论。异质性要留，耦合要去掉。
    u_cvr = rng.lognormal(-0.5 * CVR_PROPENSITY_SIGMA ** 2,
                          CVR_PROPENSITY_SIGMA, n_users)

    # 单次付费金额：围绕用户自身的价值水平
    amount = base_value * rng.lognormal(0.0, 0.6, n_users)

    # 实验前行为（CUPED 协变量）—— **必须**取自实验之前
    pre_conversions = rng.poisson(PRE_CONV_RATE * u_cvr)
    pre_revenue = pre_conversions * amount

    # 每天一次抽样：曝光 → 点击 → 转化，三层都带用户异质性
    lam = 6.0 + 24.0 * (base_value / (base_value + 30.0))

    # 业务周期：周末流量更大、点击意愿更高，但**购买意图更低**。
    # 这个「周末 CVR 掉一档」正是「只跑一天会得出错误结论」的根源。
    weekend = np.array([(d % 7) >= 5 for d in range(days)])
    imp_day_scale = np.where(weekend, 1.25, 1.0)
    ctr_day_scale = np.where(weekend, 1.10, 1.0)
    cvr_day_scale = np.where(weekend, 0.85, 1.0)

    # ⚠️ 两个基率都必须**按各自的分母加权**归一化，否则口径对不上：
    #    * CTR 的分母是**曝光** → 按期望曝光量 ``lam`` 加权；
    #    * CVR 的分母是**点击** → 按期望点击量 ``lam·p_ctr`` 加权。
    #    这个加权不能省。不归一化时，高价用户同时贡献更多点击和更高 CVR，
    #    点击加权的聚合 CVR 会被抬到目标基线的 3 倍（实测 0.183 vs 0.06）——
    #    于是「基线 6%」这个口径在实验模块里对不上，
    #    功效分析据此算出的样本量也就跟着错了一个数量级。
    p_ctr_shape = 0.6 + base_value / 40.0
    p_ctr = np.clip(
        cfg.target_ctr * p_ctr_shape
        / (np.average(p_ctr_shape, weights=lam)
           * float(ctr_day_scale.mean())), 1e-4, 0.6)
    w_click = lam * p_ctr
    # 价值对 CVR 的影响保留但**压平**（见 VALUE_TO_CVR_SLOPE 的注释）：
    # 「高价值用户转化略高」是真的，但它不该主导方差。
    p_cvr_shape = (0.8 + VALUE_TO_CVR_SLOPE * base_value) * u_cvr
    p_cvr = np.clip(
        cfg.target_cvr * p_cvr_shape
        / (np.average(p_cvr_shape, weights=w_click)
           * float(cvr_day_scale.mean())), 1e-4, 0.95)
    # 处理效应乘在**转化率**上：两组的用户构成相同，
    # 所以聚合 CVR 的相对提升就等于 cvr_lift（期望意义下）
    p_cvr_eff = np.where(effective, p_cvr * (1.0 + cvr_lift), p_cvr)

    daily_imp = np.zeros((n_users, days), dtype=np.int64)
    daily_clk = np.zeros((n_users, days), dtype=np.int64)
    daily_cnv = np.zeros((n_users, days), dtype=np.int64)
    for d in range(days):
        n_imp = rng.poisson(np.maximum(lam * imp_day_scale[d], 0.0))
        daily_imp[:, d] = n_imp
        n_clk = rng.binomial(n_imp, np.clip(p_ctr * ctr_day_scale[d], 0, 1))
        daily_clk[:, d] = n_clk
        daily_cnv[:, d] = rng.binomial(
            n_clk, np.clip(p_cvr_eff * cvr_day_scale[d], 0, 1))

    impressions = daily_imp.sum(axis=1)
    clicks = daily_clk.sum(axis=1)
    conversions = daily_cnv.sum(axis=1)

    amount = np.where(effective, amount * (1.0 + revenue_lift), amount)
    revenue = conversions * amount

    # 花费：按曝光计价（CPM 约 25 元 → 每次曝光 0.025 元）
    spend = impressions * 0.025 * rng.lognormal(0.0, 0.25, n_users)

    return ExperimentData(
        user_id=uid, impressions=impressions, clicks=clicks,
        conversions=conversions, revenue=revenue, spend=spend,
        pre_revenue=pre_revenue, pre_conversions=pre_conversions,
        daily_impressions=daily_imp, daily_clicks=daily_clk,
        daily_conversions=daily_cnv, day_is_weekend=weekend,
    )


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


@dataclass
class ExperimentReport:
    """一次 AB 实验的完整报告。

    **顺序是有意义的**：先看 SRM，再看主指标，最后才是那些好看的
    次要指标。分流坏掉的实验，后面所有数字都不值得看。
    """

    exp_id: str
    n_control: int
    n_treatment: int
    srm: ST.SRMResult
    primary: ST.TestResult
    secondary: dict[str, ST.TestResult] = field(default_factory=dict)
    guardrail: dict[str, ST.TestResult] = field(default_factory=dict)
    cuped: dict[str, float] = field(default_factory=dict)
    daily_lift: np.ndarray = field(default_factory=lambda: np.zeros(0))
    strata: dict[str, ST.TestResult] = field(default_factory=dict)
    bh_rejected: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """分流是否健康。**这是解读一切的前提。**"""
        return self.srm.p_value > 0.001

    def to_dict(self) -> dict:
        """给 JSON / README 用的扁平结构（只放能对外讲的数）。"""
        d = {
            "exp_id": self.exp_id,
            "srm_p": float(self.srm.p_value),
            "srm_chi2": float(self.srm.chi2),
            "n_control": int(self.n_control),
            "n_treatment": int(self.n_treatment),
            "healthy": bool(self.healthy),
            "primary": {
                "metric": self.primary.metric,
                "control": float(self.primary.control),
                "treatment": float(self.primary.treatment),
                "lift": float(self.primary.lift),
                "p_value": float(self.primary.p_value),
                "ci": [float(self.primary.ci_low), float(self.primary.ci_high)],
                "method": self.primary.method,
            },
            "secondary": {k: {"lift": float(v.lift),
                              "p_value": float(v.p_value),
                              "bh_rejected": bool(self.bh_rejected.get(k, False))}
                          for k, v in self.secondary.items()},
            "guardrail": {k: {"lift": float(v.lift),
                              "p_value": float(v.p_value)}
                          for k, v in self.guardrail.items()},
            "cuped": {k: float(v) for k, v in self.cuped.items()},
            "strata": {k: {"lift": float(v.lift), "p_value": float(v.p_value)}
                       for k, v in self.strata.items()},
            "warnings": list(self.warnings),
        }
        return d

    def summary(self) -> str:
        lines = [
            f"实验 {self.exp_id}",
            f"  {self.srm}",
            f"  分流健康：{'是' if self.healthy else '否 —— 以下所有数字都不可信'}",
            f"  主指标  {self.primary}",
        ]
        for k, v in self.secondary.items():
            mark = "*" if self.bh_rejected.get(k) else " "
            lines.append(f"  {mark}次要  {v}")
        for k, v in self.guardrail.items():
            lines.append(f"    护栏  {v}")
        if self.cuped:
            lines.append(f"  CUPED  方差下降 {self.cuped.get('variance_reduction', 0)*100:.1f}%"
                         f"（ρ={self.cuped.get('rho', float('nan')):.3f}）")
        for k, v in self.strata.items():
            lines.append(f"    分层  {k}: {v}")
        for w in self.warnings:
            lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# AB
# --------------------------------------------------------------------------


def run_ab(data: ExperimentData, split, *, exp_id: str = "ab",
           primary_metric: str = "cvr", alpha: float = 0.05,
           bh_q: float = 0.10, min_rho: float = 0.30) -> ExperimentReport:
    """跑一次 AB：SRM → 主指标 → CUPED → 次要指标 + BH → 分层。

    ``split`` 是 ``hash_split.SplitResult``（或任何有 ``variant`` 的对象）。
    用户顺序必须与 ``data`` 一致 —— 这里显式检查长度，
    因为它对不上时不会报错，只会把结果算成一个随机噪声。
    """
    variant = np.asarray(split.variant).reshape(-1)
    if variant.size != len(data):
        raise ValueError(f"分流结果 {variant.size} 与实验数据 {len(data)} 长度不符")
    m_c, m_t = variant == 0, variant > 0
    if m_c.sum() == 0 or m_t.sum() == 0:
        raise ValueError("两组都必须非空")

    srm = ST.srm_test([int(m_c.sum()), int(m_t.sum())])

    # ---- 主指标：CVR = 转化 / 点击（比率指标 → delta method）----
    #
    # ⚠️ 分母是**点击**不是曝光，和项目其余部分的口径一致
    #    （``Config.target_cvr = 0.06`` 就是 CVR|click）。
    #    分子分母都随用户波动，所以必须走 delta method 而不是 t 检验。
    primary = ST.ratio_delta_test(data.conversions[m_c], data.clicks[m_c],
                                  data.conversions[m_t], data.clicks[m_t],
                                  metric=primary_metric, alpha=alpha)
    report = ExperimentReport(exp_id=exp_id, n_control=int(m_c.sum()),
                              n_treatment=int(m_t.sum()), srm=srm,
                              primary=primary)
    if not report.healthy:
        report.warnings.append(
            "SRM 未通过：分流比例与设计不符，指标差异可能完全来自分流偏差")

    # ---- CUPED：对**分子**做调整，再走同一条比率检验 ----
    #
    # ⚠️ 直接对「每用户 CVR」做 CUPED 是错的：CVR 的分母（点击数）
    #    本身也是随机的，调整后的量不再是一个比率，
    #    delta method 的线性化前提就不成立了。
    #    正确做法是只调整分子（转化数），分母原样保留。
    leak = leakage_warning(data.pre_revenue, variant)
    if leak:
        report.warnings.append(leak)
    conv_adj = np.empty(len(data))
    ratios = []
    for m in (m_c, m_t):
        r = cuped_adjust(data.conversions[m], data.pre_revenue[m],
                         min_rho=min_rho)
        conv_adj[m] = r.y_adj
        ratios.append(r)
    adj_primary = ST.ratio_delta_test(conv_adj[m_c], data.clicks[m_c],
                                      conv_adj[m_t], data.clicks[m_t],
                                      metric=primary_metric, alpha=alpha)
    report.cuped = {
        "rho": float(np.mean([r.rho for r in ratios])),
        "variance_reduction": float(np.mean(
            [r.variance_reduction for r in ratios])),
        "applied": bool(all(r.applied for r in ratios)),
        "adjusted_p_value": float(adj_primary.p_value),
        "adjusted_lift": float(adj_primary.lift),
    }

    # ---- 次要指标 ----
    report.secondary = {
        "ctr": ST.ratio_delta_test(data.clicks[m_c], data.impressions[m_c],
                                   data.clicks[m_t], data.impressions[m_t],
                                   metric="ctr", alpha=alpha),
        "rpm": ST.ratio_delta_test(data.revenue[m_c], data.impressions[m_c],
                                   data.revenue[m_t], data.impressions[m_t],
                                   metric="rpm", alpha=alpha),
        "roi": ST.ratio_delta_test(data.revenue[m_c], data.spend[m_c],
                                   data.revenue[m_t], data.spend[m_t],
                                   metric="roi", alpha=alpha),
        "cpa": ST.ratio_delta_test(data.spend[m_c], data.conversions[m_c],
                                   data.spend[m_t], data.conversions[m_t],
                                   metric="cpa", alpha=alpha),
    }
    rejected = ST.benjamini_hochberg(
        [v.p_value for v in report.secondary.values()], q=bh_q)
    report.bh_rejected = {k: bool(r) for k, r in
                          zip(report.secondary.keys(), rejected)}

    # ---- 护栏指标：变差才是问题 ----
    #
    # 「超支率」不在这里 —— 它是**出价链路**的护栏（见 tests/test_06），
    # 不属于用户侧的实验。实验的护栏应该盯「用户有没有被伤害」：
    # 零转化用户比例（推了更多广告却一次没转化）和每次转化成本。
    report.guardrail = {
        "zero_conversion_rate": ST.ratio_delta_test(
            (data.conversions[m_c] == 0).astype(float),
            np.ones(int(m_c.sum())),
            (data.conversions[m_t] == 0).astype(float),
            np.ones(int(m_t.sum())), metric="zero_conv", alpha=alpha),
        "cost_per_conversion": report.secondary["cpa"],
    }

    # ---- 业务周期：逐日提升 + 工作日/周末分层 ----
    report.daily_lift = _daily_lift(data, m_c, m_t)
    report.strata = _strata_tests(data, m_c, m_t, alpha)
    return report


def _daily_lift(data: ExperimentData, m_c, m_t) -> np.ndarray:
    """逐日的 CVR 相对提升。

    **实验必须覆盖一个完整周期，看的就是这个序列**：如果提升只在
    某几天出现（比如只在大促那两天），那它不是一个可上线的功能，
    而是一次活动。序列平稳才是「效果可持续」的证据。
    """
    days = data.n_days
    out = np.full(days, float("nan"))
    for d in range(days):
        den_c = float(data.daily_clicks[m_c, d].sum())
        den_t = float(data.daily_clicks[m_t, d].sum())
        if den_c <= 0 or den_t <= 0:
            continue
        c = float(data.daily_conversions[m_c, d].sum()) / den_c
        t = float(data.daily_conversions[m_t, d].sum()) / den_t
        if c > 0:
            out[d] = t / c - 1.0
    return out


def _strata_tests(data: ExperimentData, m_c, m_t,
                  alpha: float) -> dict[str, ST.TestResult]:
    """工作日 / 周末分层。

    **两个分层必须同号且都显著**，否则「显著」只是某一个时段撑起来的，
    换个周期就没了。这是「跨越完整业务周期验证」的落地方式 ——
    比单纯把实验跑久一点更有针对性：跑久只能压低标准误，
    分层才能看出效果**在结构上**是否稳定。
    """
    out: dict[str, ST.TestResult] = {}
    for name, weekend in (("工作日", False), ("周末", True)):
        imp, clk, cnv = data.stratum_mask(weekend)
        # 该分层没有点击的用户对分母没有贡献，剔除后不影响比率，
        # 但能让聚类数（自由度）落在真实样本上
        keep = clk > 0
        c = m_c & keep
        t = m_t & keep
        if c.sum() < 30 or t.sum() < 30:
            continue
        out[name] = ST.ratio_delta_test(cnv[c], clk[c], cnv[t], clk[t],
                                        metric=name, alpha=alpha)
    return out


def simulate_experiment(cfg, *, n_users: int, seed: int, exp_id: str,
                        days: int = 28, cvr_lift: float = 0.0,
                        revenue_lift: float = 0.0, salt: str = "",
                        lift_fraction: float = 1.0,
                        bh_q: float | None = None,
                        n_buckets: int | None = None):
    """一步到位：**分流 → 造数据 → 跑检验**，返回 ``(data, split, report)``。

    ⚠️ 顺序不能反。先造数据再分流的话，处理效应就施加不到任何用户身上
       （生成器不知道谁是实验组），结果是「实验无提升」——
       而这正是最容易被误读成「功能没用」的那种失败。
       这也是 ``make_experiment_data`` 在 ``variant is None`` 且
       ``lift != 0`` 时直接报错的原因。
    """
    uid = np.arange(int(n_users), dtype=np.int64)
    split = split_users(uid, exp_id, salt=salt,
                        n_buckets=cfg.n_buckets if n_buckets is None
                        else n_buckets)
    data = make_experiment_data(cfg, n_users=n_users, seed=seed, days=days,
                                cvr_lift=cvr_lift, revenue_lift=revenue_lift,
                                variant=split.variant,
                                lift_fraction=lift_fraction)
    report = run_ab(data, split, exp_id=exp_id,
                    bh_q=cfg.bh_q if bh_q is None else bh_q)
    return data, split, report


# --------------------------------------------------------------------------
# AA 干跑
# --------------------------------------------------------------------------


@dataclass
class AAResult:
    """AA 干跑的结果。**断言的是分布，不是单次 p 值。**"""

    p_values: np.ndarray
    srm_p_values: np.ndarray
    n_per_group: int

    @property
    def false_positive_rate(self) -> float:
        return float(np.mean(self.p_values < 0.05))

    @property
    def ks_p_value(self) -> float:
        """p 值 vs U(0,1) 的 KS 检验。**H0 为真时它自己也应当不显著。**"""
        from scipy import stats as sps
        return float(sps.kstest(self.p_values, "uniform").pvalue)

    def summary(self) -> str:
        return (f"AA 干跑 {self.p_values.size} 次：假阳性率 "
                f"{self.false_positive_rate * 100:.1f}%（理论 5%），"
                f"p 值对 U(0,1) 的 KS p={self.ks_p_value:.3f}，"
                f"SRM 最小 p={self.srm_p_values.min():.3f}")


def run_aa(cfg, *, n_users: int, replications: int | None = None,
           seed: int = 0, exp_id_prefix: str = "aa") -> AAResult:
    """AA 干跑：两组跑**完全相同**的策略，重复 ``replications`` 次。

    ⚠️ 每一次都必须**重新生成数据 + 重新分流**：
       拿同一份数据只换分流种子，200 个 p 值之间高度相关，
       KS 统计量的零分布会失真（KS 假设 p 值独立）。
       这一点很容易被忽略，因为它跑起来更快、看起来也没问题。
    """
    n_rep = int(cfg.aa_replications if replications is None else replications)
    if n_rep <= 0:
        raise ValueError("AA 重复次数必须为正")
    p_values = np.empty(n_rep)
    srm_ps = np.empty(n_rep)
    for k in range(n_rep):
        # 每次换 exp_id：既保证分流不同，又模拟「每个实验一个独立前缀」
        exp_id = f"{exp_id_prefix}_{k}"
        data = make_experiment_data(cfg, n_users=n_users, seed=seed * 100_003 + k)
        split = split_users(data.user_id, exp_id, n_buckets=cfg.n_buckets)
        rep = run_ab(data, split, exp_id=exp_id)
        p_values[k] = rep.primary.p_value
        srm_ps[k] = rep.srm.p_value
    return AAResult(p_values=p_values, srm_p_values=srm_ps,
                    n_per_group=n_users // 2)
