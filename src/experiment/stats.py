"""A/B 检验的统计工具箱。

**这里最重要的一件事：比率指标不能用普通 t 检验**

CVR = 转化数 / 曝光数，ROI = 收入 / 花费 —— 分子分母**都是随机变量**。
把它们各自算成一个用户级的均值再去跑 t 检验，等于假装分母是常数，
标准误会被系统性低估（因为忽略了分母自身的波动），
于是 p 值偏小、假阳性偏高。这个错误在上线后表现为
「实验老是显著，但业务方说没感觉」。

正确做法是 **delta method（线性化）+ 用户级聚类稳健方差**：

    R = Σ nᵢ / Σ dᵢ                      （nᵢ、dᵢ 是**用户 i** 的分子/分母）
    Var(R) ≈ n · s² / D²      其中  zᵢ = nᵢ − R·dᵢ，s² = Var_c(z)

推导一句话：把 R 在真值处一阶展开，``R − R₀ ≈ (1/D)·Σ(nᵢ − R₀dᵢ)``，
所以 R 的方差就是 ``Σ(nᵢ − R·dᵢ)`` 的方差除以 D²。
这个式子里 ``n`` 是**用户数**而不是曝光数 —— 因为求和单元是用户，
这正是「聚类稳健」的含义：同一用户内部的曝光不独立，
但它们作为一个整体进入方差估计，所以不必假设曝光独立。
（真正写错的地方通常就是这里：用曝光数当 n，标准误会被低估
 ``sqrt(曝光/用户)`` 倍。）

**其他几条纪律**
* 检验统计量的分布假设：``n`` 小或事件数小时用 Fisher 精确检验 / 置换检验，
  而不是硬套正态（见 ``needs_exact_test``）。
* ``p̂`` 接近 0 或 1 时正态区间会越界（给出负的下界），用 **Wilson 区间**。
* 多个指标一起看会放大假阳性，必须做 **BH 校正**（见 ``benjamini_hochberg``）。
* **绝不能中途反复看 p 值再决定何时停**：type-I 错误率会从 5% 涨到 20%+。
  本模块不提供「顺序检验」接口，就是为了不给这条路留门。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------


@dataclass
class TestResult:
    """一次两组比较的结果。``lift`` 是**相对提升**（0.05 表示 +5%）。"""

    metric: str
    control: float
    treatment: float
    diff: float
    se: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    n_control: float
    n_treatment: float
    method: str = "delta"

    @property
    def lift(self) -> float:
        return self.diff / self.control if self.control else float("nan")

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:                      # pragma: no cover
        return (f"{self.metric:10s} 对照 {self.control:.4f} "
                f"实验 {self.treatment:.4f} "
                f"提升 {self.lift * 100:+.2f}% "
                f"[{self.ci_low * 100:+.2f}%, {self.ci_high * 100:+.2f}%] "
                f"p={self.p_value:.2e} ({self.method})")


# --------------------------------------------------------------------------
# 比率指标：delta method + 聚类稳健
# --------------------------------------------------------------------------


def linearized_ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """比率 ``R = Σnum / Σden`` 及其**聚类稳健**方差。

    返回 ``(R, Var(R))``。``num``、``den`` 是**用户级**数组。
    用户级而不是曝光级 —— 见模块 docstring。

    退化保护：``Σden == 0``（这一组没有任何曝光）时返回 ``(0.0, nan)``，
    **不抛异常**。真实实验里「某个变体在某个细分上零曝光」是常态，
    抛异常会让整份报告跑不出来，而正确的处理是把这个格子标成「无法判定」。
    """
    n = np.asarray(num, dtype=np.float64).reshape(-1)
    d = np.asarray(den, dtype=np.float64).reshape(-1)
    if n.size != d.size:
        raise ValueError("分子分母的用户数必须一致")
    n_users = n.size
    total_d = float(d.sum())
    if n_users < 2 or total_d <= 0:
        return 0.0, float("nan")

    r = float(n.sum()) / total_d
    z = n - r * d                       # 每个用户的「影响力」
    s2 = float(np.var(z, ddof=1))       # 无偏样本方差
    return r, n_users * s2 / (total_d ** 2)


def design_effect(num: np.ndarray, den: np.ndarray) -> float:
    """**设计效应**：聚类稳健方差 / 朴素二项方差。

    这个数是连接「统计检验」和「功效分析」的桥，而且它常常被忽略：

    ``power.sample_size_per_arm`` 的公式假设每个观测单元独立。
    但 CVR 的分析单元是**用户**，一个用户贡献几十次点击，
    这些点击既不独立（同一个人有稳定的转化倾向），
    又高度不均匀（重度用户的点击量是长尾用户的两个数量级）。
    于是真实方差是朴素二项方差的 ``deff`` 倍，
    有效样本量只有 ``n / deff``。

    ⚠️ 不乘这个数会**严重高估功效**，而高估的方向很危险：
       算出来「每臂 3 万人够了」，实际跑完发现 p=0.2，
       然后团队得出的结论是「功能无效」而不是「样本不够」。
       本项目的模拟里实测 deff ≈ 6.4，也就是说
       「17 万次点击」只相当于 2.7 万次独立观测 ——
       不做这一步，功效会被高估 6 倍以上
       （理论 95%、实测 32%，见 ``tests/test_07_experiment.py``）。

    ``deff`` 对 DGP 的**数值**很敏感（用户的点击量分布、
    转化倾向的离散度都会改变它），所以它只能实测，不能照抄。
    本模块的价值在于给出**量它的方法**，而不是那个具体的数。
    可复现的敏感度表见 ``src/experiment/runner`` 顶部三个旋钮的注释。

    用法：先在 AA 或历史数据上测出 deff，再拿去做功效分析。
    """
    n = np.asarray(num, dtype=np.float64).reshape(-1)
    d = np.asarray(den, dtype=np.float64).reshape(-1)
    r, var = linearized_ratio(n, d)
    total_d = float(np.sum(d))
    if not np.isfinite(var) or var <= 0 or total_d <= 0:
        return float("nan")
    if not 0 < r < 1:
        return float("nan")
    naive = r * (1 - r) / total_d
    return float(var / naive)


def ratio_delta_test(num_c: np.ndarray, den_c: np.ndarray,
                     num_t: np.ndarray, den_t: np.ndarray,
                     metric: str = "ratio", alpha: float = 0.05
                     ) -> TestResult:
    """两组比率指标的差异检验（delta method + 聚类稳健方差）。

    两组独立，所以方差直接相加 —— 这是配对设计之外的标准做法，
    也是为什么**分流必须在用户级**：同一用户同时进两组的话，
    两组不再独立，这里的方差相加就错了（而且协方差还得估，很难估准）。
    """
    r_c, v_c = linearized_ratio(num_c, den_c)
    r_t, v_t = linearized_ratio(num_t, den_t)
    return _finish(r_c, v_c, r_t, v_t,
                   n_c=float(num_c.size), n_t=float(num_t.size),
                   metric=metric, alpha=alpha, method="delta")


def two_proportion_z(x_c: int, n_c: int, x_t: int, n_t: int, *,
                     metric: str = "conversion", alpha: float = 0.05
                     ) -> TestResult:
    """经典两比例 z 检验（**合并方差**，即 H0 下的方差）。

    这条路径给「每用户恰好一次伯努利」的场景用（如「是否付费」），
    此时分子分母都是计数、没有分母波动，delta method 与它等价。
    有分母波动时（CVR、ROI）用 ``ratio_delta_test``。

    ⚠️ 合并方差 vs 非合并方差：**检验用合并**（H0 下两组同方差，
       合并估计更有效），**置信区间用非合并**（区间要在备择假设下也成立）。
       两者混用不会报错，只是 p 值和区间会轻微不自洽
       —— 表现为「区间不含 0 但 p > 0.05」。
    """
    if min(n_c, n_t) <= 0:
        return _nan_result(metric, alpha)
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_h0 = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)))
    se_h1 = float(np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t))
    if se_h1 <= 0:
        return _nan_result(metric, alpha)
    z = (p_t - p_c) / se_h0 if se_h0 > 0 else 0.0
    p_value = 2 * sps.norm.sf(abs(z))
    half = sps.norm.ppf(1 - alpha / 2) * se_h1
    return TestResult(metric, p_c, p_t, p_t - p_c, se_h1, float(z),
                      float(p_value), (p_t - p_c) - half, (p_t - p_c) + half,
                      float(n_c), float(n_t), method="z")


def _finish(r_c, v_c, r_t, v_t, *, n_c, n_t, metric, alpha, method
            ) -> TestResult:
    diff = r_t - r_c
    var = v_c + v_t
    # 方差不可估（某组零曝光）时 p 值给 NaN 而不是 1.0：
    # 「无法判定」和「判定为无差异」是两件事，混起来会让报告说谎。
    if not np.isfinite(var) or var <= 0:
        return _nan_result(metric, alpha, r_c, r_t, n_c, n_t, method)
    se = float(np.sqrt(var))
    z = diff / se
    p_value = float(2 * sps.norm.sf(abs(z)))
    half = float(sps.norm.ppf(1 - alpha / 2) * se)
    return TestResult(metric, r_c, r_t, diff, se, float(z), p_value,
                      diff - half, diff + half, n_c, n_t, method)


def _nan_result(metric, alpha, r_c=float("nan"), r_t=float("nan"),
                n_c=0.0, n_t=0.0, method="delta") -> TestResult:
    nan = float("nan")
    return TestResult(metric, r_c, r_t, nan, nan, nan, nan, nan, nan,
                      n_c, n_t, method)


# --------------------------------------------------------------------------
# 小样本兜底
# --------------------------------------------------------------------------


def needs_exact_test(x_c: int, n_c: int, x_t: int, n_t: int, *,
                     min_cell: int = 5) -> bool:
    """是否该改用精确检验。判据：任一格的事件数或非事件数 < ``min_cell``。

    正态近似在期望频数小于 5 时明显不准（尾巴太薄，p 值偏小）。
    这个判据是教科书里那条经验规则的显式化 —— 写成函数是为了让
    「什么时候不走正态」这件事**可测试**，而不是埋在 if 里面。
    """
    return min(x_c, n_c - x_c, x_t, n_t - x_t) < min_cell


def fisher_exact_test(x_c: int, n_c: int, x_t: int, n_t: int, *,
                      metric: str = "conversion", alpha: float = 0.05
                      ) -> TestResult:
    """Fisher 精确检验（小样本兜底）。

    点估计和区间沿用 z 检验的（它们不依赖正态近似下的 p 值），
    只把 **p 值**换成精确值 —— 这正是小样本下唯一不可靠的那个量。
    """
    if min(n_c, n_t) <= 0:
        return _nan_result(metric, alpha)
    _, p_value = sps.fisher_exact([[x_t, n_t - x_t], [x_c, n_c - x_c]])
    base = two_proportion_z(x_c, n_c, x_t, n_t, metric=metric, alpha=alpha)
    base.p_value = float(p_value)
    base.method = "fisher"
    return base


def permutation_test(num_c: np.ndarray, den_c: np.ndarray,
                     num_t: np.ndarray, den_t: np.ndarray, *,
                     n_perm: int = 2000, seed: int = 0,
                     metric: str = "ratio", alpha: float = 0.05) -> TestResult:
    """置换检验：把两组标签打乱 ``n_perm`` 次，看真实差异有多极端。

    用在「比率指标 + 样本很小」的场景，那里既不能用正态近似，
    也没有比值型的精确检验可用。代价是 p 值有 ``1/(n_perm+1)`` 的分辨率下限
    （所以 ``p=0`` 要报成 ``< 1/(n_perm+1)``）。
    """
    nc = np.asarray(num_c, dtype=np.float64).reshape(-1)
    dc = np.asarray(den_c, dtype=np.float64).reshape(-1)
    nt = np.asarray(num_t, dtype=np.float64).reshape(-1)
    dt = np.asarray(den_t, dtype=np.float64).reshape(-1)
    n_all = nc.size + nt.size
    if nc.size < 2 or nt.size < 2:
        return _nan_result(metric, alpha)

    num_all = np.concatenate([nc, nt])
    den_all = np.concatenate([dc, dt])
    obs = _ratio(nt, dt) - _ratio(nc, dc)

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_all)
        a, b = idx[:nc.size], idx[nc.size:]
        if abs(_ratio(num_all[b], den_all[b])
               - _ratio(num_all[a], den_all[a])) >= abs(obs):
            hits += 1
    p_value = (hits + 1) / (n_perm + 1)
    r_c, r_t = _ratio(nc, dc), _ratio(nt, dt)
    return TestResult(metric, r_c, r_t, r_t - r_c, float("nan"),
                      float("nan"), float(p_value), float("nan"),
                      float("nan"), float(nc.size), float(nt.size),
                      method="permutation")


def _ratio(num, den) -> float:
    d = float(np.sum(den))
    return float(np.sum(num)) / d if d > 0 else 0.0


# --------------------------------------------------------------------------
# SRM：分流比例失衡检测
# --------------------------------------------------------------------------


@dataclass
class SRMResult:
    chi2: float
    p_value: float
    counts: np.ndarray
    expected: np.ndarray

    @property
    def ok(self) -> bool:
        return self.p_value > 0.001

    def __str__(self) -> str:                      # pragma: no cover
        return (f"SRM 卡方 {self.chi2:.3f} p={self.p_value:.4f} "
                f"实际 {self.counts.tolist()} 期望 "
                f"{np.round(self.expected, 1).tolist()}")


def srm_test(counts, weights=None) -> SRMResult:
    """样本比例失衡（Sample Ratio Mismatch）检验。

    **SRM 是最灵敏的 bug 探测器，优先级高于一切指标。**
    分流比例一旦和设计不符，说明分流链路本身有问题（哈希有偏、
    缓存串组、某组请求被丢弃、日志丢事件……），此时**任何指标差异
    都不可信** —— 因为能被分流偏差影响的东西，同样能影响指标。

    所以实验报告的第一行应该是 SRM，不是 p 值。
    判别阈值用 0.001 而不是 0.05：这是**上线前的守门检查**，
    会被反复执行（每个实验、每个细分、每天），用 0.05 意味着
    20 个正常实验里就有 1 个被误判为「分流坏了」而白跑一遍排查。
    """
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    if c.size < 2:
        raise ValueError("至少两组才能做 SRM 检验")
    if c.sum() <= 0:
        raise ValueError("总样本为 0，无法做 SRM 检验")
    w = (np.full(c.size, 1.0 / c.size) if weights is None
         else np.asarray(weights, dtype=np.float64).reshape(-1))
    if w.size != c.size:
        raise ValueError("权重个数与组数不符")
    if w.sum() <= 0:
        raise ValueError("权重之和必须为正")
    w = w / w.sum()
    expected = w * c.sum()
    chi2 = float(np.sum((c - expected) ** 2 / expected))
    p_value = float(sps.chi2.sf(chi2, df=c.size - 1))
    return SRMResult(chi2, p_value, c, expected)


# --------------------------------------------------------------------------
# 区间估计
# --------------------------------------------------------------------------


def wilson_interval(x: int, n: int, alpha: float = 0.05
                    ) -> tuple[float, float]:
    """比例的 Wilson 区间。``p̂`` 接近 0/1 或 ``n`` 小时用它，不用正态区间。

    正态区间（``p̂ ± z·sqrt(p̂(1−p̂)/n)``）在 ``p̂ = 0`` 时退化成一个点，
    还会给出负的下界 —— 报「转化率的 95% 区间是 [−0.002, 0.004]」
    在业务方那里直接失去可信度。Wilson 区间用「加 4 个虚拟观测」
    的思路修正，天然落在 ``[0, 1]`` 内，小样本下覆盖率也更接近名义值。
    """
    if n <= 0:
        return float("nan"), float("nan")
    z = float(sps.norm.ppf(1 - alpha / 2))
    p = x / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# --------------------------------------------------------------------------
# 多重比较
# --------------------------------------------------------------------------


def benjamini_hochberg(p_values, q: float = 0.10) -> np.ndarray:
    """BH 校正，返回「是否拒绝」的布尔数组（FDR 控制在 ``q`` 以内）。

    为什么必须做：一次实验会看 CTR、CVR、RPM、ROI、花费……五六个指标。
    每个都用 0.05 判定的话，「至少误报一个」的概率是
    ``1 − 0.95⁶ ≈ 26%``。业务方看到「三个指标显著提升」时，
    其中可能有一两个纯属运气。

    为什么用 BH 而不是 Bonferroni：后者控制的是「**一个都不误报**」的概率
    （FWER），在指标数量多、且各指标高度相关（CTR 和 CVR 就很相关）时
    过于保守，会漏掉真实效应。BH 控制的是「**误报占所有报出的比例**」
    （FDR），这正是业务决策真正在意的量。

    ⚠️ 输入里的 NaN（无法判定的指标）按「不拒绝」处理，且**不参与排序**
        —— 把它当成 1.0 混进去会让校正阈值整体变松。
    """
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    reject = np.zeros(p.size, dtype=bool)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return reject
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(p[idx])]
    thresholds = q * (np.arange(1, m + 1) / m)
    passed = np.flatnonzero(p[order] <= thresholds)
    if passed.size:
        k = int(passed[-1]) + 1
        reject[order[:k]] = True
    return reject


# --------------------------------------------------------------------------
# 相对提升的置信区间（业务方真正要看的数）
# --------------------------------------------------------------------------


def relative_lift_ci(res: TestResult, alpha: float = 0.05
                     ) -> tuple[float, float]:
    """把绝对差异的区间换算成**相对提升**区间。

    ⚠️ 是「区间的端点各自除以对照均值」，不是「点估计的区间再乘」——
       两者只在对称区间下等价。这里直接换算端点是精确的。
       对照均值为 0（或负）时返回 NaN：此时相对提升没有定义，
       报一个「+∞%」出来只会让人误读。
    """
    if not np.isfinite(res.control) or res.control <= 0:
        return float("nan"), float("nan")
    return res.ci_low / res.control, res.ci_high / res.control
