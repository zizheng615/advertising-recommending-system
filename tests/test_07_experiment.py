"""职责点 4：分流 / 统计检验 / 功效分析 / CUPED / AA + AB 全链路。

测试策略
--------
这一块的「正确答案」大多是**分布性质**而不是单个数字，所以断言方式
和前面几个模块完全不同：

* **能闭式算的**（θ、方差下降比、BH 阈值、Wilson 端点、去重保护）
  → 精确断言。这些是确定性计算，断言松了等于没测。
* **有已知真值的**（模拟出来的 CVR 基率、注入的 ``cvr_lift``、
  iid 情形下的 ``deff = 1``）→ 门槛断言，容差由样本量反推。
* **统计检验的 p 值** → **只断言分布，绝不断言单次结果**。
  这是本文件最重要的一条纪律：H0 为真时单次 p 值必然服从 U(0,1)，
  它本来就有 5% 的概率「显著」。写 ``assert p > 0.05`` 就是写一个
  每 20 次红一次的测试，最后的结局是被人加上 ``flaky`` 标记 ——
  反而把真正的分流 bug 一起掩盖掉。
  正确写法：跑 N 次独立 AA，断言 KS 均匀性 + 假阳性率落在合理区间。

⚠️ **本文件里所有涉及 p 值的随机断言都固定了种子。**
   这不是为了「让它过」，而是因为「分布断言」本身也有失败概率
   （KS 检验自己就有 5% 的 I 类错误）。固定种子把「一次确定的运行」
   变成可复现的基线；换种子重跑仍然应该通过，但那是**回归测试**，
   不是每次 CI 都必须无懈可击的断言。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from config import get_config
from src.experiment import power as PW
from src.experiment import stats as ST
from src.experiment.cuped import (cuped_adjust, leakage_warning,
                                  standardized_mean_diff)
from src.experiment.hash_split import (DEFAULT_BUCKETS, assign_buckets,
                                       split_users, user_bucket, variant_of)
from src.experiment.runner import (CVR_PROPENSITY_SIGMA, PRE_CONV_RATE,
                                   VALUE_TO_CVR_SLOPE, ExperimentData,
                                   make_experiment_data, run_aa, run_ab,
                                   simulate_experiment)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return get_config("mini")


# ==========================================================================
# 1. 分流：跨进程确定性、正交性、均匀性
# ==========================================================================


def test_hash_is_stable_across_processes_and_hash_seeds():
    """分流结果必须与 ``PYTHONHASHSEED`` 无关。

    这是分流模块存在的**唯一技术理由**。用内置 ``hash()`` 的话，
    同一批用户重启一次服务就会被重新分组，实验静默失效 ——
    而且单进程调试时完全看不出来。

    用真的子进程测，不用 mock：``PYTHONHASHSEED`` 是进程启动时
    才生效的，同一进程内改 ``os.environ`` 没有任何作用，
    mock 掉的话这个测试就永远通过、也永远抓不到问题。
    """
    code = (
        "import sys;"
        f"sys.path.insert(0, r'{PROJECT_ROOT}');"
        "from src.experiment.hash_split import user_bucket;"
        "print([user_bucket(u, 'exp_a', 'salt') for u in (0, 1, 7, 12345)])"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True,
                           env=_subprocess_env(seed),
                           cwd=str(PROJECT_ROOT), timeout=120)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2], (
        f"分流结果随 PYTHONHASHSEED 变化：{outs} —— "
        f"说明用了带随机盐的 hash()")


def _subprocess_env(hash_seed: str) -> dict:
    """子进程环境：保留系统变量，只覆盖 ``PYTHONHASHSEED``。

    ⚠️ 不能用 ``{"PYTHONHASHSEED": ...}`` 这样的干净环境：
       Windows 上 Python 启动时要靠 ``SYSTEMROOT`` 等变量初始化
       随机数源，环境一空就直接 fatal error 退出。
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    return env


def test_builtin_hash_really_is_unstable_across_processes():
    """反面对照：内置 ``hash()`` 确实会变 —— 证明上一条测试有意义。

    没有这个对照，上一条测试可能只是「凑巧三个种子下都一样」。
    两个 64 位哈希相等的概率是 2⁻⁶⁴，可以放心断言不等。
    """
    code = "print(hash('exp_a|salt|7'))"
    outs = []
    for seed in ("0", "1"):
        r = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True,
                           env=_subprocess_env(seed), timeout=120)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] != outs[1], (
        "内置 hash() 在两个 PYTHONHASHSEED 下相同 —— "
        "这个对照失去了意义，请检查子进程环境是否真的传进去了")


def test_separator_prevents_collisions():
    """``exp_id|salt|user_id`` 里的分隔符不能省。

    省掉它之后 ``("1", "23")`` 和 ``("12", "3")`` 会拼成同一个串，
    两个实验之间就产生了系统性关联 —— 而那正是「正交分层」要避免的。
    """
    assert user_bucket(23, "1", "") != user_bucket(3, "12", "")
    # 直接验证拼接串确实不同
    assert f"{'1'}|{'23'}" != f"{'12'}|{'3'}"


def test_different_exp_ids_are_orthogonal():
    """不同实验的分组必须**互相独立**（正交分层的前提）。

    用 2×2 列联表的卡方检验。正交时它应该只有随机波动。
    这条断言比「两个实验的 variant 数组不相等」强得多：
    后者在两组高度相关（比如完全反相关）时也成立。
    """
    uid = np.arange(20_000)
    a = split_users(uid, "exp_a").variant
    b = split_users(uid, "exp_b").variant
    table = np.array([[np.sum((a == i) & (b == j)) for j in (0, 1)]
                      for i in (0, 1)])
    _, p = sps.chi2_contingency(table)[:2]
    assert p > 0.001, f"两个实验的分组不独立（卡方 p={p:.2e}）：{table}"
    # 同时确认它们不是同一套分组（否则「独立」是假的）
    assert np.mean(a == b) < 0.55


def test_salt_also_orthogonalises():
    """同 ``exp_id`` 下不同 ``salt`` 也必须独立。"""
    uid = np.arange(20_000)
    a = split_users(uid, "exp_a", salt="v1").variant
    b = split_users(uid, "exp_a", salt="v2").variant
    table = np.array([[np.sum((a == i) & (b == j)) for j in (0, 1)]
                      for i in (0, 1)])
    _, p = sps.chi2_contingency(table)[:2]
    assert p > 0.001


def test_split_is_even():
    """分流比例必须接近 50/50。偏了就说明分桶有问题。"""
    uid = np.arange(100_000)
    counts = split_users(uid, "exp_even").counts()
    assert counts.sum() == 100_000
    assert np.all(np.abs(counts / counts.sum() - 0.5) < 0.01), counts


def test_variant_of_boundaries_are_exact():
    """桶区间切分要**精确**：每个变体拿到的桶数相差不超过 1。

    用取模（``bucket % k``）的话，``n_buckets`` 不是 ``k`` 的整数倍时
    前面的变体会多拿几个桶 —— 比例偏差只有 1/10000，看着微不足道，
    但它是**系统性**偏差，而 SRM 检验对这个量级极其敏感。
    """
    n = 10_000
    b = np.arange(n)
    v2 = variant_of(b, 2, n_buckets=n)
    assert np.array_equal(v2[:5000], np.zeros(5000, dtype=np.int8))
    assert np.array_equal(v2[5000:], np.ones(5000, dtype=np.int8))

    v4 = variant_of(b, 4, n_buckets=n)
    assert [int((v4 == k).sum()) for k in range(4)] == [2500] * 4

    # 桶数不是变体数整数倍：差值最多 1
    v3 = variant_of(np.arange(10_000), 3, n_buckets=10_000)
    counts = [int((v3 == k).sum()) for k in range(3)]
    assert max(counts) - min(counts) <= 1, counts


def test_variant_of_rejects_out_of_range_buckets():
    with pytest.raises(ValueError):
        variant_of(np.array([0, 10_000]), 2, n_buckets=10_000)
    with pytest.raises(ValueError):
        variant_of(np.array([-1]), 2, n_buckets=10_000)
    with pytest.raises(ValueError):
        variant_of(np.array([0, 1]), 1, n_buckets=10_000)


def test_variant_of_n_buckets_cannot_be_inferred_from_data():
    """⚠️ 桶数必须显式传，不能从 ``bucket.max()`` 推断。

    小规模分流（比如测试时只用 100 个桶）之后拿默认的 10000 去切分，
    所有桶号都落在第一个变体里 —— **全员进对照组**。
    这个错误不会报错，只会让「实验组」空掉。
    """
    small = assign_buckets(np.arange(500), "exp", n_buckets=100)
    wrong = variant_of(small, 2, n_buckets=DEFAULT_BUCKETS)
    right = variant_of(small, 2, n_buckets=100)
    assert wrong.max() == 0, "本例正是要展示「推断桶数」会全分到一组"
    assert right.min() == 0 and right.max() == 1


def test_split_users_rejects_duplicate_ids():
    """重复的 user_id 必须报错。

    不报错的话，同一个用户在两组各出现一次，两组不再独立，
    ``ratio_delta_test`` 里「方差直接相加」的前提就不成立了 ——
    但结果看起来完全正常。
    """
    with pytest.raises(ValueError):
        split_users(np.array([1, 2, 2, 3]), "exp")


def test_same_user_always_lands_in_the_same_bucket():
    uid = np.arange(1000)
    a = assign_buckets(uid, "exp_x")
    b = assign_buckets(uid, "exp_x")
    assert np.array_equal(a, b)


# ==========================================================================
# 2. 比率指标的 delta method
# ==========================================================================


def test_linearized_ratio_matches_the_plain_ratio():
    rng = np.random.default_rng(0)
    d = rng.poisson(20, 2000).astype(float)
    n = rng.binomial(d.astype(int), 0.06).astype(float)
    r, var = ST.linearized_ratio(n, d)
    assert r == pytest.approx(n.sum() / d.sum())
    assert var > 0


def test_linearized_ratio_degenerate_guards():
    """零分母、单用户返回 ``(0, nan)`` —— **不抛异常**。

    真实实验里「某变体在某细分上零曝光」是常态，抛异常会让整份报告
    跑不出来，而正确做法是把这个格子标成「无法判定」。
    ``nan`` 而不是 0：方差无法估计和方差为零是两件事。
    """
    r, var = ST.linearized_ratio(np.array([0.0]), np.array([0.0]))
    assert r == 0.0 and np.isnan(var)
    r, var = ST.linearized_ratio(np.zeros(5), np.zeros(5))
    assert r == 0.0 and np.isnan(var)
    with pytest.raises(ValueError):
        ST.linearized_ratio(np.zeros(3), np.zeros(4))


def test_delta_method_se_agrees_with_user_level_bootstrap():
    """delta method 的标准误要和**用户级 bootstrap** 对得上。

    这是本模块最核心的等式。对不上就说明线性化写错了
    （比如把用户数写成了曝光数，SE 会低估 sqrt(曝光/用户) 倍）。
    bootstrap 慢但假设少，用它当裁判。
    """
    rng = np.random.default_rng(11)
    n_users = 4000
    d = rng.poisson(15, n_users).astype(float)
    p = rng.beta(2, 30, n_users)                      # 用户间 CVR 有异质性
    n = rng.binomial(d.astype(int), p).astype(float)
    half = n_users // 2
    res = ST.ratio_delta_test(n[:half], d[:half], n[half:], d[half:])

    # 用户级 bootstrap：重抽用户，而不是重抽曝光
    r = rng.integers(0, n_users, size=(400, n_users))
    diffs = np.empty(400)
    for k in range(400):
        idx = r[k]
        c, t = idx < half, idx >= half
        diffs[k] = (n[t].sum() / d[t].sum()) - (n[c].sum() / d[c].sum())
    boot_se = float(np.std(diffs, ddof=1))
    assert res.se == pytest.approx(boot_se, rel=0.15), (
        f"delta method SE={res.se:.6f}，bootstrap SE={boot_se:.6f}")


def test_delta_method_beats_naive_binomial_by_the_design_effect():
    """聚类稳健方差必须**大于**朴素二项方差，倍数就是 ``deff``。

    ``deff`` 有两个来源，这里**故意都放进去**（真实数据里它们同时存在）：
    点击量的长尾（``E[n²]/n̄`` 大）和用户间 CVR 的差异（``Var(p)`` 大）。
    只放 Poisson 点击量 + 温和的 Beta 时 ``deff`` 只有 1.6 ——
    量级对但看不出问题的严重性。
    """
    rng = np.random.default_rng(3)
    n_users = 5000
    d = np.maximum(rng.lognormal(np.log(18.0) - 0.72, 1.2, n_users), 0.0)
    p = rng.beta(0.5, 8.0, n_users)                 # 用户间 CVR 差异很大
    n = rng.binomial(d.round().astype(int), p).astype(float)
    r, var = ST.linearized_ratio(n, d)
    naive = r * (1 - r) / d.sum()
    assert var > naive * 3, "有用户异质性时聚类方差必须明显大于朴素二项方差"
    assert ST.design_effect(n, d) == pytest.approx(var / naive, rel=1e-9)


def test_design_effect_is_one_for_iid_equal_clusters():
    """每个用户点击量相同、转化率也相同时，``deff`` 应当 ≈ 1。

    这条是 deff 的**零点校准**：如果连这种情况都测出 deff=5，
    那公式里的 ``n`` 一定写错了（用曝光数代替用户数就会出现这种结果）。
    """
    rng = np.random.default_rng(5)
    n_users, per_user = 4000, 20
    d = np.full(n_users, float(per_user))
    n = rng.binomial(per_user, 0.06, n_users).astype(float)
    deff = ST.design_effect(n, d)
    assert deff == pytest.approx(1.0, abs=0.15), f"deff={deff}"


def test_design_effect_grows_with_user_heterogeneity():
    """``deff`` 必须随用户间差异**单调增** —— 这是它唯一的含义。

    ⚠️ 用 ``Beta(p̄·κ, (1−p̄)·κ)`` 参数化：``κ`` 是**集中度**，
       κ 越大分布越集中、用户间差异越小。直接调 Beta 的两个形状参数
       会踩到 U 形分布（两参数都 < 1 时方差反而变大），
       于是「差异越大 deff 越大」这条会时真时假、测试变成随机通过。
    """
    rng = np.random.default_rng(6)
    n_users = 6000
    d = rng.poisson(18, n_users).astype(float)
    p_bar = 0.06
    out = []
    for kappa in (10_000.0, 500.0, 50.0, 5.0):
        p = rng.beta(p_bar * kappa, (1 - p_bar) * kappa, n_users)
        n = rng.binomial(d.astype(int), np.clip(p, 1e-6, 0.99)).astype(float)
        out.append(ST.design_effect(n, d))
    assert out[0] < out[1] < out[2] < out[3], out
    assert out[0] == pytest.approx(1.0, abs=0.10)   # 同质时退化成二项
    assert out[3] > 3.0                             # 异质时方差涨 3 倍以上


def test_design_effect_returns_nan_when_ratio_is_degenerate():
    assert np.isnan(ST.design_effect(np.ones(10), np.ones(10)))     # R = 1
    assert np.isnan(ST.design_effect(np.zeros(10), np.ones(10)))    # R = 0
    assert np.isnan(ST.design_effect(np.zeros(5), np.zeros(5)))


# ==========================================================================
# 3. 两比例检验 / SRM / 区间 / 多重比较
# ==========================================================================


def test_two_proportion_z_uses_pooled_variance_for_the_test():
    """检验用**合并**方差，区间用**非合并**方差，两者不能混。

    混用不会报错，只会让 p 值和区间轻微不自洽 ——
    表现为「区间不含 0，但 p > 0.05」。手工重算两个 SE 来钉住这个契约。
    """
    x_c, n_c, x_t, n_t = 120, 2000, 150, 2000
    res = ST.two_proportion_z(x_c, n_c, x_t, n_t)
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_h0 = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    se_h1 = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    assert res.z == pytest.approx((p_t - p_c) / se_h0)
    assert res.se == pytest.approx(se_h1)
    assert res.p_value == pytest.approx(2 * sps.norm.sf(abs(res.z)))
    # 区间用的是 se_h1
    assert (res.ci_high - res.ci_low) / 2 == pytest.approx(
        sps.norm.ppf(0.975) * se_h1)


def test_two_proportion_z_degenerate_returns_nan_not_error():
    res = ST.two_proportion_z(0, 0, 5, 10)
    assert np.isnan(res.p_value) and np.isnan(res.se)


def test_ratio_test_p_value_matches_a_hand_built_z():
    rng = np.random.default_rng(21)
    d = rng.poisson(20, 3000).astype(float)
    n = rng.binomial(d.astype(int), 0.06).astype(float)
    half = 1500
    res = ST.ratio_delta_test(n[:half], d[:half], n[half:], d[half:])
    _, v_c = ST.linearized_ratio(n[:half], d[:half])
    _, v_t = ST.linearized_ratio(n[half:], d[half:])
    assert res.se == pytest.approx(np.sqrt(v_c + v_t))
    assert res.p_value == pytest.approx(2 * sps.norm.sf(abs(res.diff / res.se)))


def test_srm_passes_on_a_balanced_split():
    res = ST.srm_test([5010, 4990])
    assert res.p_value > 0.001 and res.ok


def test_srm_catches_an_imbalance():
    """SRM 是最灵敏的 bug 探测器，必须真的能响。"""
    assert not ST.srm_test([5000, 5500]).ok
    assert ST.srm_test([5000, 5500]).p_value < 1e-5
    # 3% 的失衡就该报警了（0.001 阈值下对应约 2.5%）
    assert not ST.srm_test([50_000, 52_000]).ok


def test_srm_respects_non_uniform_weights():
    """设计就是不均等分流（如 90/10）时，不能拿 50/50 去卡。"""
    assert ST.srm_test([9000, 1000], weights=[0.9, 0.1]).ok
    assert not ST.srm_test([9500, 500], weights=[0.9, 0.1]).ok


def test_srm_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ST.srm_test([100])
    with pytest.raises(ValueError):
        ST.srm_test([0, 0])
    with pytest.raises(ValueError):
        ST.srm_test([1, 2], weights=[1.0])


def test_wilson_interval_stays_inside_zero_one():
    """正态区间在 ``p̂=0`` 时退化成 [0,0] 并给出负下界，Wilson 不会。"""
    lo, hi = ST.wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12)   # 截断到 0，不是 6.9e-18
    assert 0 < hi < 0.10                         # 「50 次零转化」上界约 7%
    lo, hi = ST.wilson_interval(50, 50)
    assert hi == pytest.approx(1.0, abs=1e-12) and 0.90 < lo < 1.0
    assert ST.wilson_interval(1, 3)[0] >= 0.0
    assert np.isnan(ST.wilson_interval(0, 0)[0])


def test_wilson_interval_covers_the_true_rate():
    """覆盖率检验：95% 区间应当覆盖真值的比例接近 0.95。"""
    rng = np.random.default_rng(8)
    p_true, n = 0.02, 200
    x = rng.binomial(n, p_true, 800)
    cover = [ST.wilson_interval(int(k), n) for k in x]
    rate = np.mean([lo <= p_true <= hi for lo, hi in cover])
    assert 0.90 < rate < 0.99, f"Wilson 覆盖率 {rate:.3f}"


def test_wilson_is_wider_than_normal_at_small_n():
    """小样本 + 极端 p̂ 时 Wilson 必须比正态区间**宽**（因为它才够保守）。"""
    n, x = 20, 0
    lo_w, hi_w = ST.wilson_interval(x, n)
    z = sps.norm.ppf(0.975)
    p = x / n
    half = z * np.sqrt(p * (1 - p) / n)
    assert (hi_w - lo_w) > 2 * half


def test_benjamini_hochberg_step_up_is_correct():
    """BH 的判据是「**最大的**满足 p₍ₖ₎ ≤ q·k/m 的 k」。"""
    p = [0.001, 0.02, 0.5, 0.9]
    r = ST.benjamini_hochberg(p, q=0.10)
    assert r.tolist() == [True, True, False, False]


def test_benjamini_hochberg_ignores_nan_without_loosening():
    """NaN 必须**不参与排序**。当成 1.0 混进去会把阈值整体放松。"""
    r = ST.benjamini_hochberg([0.001, float("nan"), 0.5], q=0.10)
    assert r.tolist() == [True, False, False]
    # 全 NaN 时一个都不拒绝
    assert ST.benjamini_hochberg([float("nan")] * 3).tolist() == [False] * 3


def test_benjamini_hochberg_controls_fdr_under_the_null():
    """全零假设下，BH 的误报比例应当接近 ``q`` 而不是 1。"""
    rng = np.random.default_rng(4)
    ps = rng.random((300, 20))                 # 300 次实验、每次 20 个指标
    rej = np.array([ST.benjamini_hochberg(row, q=0.10) for row in ps])
    fdr = rej.sum() / max(1, rej.any(axis=1).sum() * 20)
    assert rej.sum() / rej.size < 0.10, f"FDR 失控：{rej.mean():.3f}"


def test_needs_exact_test_rule():
    assert ST.needs_exact_test(2, 100, 8, 100)
    assert not ST.needs_exact_test(50, 1000, 60, 1000)


def test_fisher_p_value_differs_but_point_estimate_agrees_with_z():
    """Fisher 只换 **p 值**，点估计/区间沿用 z 的。

    小样本下唯一不可靠的就是 p 值，没理由把点估计也一起换掉。
    """
    res = ST.fisher_exact_test(2, 40, 9, 40)
    z = ST.two_proportion_z(2, 40, 9, 40)
    assert res.control == z.control and res.treatment == z.treatment
    assert res.ci_low == z.ci_low and res.ci_high == z.ci_high
    assert res.method == "fisher" and res.p_value != z.p_value
    assert 0 < res.p_value < 0.05


def test_permutation_test_agrees_with_delta_on_clear_signal():
    """置换检验和 delta method 是两套完全不同的推理，结论要一致。"""
    rng = np.random.default_rng(9)
    d_c = rng.poisson(20, 300).astype(float)
    d_t = rng.poisson(20, 300).astype(float)
    n_c = rng.binomial(d_c.astype(int), 0.05).astype(float)
    n_t = rng.binomial(d_t.astype(int), 0.20).astype(float)
    delta = ST.ratio_delta_test(n_c, d_c, n_t, d_t)
    perm = ST.permutation_test(n_c, d_c, n_t, d_t, n_perm=500, seed=1)
    assert delta.p_value < 0.01 and perm.p_value < 0.01
    assert perm.diff == pytest.approx(delta.diff)


def test_permutation_test_has_a_resolution_floor():
    """置换检验的 p 值不可能小于 ``1/(n_perm+1)`` —— 要显式体现出来。"""
    rng = np.random.default_rng(10)
    d = rng.poisson(20, 100).astype(float)
    n_c = rng.binomial(d.astype(int), 0.02).astype(float)
    n_t = rng.binomial(d.astype(int), 0.60).astype(float)
    res = ST.permutation_test(n_c, d, n_t, d, n_perm=199, seed=2)
    assert res.p_value >= 1 / 200


def test_relative_lift_ci_is_endpoint_division():
    """相对提升区间 = 绝对区间端点 ÷ 对照均值，**不是**点估计再乘。"""
    rng = np.random.default_rng(12)
    d = rng.poisson(20, 2000).astype(float)
    n = rng.binomial(d.astype(int), 0.06).astype(float)
    res = ST.ratio_delta_test(n[:1000], d[:1000], n[1000:], d[1000:])
    lo, hi = ST.relative_lift_ci(res)
    assert lo == pytest.approx(res.ci_low / res.control)
    assert hi == pytest.approx(res.ci_high / res.control)
    assert lo < res.lift < hi


def test_relative_lift_ci_is_nan_for_zero_control():
    res = ST.ratio_delta_test(np.zeros(10), np.ones(10),
                              np.ones(10), np.ones(10))
    assert np.isnan(ST.relative_lift_ci(res)[0])


# ==========================================================================
# 4. 功效分析
# ==========================================================================


def test_sample_size_round_trips_through_achieved_power():
    """``sample_size_per_arm`` 与 ``achieved_power`` 必须互为反函数。

    这条等式一旦不成立，说明两个公式里有一项写错了 ——
    而样本量算错的后果是「实验做完了才发现测不出来」。
    """
    for baseline, mde_rel in ((0.06, 0.05), (0.02, 0.20), (0.15, 0.03)):
        n = PW.sample_size_per_arm(baseline, mde_rel, alpha=0.05, power=0.80)
        got = PW.achieved_power(baseline, n, mde_rel, alpha=0.05)
        assert got == pytest.approx(0.80, abs=0.005), (
            f"基线 {baseline}、MDE {mde_rel}：需要 {n:.0f}，"
            f"但那个样本量下的功效是 {got:.4f}")


def test_design_effect_scales_the_sample_size_linearly():
    """``deff`` 的作用就是「有效样本量除以 deff」，必须严格线性。"""
    n1 = PW.sample_size_per_arm(0.06, 0.05, deff=1.0)
    n4 = PW.sample_size_per_arm(0.06, 0.05, deff=4.0)
    assert n4 / n1 == pytest.approx(4.0, rel=1e-6)
    # 而且 deff 修正后的答案必须更大 —— 这是它唯一的方向
    assert n4 > n1


def test_design_effect_multiplies_power_down_not_up():
    n = 200_000
    p1 = PW.achieved_power(0.06, n, 0.05, deff=1.0)
    p3 = PW.achieved_power(0.06, n, 0.05, deff=6.0)
    assert p1 > p3
    # 同样的单元数，等效独立单元只有 1/deff
    assert PW.achieved_power(0.06, n, 0.05, deff=6.0) == pytest.approx(
        PW.achieved_power(0.06, n / 6, 0.05, deff=1.0))


def test_the_naive_formula_is_wildly_optimistic():
    """这是本模块要讲的核心事实，用两个角度钉住它。

    角度一：同样的样本量，两种算法给出的功效差一个数量级。
    角度二（更有业务含义）：要真的达到 80% 功效，
    deff 修正后需要的样本量是朴素公式的 ``deff`` 倍。
    """
    n = 173_000                       # 约等于 1 万用户 × 17 次点击
    naive = PW.achieved_power(0.06, n, 0.05, deff=1.0)
    real = PW.achieved_power(0.06, n, 0.05, deff=6.4)
    assert naive > 0.90               # 「理论说肯定测得出来」
    assert real < 0.35                # 「实际只有三成把握」
    assert naive / real > 3
    assert naive - real > 0.55

    n_naive = PW.sample_size_per_arm(0.06, 0.05)
    n_real = PW.sample_size_per_arm(0.06, 0.05, deff=6.4)
    assert n_real / n_naive == pytest.approx(6.4, rel=1e-6)
    assert n_naive < 110_000          # 朴素答案：「10 万次点击，看着够了」
    assert n_real > 600_000           # 真实答案：差 6 倍


def test_mde_is_the_inverse_of_achieved_power():
    n = 300_000
    m = PW.mde(0.06, n, alpha=0.05, power=0.80)
    assert PW.achieved_power(0.06, n, m, alpha=0.05) == pytest.approx(
        0.80, abs=0.005)
    # deff 越大，同样样本量能测出的提升越大
    assert PW.mde(0.06, n, deff=6.0) > PW.mde(0.06, n, deff=1.0)


def test_mde_returns_the_cap_when_even_a_huge_lift_is_undetectable():
    """连上限都测不出来时必须返回上限，而不是 NaN 或 inf 之外的东西。

    返回 NaN 会被下游静默吃掉；返回上限至少能让调用方看见
    「样本量小到连 50% 的提升都测不出来」这个结论。
    """
    assert PW.mde(0.06, 100, hi=0.5) == 0.5
    assert PW.mde(0.06, 0) == float("inf")


def test_sample_size_degenerate_inputs_return_inf_not_raise():
    """「测不出来」是合法结论；抛异常会逼调用方写 try，而写 try 的人
    往往会顺手 return 一个假数字。"""
    assert PW.sample_size_per_arm(0.06, 0.0) == float("inf")
    assert PW.sample_size_per_arm(0.06, 0.05, deff=0.0) == float("inf")
    assert PW.sample_size_per_arm(0.06, 50.0) == float("inf")   # p₂ > 1
    with pytest.raises(ValueError):
        PW.sample_size_per_arm(0.0, 0.05)
    with pytest.raises(ValueError):
        PW.sample_size_per_arm(1.0, 0.05)


def test_plan_experiment_units_are_not_users():
    """``plan_experiment`` 的口径是**观测单元**（点击），不是用户。

    这个区分是本模块最容易出错的地方：把用户数当单元数直接传进去，
    会同时漏掉「人均点击数」和「deff」两个因子，
    样本量能差两个数量级，而且结果看起来完全合理。
    """
    plan = PW.plan_experiment(0.06, 0.05, available_units=173_000,
                              units_per_user=17.0, deff=6.4,
                              alpha=0.05, power=0.80)
    assert plan["units_per_arm"] == PW.sample_size_per_arm(
        0.06, 0.05, deff=6.4)
    assert plan["total_units"] == pytest.approx(2 * plan["units_per_arm"])
    assert plan["users_per_arm"] == pytest.approx(
        plan["units_per_arm"] / 17.0)
    assert plan["available_users"] == pytest.approx(173_000 / 17.0)
    assert plan["mde_at_available"] > plan["mde_relative"]
    assert plan["feasible"] is False
    # 不传 units_per_user 时就不该有人数口径的字段
    bare = PW.plan_experiment(0.06, 0.05)
    assert "users_per_arm" not in bare and "total_users" not in bare
    assert bare["deff"] == 1.0


def test_power_requires_sane_alpha_and_power():
    with pytest.raises(ValueError):
        PW.sample_size_per_arm(0.06, 0.05, alpha=0.0)
    with pytest.raises(ValueError):
        PW.sample_size_per_arm(0.06, 0.05, power=1.0)


# ==========================================================================
# 5. CUPED
# ==========================================================================


def test_cuped_theta_and_variance_reduction_match_the_closed_form():
    """``θ = Cov/Var``、``方差比 = 1 − ρ²`` —— 两个都是闭式解，精确断言。"""
    rng = np.random.default_rng(31)
    n = 20_000
    x = rng.normal(10, 3, n)
    y = 2.5 * x + rng.normal(0, 4, n)
    res = cuped_adjust(y, x, min_rho=0.30)
    theta_true = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    rho_true = np.corrcoef(y, x)[0, 1]
    assert res.applied
    assert res.theta == pytest.approx(theta_true)
    assert res.rho == pytest.approx(rho_true)
    assert res.variance_ratio == pytest.approx(1 - rho_true ** 2, abs=1e-9)
    assert res.variance_reduction == pytest.approx(rho_true ** 2, abs=1e-9)
    assert res.effective_sample_multiplier == pytest.approx(
        1 / (1 - rho_true ** 2))


def test_cuped_preserves_the_mean_exactly():
    """CUPED 改变方差、**不改变点估计**的前提是均值不变。

    这条不成立的话，「调整后」的提升就和「调整前」不是同一个量，
    整个方法失去意义。
    """
    rng = np.random.default_rng(32)
    x = rng.lognormal(0, 1, 5000)
    y = rng.poisson(3, 5000) + x
    res = cuped_adjust(y, x)
    assert res.y_adj.mean() == pytest.approx(y.mean(), rel=1e-12)


def test_cuped_min_rho_guard_skips_weak_covariates():
    """弱协变量不调 —— θ 自身的噪声会让方差**变大**而收益接近 0。"""
    rng = np.random.default_rng(33)
    x = rng.normal(0, 1, 20_000)
    y = rng.normal(0, 1, 20_000)         # 与 x 独立
    res = cuped_adjust(y, x, min_rho=0.30)
    assert not res.applied
    assert res.variance_ratio == 1.0 and res.theta == 0.0
    assert np.array_equal(res.y_adj, y)


def test_cuped_guards_against_constant_and_tiny_inputs():
    """常数协变量往往意味着**取错了列**，这里不能静默地 θ=0 就算完。"""
    x = np.full(100, 3.0)
    y = np.arange(100.0)
    res = cuped_adjust(y, x)
    assert not res.applied and res.theta == 0.0
    assert np.isnan(res.rho)
    short = cuped_adjust(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert not short.applied
    with pytest.raises(ValueError):
        cuped_adjust(np.zeros(5), np.zeros(4))


def test_cuped_on_a_shuffled_covariate_does_nothing_good():
    """把协变量打乱 → ρ≈0 → 必须拒绝调整（而不是硬调出一个假收益）。"""
    rng = np.random.default_rng(34)
    x = rng.lognormal(0, 1, 10_000)
    y = rng.poisson(2, 10_000) + x
    shuffled = rng.permutation(x)
    res = cuped_adjust(y, shuffled, min_rho=0.10)
    assert not res.applied


def test_standardized_mean_diff_is_scale_free():
    """SMD 的意义就在于**不受量纲影响** —— 同一份数据放大 100 倍结果不变。"""
    rng = np.random.default_rng(35)
    v = rng.integers(0, 2, 5000)
    x = rng.normal(0, 1, 5000) + 0.02 * v
    assert standardized_mean_diff(x, v) == pytest.approx(
        standardized_mean_diff(x * 100 + 7, v))
    assert np.isnan(standardized_mean_diff(x, np.zeros(5000)))


def test_leakage_warning_fires_on_a_post_treatment_covariate():
    """协变量带上了处理效应（说明它取自实验**之后**）时必须报警。

    这是 CUPED 唯一不可让步的前提。用实验后的量做协变量会把处理效应
    本身调整掉 —— 实验明明是正的，调整完变成 0，而且不报任何错。
    """
    rng = np.random.default_rng(36)
    v = rng.integers(0, 2, 5000)
    post = np.where(v == 1, 5.0, 0.0) + rng.normal(0, 1, 5000)   # 被处理污染
    pre = rng.normal(0, 1, 5000)                                  # 干净的
    assert leakage_warning(post, v) is not None
    assert leakage_warning(pre, v) is None
    assert leakage_warning(np.full(5000, 2.0), v) is not None     # 常数


# ==========================================================================
# 6. 实验数据生成器：口径必须对得上
# ==========================================================================


def test_base_rates_are_anchored_to_the_config(cfg):
    """聚合 CTR / CVR 必须锚在 ``cfg.target_ctr`` / ``cfg.target_cvr`` 上。

    ⚠️ 这条比看上去重要。CVR 的分母是**点击**，所以「聚合 CVR」是
       点击加权的：高价用户同时贡献更多点击和更高 CVR 时，
       不按点击加权归一化就会把聚合值抬到基线的 3 倍
       （实测 0.183 vs 目标 0.06），而功效分析里的基线口径
       是由它决定的 —— 基线错 3 倍，样本量错得更多。
    """
    ctrs, cvrs = [], []
    for k in range(4):
        d = make_experiment_data(cfg, n_users=20_000, seed=k)
        ctrs.append(d.clicks.sum() / d.impressions.sum())
        cvrs.append(d.conversions.sum() / d.clicks.sum())
    assert np.mean(ctrs) == pytest.approx(cfg.target_ctr, rel=0.05)
    assert np.mean(cvrs) == pytest.approx(cfg.target_cvr, rel=0.05)
    # 单次生成之间的抖动要小（否则上面两条只是碰巧对上）
    assert np.std(cvrs) < 0.002


def test_the_dgp_knobs_are_in_the_documented_range():
    """DGP 的三个旋钮各自对应一个下游结论，值本身也要被测住。

    ``VALUE_TO_CVR_SLOPE`` 是其中最容易把 deff 搞爆的一个：
    取 1/30 时实测 deff≈56，比真实广告实验高一个数量级，
    功效分析会因此得出「需要 50 万用户」这种误导性的结论。
    """
    assert 0 < VALUE_TO_CVR_SLOPE <= 1 / 300
    assert 0.2 <= CVR_PROPENSITY_SIGMA <= 0.8
    assert 0.2 <= PRE_CONV_RATE <= 3.0


def test_the_simulated_design_effect_is_in_a_realistic_band(cfg):
    """实测 deff 要落在个位数到十几 —— 真实广告实验的量级。

    既不能是 1（那说明用户异质性根本没生成出来，功效分析会盲目乐观），
    也不该是几十（那说明方差被极少数极端用户主导了）。
    """
    deffs = [ST.design_effect(*(lambda d: (d.conversions, d.clicks))(
        make_experiment_data(cfg, n_users=20_000, seed=k)))
        for k in range(4)]
    assert 2.0 < np.mean(deffs) < 15.0, f"deff={deffs}"
    # 且必须显著大于 1 —— 这就是「比率指标不能用二项公式」的量化证据
    assert np.min(deffs) > 1.5


def test_user_level_click_distribution_is_overdispersed(cfg):
    """点击量必须是**过度离散**的（长尾），这是 deff > 1 的主要来源。

    换成泊松分布，deff 会掉到 2 以下，「功效被高估」这件事
    就演示不出来了。
    """
    d = make_experiment_data(cfg, n_users=20_000, seed=0)
    n = d.clicks.astype(float)
    assert n.mean() == pytest.approx(n.var(), rel=0.6) or n.var() > 5 * n.mean()
    assert np.percentile(n, 99) > 8 * np.median(n)


def test_lift_must_be_paired_with_a_split():
    """非零 ``lift`` 但不传 ``variant`` 必须**报错**。

    静默丢弃处理效应的结果是「实验看起来没有提升」——
    而这正是最容易被误读成「功能没用」的一种失败。
    """
    cfg = get_config("mini")
    with pytest.raises(ValueError, match="分流"):
        make_experiment_data(cfg, n_users=1000, seed=0, cvr_lift=0.05)
    # 传了 variant 就正常
    v = np.zeros(1000, dtype=np.int8)
    v[500:] = 1
    d = make_experiment_data(cfg, n_users=1000, seed=0, cvr_lift=0.05,
                             variant=v)
    assert len(d) == 1000


def test_pre_period_covariate_is_measured_before_not_during(cfg):
    """``pre_revenue`` 必须与处理**无关** —— 它是实验前的量。

    这是 CUPED 唯一的前提。一旦它带上处理效应，调整就会把效应一起
    消掉，而实验看起来只是「效果消失了」，不报任何错。
    """
    d = make_experiment_data(cfg, n_users=20_000, seed=1)
    v = np.zeros(len(d), dtype=np.int8)
    v[10_000:] = 1
    assert leakage_warning(d.pre_revenue, v) is None
    # 但处理效应确实存在的话，实验期的量就会报警（反面对照）
    d2 = make_experiment_data(cfg, n_users=20_000, seed=1, cvr_lift=2.0,
                              variant=v)
    assert d2.pre_revenue.mean() == pytest.approx(
        make_experiment_data(cfg, n_users=20_000, seed=1,
                             cvr_lift=0.0, variant=v).pre_revenue.mean(),
        rel=0.10)


def test_experiment_data_shapes_are_consistent(cfg):
    d = make_experiment_data(cfg, n_users=3000, seed=2, days=14)
    assert d.n_days == 14
    assert d.daily_clicks.shape == (3000, 14)
    assert np.array_equal(d.clicks, d.daily_clicks.sum(axis=1))
    assert np.array_equal(d.impressions, d.daily_impressions.sum(axis=1))
    assert np.all(d.clicks <= d.impressions)
    assert np.all(d.conversions <= d.clicks)
    assert d.day_is_weekend.sum() == 4


def test_stratum_mask_splits_by_weekday_and_weekend(cfg):
    """``stratum_mask`` 必须按**天**聚合，不能按用户。

    早期版本把每用户一个的 weekday 标记当成逐日标记用，
    结果**每个用户**都被判成「周末」，两个分层完全重合 ——
    这类错误不会报错，只会让分层检验失去意义。
    """
    d = make_experiment_data(cfg, n_users=2000, seed=3, days=28)
    wi, wc, wn = d.stratum_mask(True)
    di, dc, dn = d.stratum_mask(False)
    assert np.all(wi + di == d.impressions)
    assert np.all(wc + dc == d.clicks)
    assert np.all(wn + dn == d.conversions)
    # 周末天数少 → 曝光应该更少（周末单日流量更大，但天数只有 8/28）
    assert wi.sum() < di.sum()


def test_make_experiment_data_validates_input():
    cfg = get_config("mini")
    with pytest.raises(ValueError):
        make_experiment_data(cfg, n_users=0, seed=0)
    with pytest.raises(ValueError):
        make_experiment_data(cfg, n_users=100, seed=0, days=0)
    with pytest.raises(ValueError, match="长度"):
        make_experiment_data(cfg, n_users=100, seed=0,
                             variant=np.zeros(50, dtype=np.int8))


def test_generation_is_reproducible(cfg):
    a = make_experiment_data(cfg, n_users=2000, seed=17)
    b = make_experiment_data(cfg, n_users=2000, seed=17)
    assert np.array_equal(a.conversions, b.conversions)
    assert np.array_equal(a.revenue, b.revenue)
    c = make_experiment_data(cfg, n_users=2000, seed=18)
    assert not np.array_equal(a.conversions, c.conversions)


# ==========================================================================
# 7. AB 报告
# ==========================================================================


@pytest.fixture(scope="module")
def ab_case(cfg):
    """一次带 +30% CVR 提升的 AB。用大提升是为了让**单次**实验稳定可检出 ——
    小提升的检出率本身就是概率事件，拿来做单次断言必然 flaky，
    那条路径由下面「重复 N 次看检出率」的测试覆盖。
    """
    return simulate_experiment(cfg, n_users=12_000, seed=42, exp_id="ab_main",
                               days=28, cvr_lift=0.30)


def test_srm_is_healthy_on_the_simulated_split(ab_case):
    """分流健康是解读一切数字的前提，所以这是报告的第一行。"""
    _, split, rep = ab_case
    assert rep.healthy and rep.srm.p_value > 0.001
    counts = split.counts()
    assert abs(counts[0] - counts[1]) / counts.sum() < 0.02


def test_ab_detects_a_large_known_lift(ab_case):
    """注入了 +30% 的 CVR 提升，检验要认出来，且点估计要落在真值附近。

    ⚠️ ``TestResult.ci_low/ci_high`` 是**绝对差**的区间，
       相对提升的区间要用 ``relative_lift_ci`` 换算 ——
       拿绝对区间去卡相对提升会得到「0.3 不在 [0.014, 0.025] 内」
       这种看着像 bug、实则是口径搞混的失败。
    """
    _, _, rep = ab_case
    assert rep.primary.p_value < 1e-6
    assert rep.primary.lift == pytest.approx(0.30, abs=0.07)
    lo, hi = ST.relative_lift_ci(rep.primary)
    assert lo < 0.30 < hi
    assert rep.primary.method == "delta"
    assert rep.primary.se > 0


def test_primary_metric_denominator_is_clicks(ab_case):
    """主指标是 CVR = 转化/点击，不是 转化/曝光。口径错了整条链路都对不上。"""
    data, split, rep = ab_case
    m = split.variant == 0
    assert rep.primary.control == pytest.approx(
        data.conversions[m].sum() / data.clicks[m].sum())


def test_secondary_metrics_have_the_right_denominators(ab_case):
    data, split, rep = ab_case
    m, t = split.variant == 0, split.variant > 0
    assert rep.secondary["ctr"].control == pytest.approx(
        data.clicks[m].sum() / data.impressions[m].sum())
    assert rep.secondary["roi"].control == pytest.approx(
        data.revenue[m].sum() / data.spend[m].sum())
    assert set(rep.secondary) == {"ctr", "rpm", "roi", "cpa"}


def test_only_the_treated_metric_moves(ab_case):
    """只注入了 CVR 提升 → CTR 不该显著变化。

    这条是「护栏」：如果连 CTR 都跟着显著动，说明处理效应
    被错误地施加到了漏斗的上游，或者两个指标的口径串了。
    """
    _, _, rep = ab_case
    assert rep.secondary["ctr"].p_value > 0.01
    assert abs(rep.secondary["ctr"].lift) < 0.10


def test_cuped_applies_and_helps(ab_case):
    _, _, rep = ab_case
    assert rep.cuped["applied"]
    assert 0.3 < rep.cuped["rho"] < 0.8
    assert rep.cuped["variance_reduction"] > 0.10
    # CUPED **不能改变结论的方向**，点估计也该基本一致
    assert rep.cuped["adjusted_lift"] == pytest.approx(rep.primary.lift,
                                                       abs=0.06)


def test_guardrails_are_user_side_not_bidding_side(ab_case):
    """护栏必须盯「用户有没有被伤害」，不是出价链路的超支率。

    超支率是出价模块的护栏（见 ``tests/test_06_bidding.py``），
    它和用户侧实验无关；早期版本把它放进来，
    结果是两组都恒为 0、指标退化成一个常数。
    """
    _, _, rep = ab_case
    assert set(rep.guardrail) == {"zero_conversion_rate",
                                  "cost_per_conversion"}
    assert np.isfinite(rep.guardrail["zero_conversion_rate"].p_value)


def test_bh_is_applied_to_secondary_metrics(ab_case):
    _, _, rep = ab_case
    assert set(rep.bh_rejected) == set(rep.secondary)
    assert all(isinstance(v, bool) for v in rep.bh_rejected.values())


def test_strata_both_show_the_same_sign(ab_case):
    """「跨越完整业务周期」的落地方式：工作日和周末**必须同号**。

    只在一个时段显著，说明效果是活动带来的，不是功能的。
    """
    _, _, rep = ab_case
    assert set(rep.strata) == {"工作日", "周末"}
    for name, res in rep.strata.items():
        assert res.lift > 0.10, f"{name} 分层：{res}"
        assert res.p_value < 0.05, f"{name} 分层：{res}"


def test_daily_lift_is_stable_around_the_injected_effect(ab_case):
    """逐日提升围绕真值波动，而不是「只有某一天有效果」。"""
    _, _, rep = ab_case
    dl = rep.daily_lift
    assert dl.size == 28 and np.all(np.isfinite(dl))
    assert np.mean(dl) == pytest.approx(0.30, abs=0.12)
    assert np.median(dl) > 0.15
    # 不能靠一两天撑起来：多数天都要是正的
    assert np.mean(dl > 0) > 0.85


def test_report_is_json_serialisable_and_flat(ab_case):
    _, _, rep = ab_case
    d = rep.to_dict()
    s = json.dumps(d)                          # 必须能落盘进 artifacts/
    assert "NaN" not in s.replace('"', '')     # 不允许出现裸 NaN
    assert d["primary"]["method"] == "delta"
    assert d["healthy"] is True
    assert isinstance(d["guardrail"], dict)


def test_report_summary_mentions_srm_first(ab_case):
    """报告的文字顺序也是契约：先看 SRM，再看指标。"""
    _, _, rep = ab_case
    text = rep.summary()
    assert text.index("SRM") < text.index("主指标")


def test_run_ab_rejects_mismatched_split_length(ab_case):
    data, split, _ = ab_case
    with pytest.raises(ValueError, match="长度"):
        run_ab(data, split, exp_id="x") if False else _bad_split(data, split)


def _bad_split(data, split):
    """用一个长度不符的假 split 触发保护分支。"""
    class _Short:
        variant = np.zeros(10, dtype=np.int8)
    return run_ab(data, _Short(), exp_id="x")


def test_run_ab_rejects_empty_arm(cfg):
    d = make_experiment_data(cfg, n_users=500, seed=0)
    with pytest.raises(ValueError, match="非空"):
        run_ab(d, _AllControl(), exp_id="x")


class _AllControl:
    variant = np.zeros(500, dtype=np.int8)


# ==========================================================================
# 8. AB 的重复实验：估计量无偏 + 处理覆盖比例
# ==========================================================================


def test_lift_estimate_is_unbiased_over_replications(cfg):
    """重复 10 次，估计提升的**均值**要落在真值附近。

    单次估计的抖动很大（这正是 deff 的后果），所以只能看均值。
    容差由观测到的标准差反推：``3·sd/sqrt(n_rep)``。
    """
    lifts = [simulate_experiment(cfg, n_users=6000, seed=200 + k,
                                 exp_id=f"unb_{k}", days=28,
                                 cvr_lift=0.25)[2].primary.lift
             for k in range(10)]
    lifts = np.array(lifts)
    se = np.std(lifts, ddof=1) / np.sqrt(lifts.size)
    assert abs(np.mean(lifts) - 0.25) < max(3 * se, 0.02), (
        f"估计提升均值 {np.mean(lifts):.4f}（真值 0.25），各次 {lifts.round(3)}")


def test_lift_fraction_scales_the_effect(cfg):
    """处理只覆盖一半流量 → ATE 应当是 ``lift × fraction`` 的一半。

    这条同时验证了两件事：处理效应**确实**按用户施加
    （而不是全量施加），以及 ``lift_fraction`` 的语义是「覆盖比例」
    而不是「效果打折」。
    """
    full = [simulate_experiment(cfg, n_users=20_000, seed=300 + k,
                                exp_id=f"lf_full_{k}", cvr_lift=0.40)[2]
            .primary.lift for k in range(4)]
    half = [simulate_experiment(cfg, n_users=20_000, seed=300 + k,
                                exp_id=f"lf_half_{k}", cvr_lift=0.40,
                                lift_fraction=0.5)[2].primary.lift
            for k in range(4)]
    assert np.mean(half) == pytest.approx(np.mean(full) / 2, rel=0.30)


def test_zero_lift_produces_no_systematic_effect(cfg):
    """没有处理效应时，估计量的均值应当在 0 附近（AA 的单次版本）。"""
    lifts = np.array([
        simulate_experiment(cfg, n_users=8000, seed=400 + k,
                            exp_id=f"null_{k}", days=28)[2].primary.lift
        for k in range(12)])
    assert abs(np.mean(lifts)) < 0.05, f"均值 {np.mean(lifts):.4f}"


# ==========================================================================
# 9. AA 干跑：断言的是分布，不是单次 p 值
# ==========================================================================


@pytest.fixture(scope="module")
def aa_result(cfg):
    return run_aa(cfg, n_users=6000, replications=80, seed=7)


def test_aa_p_values_are_uniformly_distributed(aa_result):
    """H0 为真 → p 值服从 U(0,1)。**这是 AA 唯一正确的断言方式。**

    ⚠️ 绝不能断言「跑一次 AA，p > 0.05」：那是个 5% 概率失败的断言。
       写进 CI 就是每 20 次红一次，最后被人加 ``flaky`` 标记，
       反而把真正的分流 bug 一起掩盖掉。
    """
    assert aa_result.ks_p_value > 0.05, (
        f"AA 的 p 值偏离均匀分布（KS p={aa_result.ks_p_value:.4f}）—— "
        f"分流链路很可能有系统性偏差")


def test_aa_false_positive_rate_is_near_nominal(aa_result):
    """假阳性率要落在 [0.01, 0.12]（理论 0.05）。

    区间给得宽是因为 80 次重复下 FPR 自身的标准误约
    ``sqrt(0.05·0.95/80) ≈ 0.024``；上界放到 0.12 是为了容忍抖动，
    下界 0.01 则是为了抓「p 值被系统性地做大」这类**保守**的 bug
    （比如方差被高估），它同样会让实验永远测不出真效果。
    """
    fpr = aa_result.false_positive_rate
    assert 0.01 <= fpr <= 0.12, (
        f"AA 假阳性率 {fpr:.3f} 偏离名义值 0.05太远；"
        f"p 值分位数 p10/p50/p90 = "
        f"{np.percentile(aa_result.p_values, [10, 50, 90]).round(3)}")


def test_aa_median_p_value_is_near_one_half(aa_result):
    """均匀分布的另一个必要条件，用来在 KS 不够灵敏时补一刀。"""
    assert 0.25 < np.median(aa_result.p_values) < 0.75


def test_aa_split_counts_always_balance(aa_result):
    """SRM 必须在**每一次** AA 里通过 —— 分流是确定性的，
    它不该因为换了 ``exp_id`` 而时好时坏。"""
    assert np.all(aa_result.srm_p_values > 0.001), (
        f"最小 SRM p={aa_result.srm_p_values.min():.2e}")


def test_aa_replications_use_independent_data(cfg):
    """每一次 AA 都必须**重新生成数据 + 重新分流**。

    拿同一份数据只换分流种子的话，200 个 p 值之间高度相关，
    KS 检验的零分布会失真（KS 假设 p 值独立），
    而跑起来更快、看起来也没问题 —— 所以这件事值得测一下：
    逐次 p 值的自相关应当接近 0。
    """
    res = run_aa(cfg, n_users=4000, replications=40, seed=11)
    ac = float(np.corrcoef(res.p_values[:-1], res.p_values[1:])[0, 1])
    assert abs(ac) < 0.35, f"相邻两次 AA 的 p 值相关系数 {ac:.3f}"


def test_aa_count_and_summary(cfg):
    res = run_aa(cfg, n_users=2000, replications=5, seed=0)
    assert res.p_values.size == 5
    assert res.n_per_group == 1000
    assert "AA" in res.summary() and "假阳性率" in res.summary()
    with pytest.raises(ValueError):
        run_aa(cfg, n_users=1000, replications=0)


def test_aa_and_ab_use_the_same_harness(cfg):
    """AA 就是「lift = 0 的 AB」。若两条路径的实现不同，
    AA 通过就不能保证 AB 的检验没问题。"""
    data = make_experiment_data(cfg, n_users=4000, seed=99)
    split = split_users(data.user_id, "same_harness")
    direct = run_ab(data, split, exp_id="same_harness")
    same = simulate_experiment(cfg, n_users=4000, seed=99,
                               exp_id="same_harness")
    assert direct.primary.p_value == pytest.approx(
        same[2].primary.p_value)
