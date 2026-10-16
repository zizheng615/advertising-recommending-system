"""分流：把用户确定性地哈希进桶。

**分流单元是 ``user_id``，不是曝光、不是 session。**
按曝光分流的实验是常见的错误做法：同一个用户一会儿看 A 一会儿看 B，
两组之间不再是独立人群，学习效应和用户体验断层会同时污染结果，
而且「同一用户既在实验组又在对照组」这件事在任何汇总指标上都看不出来。

**为什么必须是 blake2b 而不是内置 ``hash()``**

CPython 对 ``str`` 的 ``hash()`` 带 **per-process 随机盐**
（``PYTHONHASHSEED``，默认每次启动都变）。用它分流意味着：

* 同一批用户在**同一个实验**里，早上分到 A 组、重启服务后分到 B 组 ——
  用户会被反复「重新随机」，实验彻底失效；
* 本地跑出来的结果和服务端跑出来的结果对不上，且**无法复现**。

这个坑的恶劣之处在于单进程调试时一切正常（同一进程内 ``hash`` 稳定），
只有上线或重启才暴露。所以哈希必须用**跨进程稳定**的算法。
这里用 ``hashlib.blake2b``：标准库自带、速度够（约 1.5M 次/秒）、
输出 64 位足够避免冲突。

⚠️ 另外两个常见错误：
1. ``user_id % 2`` —— 用户 ID 往往**不是随机的**（自增、带区段含义、
   按注册时间分配）。取模会把「ID 的规律」直接变成「分组的规律」，
   例如 ID 全部是奇数时全进一组。必须先哈希再取模。
2. 不同实验复用同一个盐 —— 那么「在实验 1 里分到 A 组的人」
   和「在实验 2 里分到 A 组的人」是**同一批人**。
   整体上不影响单次实验的正确性，但两个实验的效果会互相干扰，
   而且无法做正交分层（同一个用户同时进多个互不干扰的实验）。
   所以 ``exp_id`` 必须参与哈希。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# 每个实验的分桶数。10000 桶 + 50/50 切分 → 每组 5000 桶，
# 桶级随机性带来的分流偏差约 1/sqrt(5000) ≈ 1.4%，
# 相对「按用户逐个哈希」的差别可以忽略（SRM 检验会兜住）。
DEFAULT_BUCKETS = 10_000

_DIGEST_SIZE = 8          # 64 位，10 万用户下碰撞概率约 2.7e-10


def user_bucket(user_id, exp_id: str, salt: str = "",
                n_buckets: int = DEFAULT_BUCKETS) -> int:
    """单个用户落在哪个桶。**跨进程、跨机器、跨语言可复现。**

    参与哈希的字符串是 ``f"{exp_id}|{salt}|{user_id}"``。
    分隔符 ``|`` 不能省：没有它 ``("1", "23")`` 和 ``("12", "3")``
    会哈希到同一个串上，两个实验（或两个用户）就撞了。
    """
    if n_buckets <= 0:
        raise ValueError("桶数必须为正")
    key = f"{exp_id}|{salt}|{user_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=_DIGEST_SIZE).digest()
    return int.from_bytes(digest, "big") % n_buckets


def assign_buckets(user_ids, exp_id: str, salt: str = "",
                   n_buckets: int = DEFAULT_BUCKETS) -> np.ndarray:
    """批量分流，返回每个用户的桶号（``int32`` 数组）。

    10 万用户约 0.07 秒 —— 纯 Python 循环足够，不值得为它上 numba。
    """
    return np.fromiter(
        (user_bucket(u, exp_id, salt, n_buckets) for u in user_ids),
        dtype=np.int32, count=len(np.asarray(user_ids).reshape(-1))
        if hasattr(user_ids, "__len__") else -1)


def variant_of(bucket: np.ndarray, n_variants: int,
               n_buckets: int = DEFAULT_BUCKETS) -> np.ndarray:
    """桶号 → 变体编号（``0`` 是对照组）。

    按**等宽的桶区间**切分，而不是 ``bucket % n_variants``：
    取模在 ``n_buckets`` 不是 ``n_variants`` 的整数倍时会让前面的变体
    多拿几个桶，比例偏差约 ``1/n_buckets`` —— 单看微不足道，
    但它是个**系统性**偏差，而 SRM 检验对这个量级非常敏感
    （这正是 SRM 存在的意义：它专门抓这种「看起来没差多少」的泄漏）。

    ⚠️ ``n_buckets`` 必须显式传入，不能从 ``bucket.max()`` 推断：
       抽样后最大桶号几乎必然小于 ``n_buckets``（10000 桶抽 4000 人，
       空桶一大片），推断出来的分母会让切分点整体偏移。
    """
    if n_variants < 2:
        raise ValueError("至少要有两个变体（对照 + 实验）")
    b = np.asarray(bucket)
    if b.size == 0:
        return np.zeros(0, dtype=np.int8)
    if b.min() < 0 or b.max() >= n_buckets:
        raise ValueError(f"桶号必须在 [0, {n_buckets}) 内")
    # floor(b · k / N)：每个变体拿到的桶数相差不超过 1，
    # 且切分点只依赖桶号，与到达顺序无关。
    return np.minimum(b * n_variants // n_buckets,
                      n_variants - 1).astype(np.int8)


@dataclass
class SplitResult:
    """一次分流的产物。"""

    user_id: np.ndarray
    bucket: np.ndarray
    variant: np.ndarray

    @property
    def n_variants(self) -> int:
        return int(self.variant.max()) + 1 if self.variant.size else 0

    def mask(self, variant: int) -> np.ndarray:
        return self.variant == variant

    def counts(self) -> np.ndarray:
        return np.bincount(self.variant, minlength=max(2, self.n_variants))


def split_users(user_ids, exp_id: str, *, salt: str = "",
                n_variants: int = 2,
                n_buckets: int = DEFAULT_BUCKETS) -> SplitResult:
    """分流入口：``user_ids`` → 每个用户的桶号与变体。"""
    uid = np.asarray(user_ids).reshape(-1)
    if np.unique(uid).size != uid.size:
        raise ValueError("user_id 必须唯一，同一用户重复进入会被分到同一个桶"
                         "但在汇总时会重复计数")
    bucket = assign_buckets(uid, exp_id, salt, n_buckets)
    return SplitResult(user_id=uid, bucket=bucket,
                       variant=variant_of(bucket, n_variants, n_buckets))
