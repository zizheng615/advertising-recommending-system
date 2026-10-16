"""随机种子与确定性。

踩坑记录（血泪）
----------------
1. CPU 上 ``torch.set_num_threads(>1)`` 会让 ``nn.Linear`` 的归约顺序非确定，
   两次训练的 loss 从第 6 位小数开始漂移。**测试里一律用 1 线程做精确断言**，
   脚本里才用多线程换速度。
2. ``torch.use_deterministic_algorithms(True)`` 在 CPU + ``nn.Embedding`` 的
   反向 (index_add) 上会直接抛 RuntimeError，**不要开**。
3. 任何会随线程数改变的结果，都不允许做精确断言。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_threads: bool = False,
             n_threads: int = 6) -> np.random.Generator:
    """设置全部随机源，返回一个独立的 numpy Generator。

    Parameters
    ----------
    seed : int
    deterministic_threads : bool
        True  → 单线程（测试用，保证逐位可复现）
        False → 多线程（脚本用，换速度）
    n_threads : int
        多线程模式下的线程数。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic_threads:
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(n_threads)

    # 不要开 use_deterministic_algorithms，CPU + Embedding 反向会报错
    return np.random.default_rng(seed)


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """构造一个由 (seed, stream) 唯一决定的 Generator。

    用 stream 区分同一个种子下的不同随机用途，避免"改了一处、别处全变"。
    """
    return np.random.default_rng([seed, stream])
