"""端到端一键入口 —— 按依赖顺序跑完 01→09。

设计上的三个取舍
----------------
**1. 用子进程，不用 import。**
每个 ``scripts/NN_*.py`` 都能单独重跑，这是调试时最值钱的属性（改一环不用
把整条链路重来）。代价是它们之间有**文件依赖**：02 读 01 落的 pkl，03 读
02 的 bundle……所以这里必须**按顺序**跑，而且中途失败要**立刻停** ——
继续跑下去只会在下游报一堆「缺产物」，把真正的错误埋掉。

**2. 路径全程走 ``pathlib``，subprocess 用 list 形式。**
项目根目录叫 ``D:\\advertising recommending system``，**带空格**。
用 ``shell=True`` 加字符串拼接，空格会把一个路径劈成两个参数，
而且报出来的错是「找不到 scripts/01_generate.py」这种指错方向的信息。

**3. 失败就让退出码非零，并把失败的那一步单独点名。**
``run_all.py`` 是交付验收的主入口，它必须在任何一环挂掉时**明确地挂掉** ——
「跑完了但有几行红字没人注意」是最坏的结果。

用法::

    python run_all.py --scale mini      # 秒级，验证链路通
    python run_all.py --scale small     # 默认，测试与 README 用的档
    python run_all.py --scale full      # 10 万用户，慢
    python run_all.py --from 06         # 从 06 接着跑（前面几步没改就跳过）
    python run_all.py --list            # 只打印计划
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
METRIC_DIR = ROOT / "artifacts" / "metrics"

# 控制台编码 —— 和 scripts/_common.py 里那段是同一件事，这里必须再来一次。
#
# 理由不是「顺便统一一下」：run_all 的输出和它**子进程**的输出是交织在
# 同一个控制台里的，而子脚本经 _common 已经把 stdout 换成了 UTF-8。
# 父进程还留在 GBK 的话，同一个终端里就会出现一半乱码一半正常 ——
# 比整体乱码更难读，因为会让人以为只有某几步有问题。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):        # pragma: no cover
            pass

# (编号, 脚本, 一句话说明) —— 顺序就是依赖顺序，不要重排
STEPS: list[tuple[str, str, str]] = [
    ("01", "01_generate.py",     "模拟数据生成（隐变量真值 + 延迟 + LTV）"),
    ("02", "02_features.py",     "特征工程（192 列）+ PCA + KMeans++ 分层"),
    ("03", "03_train_ranker.py", "LR/FM/DeepFM 消融 + 先验校正 + DFM 去偏"),
    ("04", "04_recall.py",       "双塔召回（内积组合，为 O(1) 检索）"),
    ("05", "05_ziln.py",         "ZILN 零膨胀金额模型"),
    ("06", "06_bidding.py",      "智能出价与水位控制（合成市场六策略）"),
    ("07", "07_experiment.py",   "A/B 实验（AA 空跑 + 功效 + CUPED）"),
    ("08", "08_bidding_sim.py",  "接真实模型的出价模拟"),
    ("09", "09_make_figures.py", "出九张图（只读指标 json，不重算）"),
]


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="small",
                    choices=["mini", "small", "full"])
    ap.add_argument("--from", dest="start", default=None,
                    help="从该编号开始跑（如 06），前面的步骤视为已完成")
    ap.add_argument("--list", action="store_true", help="只打印计划")
    return ap.parse_args()


def _summary(scale: str) -> list[tuple[str, str]]:
    """从落盘的 json 里摘几条头条数字，让 run_all 自己说话。

    ⚠️ 这里**只读 json、不重算**，和 09 出图同一个原则：屏幕上、README 里、
    图上出现的数字必须来自**同一次运行**。重算就有两个真相了。
    """
    def load(name: str):
        p = METRIC_DIR / f"{name}_{scale}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    out: list[tuple[str, str]] = []
    try:
        d = load("01_data")
        if d:
            out.append(("数据", f"CTR {d['ctr']:.4f}（目标 {d['target_ctr']}）"
                                f"、CVR|click {d['cvr']:.4f}"
                                f"（目标 {d['target_cvr']}）"))
        d = load("02_features")
        if d:
            out.append(("分层", f"K={d['kmeans']['k']}、轮廓 "
                                f"{d['kmeans']['silhouette']:.4f}、"
                                f"ARI {d['kmeans']['ari_vs_true_segment']:.4f}"))
        d = load("03_ranker")
        if d:
            ab = {r["variant"]: r["auc_ctr"] for r in d["ablation"]}
            out.append(("排序", f"LR {ab['lr']:.4f} → FM {ab['fm']:.4f}"
                                f" → DeepFM {ab['deepfm']:.4f}"))
        d = load("05_ziln")
        if d:
            out.append(("ZILN", f"AUC {d['metrics']['auc_pay']:.4f}、"
                                f"Gini {d['metrics']['normalized_gini']:.4f}"))
        d = load("06_bidding")
        if d:
            out.append(("出价(合成)", f"B4 相对固定出价 "
                                      f"+{d['headline']['lift_vs_fixed_bid']:.0%}、"
                                      f"达上界 "
                                      f"{d['headline']['ratio_b4_to_oracle']:.1%}"))
        d = load("08_bidding_sim")
        if d:
            b = d["best_variant"]
            out.append(("出价(真模型)", f"{b} B4/B5 "
                                        f"{d['runs'][b]['ratio_b4_to_oracle']:.4f}、"
                                        f"相对 B0 "
                                        f"+{d['runs'][b]['lift_vs_fixed_bid']:.0%}"))
        d = load("07_experiment")
        if d:
            out.append(("实验", f"AA 的 KS p={d['aa']['ks_pvalue']:.3f}、"
                                f"假阳性 {d['aa']['false_positive_rate']:.1%}、"
                                f"主指标提升 "
                                f"{d['ab']['primary']['lift']:.2%}"
                                f"(p={d['ab']['primary']['p_value']:.4f})"))
        d = load("09_figures")
        if d:
            out.append(("出图", f"{d['n_figures']} 张"))
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        out.append(("摘要", f"读取失败：{type(e).__name__}: {e}"))
    return out


def main() -> int:
    args = _parse()
    steps = STEPS
    if args.start:
        idx = [i for i, (n, _, _) in enumerate(STEPS) if n == args.start]
        if not idx:
            print(f"未知的起始编号 {args.start!r}，可选："
                  f"{', '.join(n for n, _, _ in STEPS)}", file=sys.stderr)
            return 2
        steps = STEPS[idx[0]:]

    print("=" * 72)
    print(f"广告推荐与智能出价系统 —— 端到端（--scale {args.scale}）")
    print("=" * 72)
    for n, s, desc in steps:
        print(f"  {n}  {s:22} {desc}")
    print(f"\n解释器：{sys.executable}")
    print(f"产物目录：{ROOT / 'artifacts'}")
    if args.list:
        return 0

    results: list[tuple[str, str, float]] = []
    t_all = time.time()
    for n, s, desc in steps:
        script = SCRIPTS / s
        if not script.exists():
            print(f"\n[{n}] 找不到 {script}", file=sys.stderr)
            return 2
        print("\n" + "-" * 72)
        print(f"[{n}] {desc}")
        print("-" * 72, flush=True)
        t0 = time.time()
        # list 形式 + 不带 shell：路径里的空格不会被拆开
        proc = subprocess.run([sys.executable, "-I", str(script),
                               "--scale", args.scale])
        dt = time.time() - t0
        ok = proc.returncode == 0
        results.append((n, "通过" if ok else f"失败(exit {proc.returncode})", dt))
        if not ok:
            print(f"\n[{n}] 失败，已停止。修好后可以接着跑："
                  f"python run_all.py --scale {args.scale} --from {n}",
                  file=sys.stderr)
            break

    total = time.time() - t_all
    print("\n" + "=" * 72)
    print("各步耗时")
    print("=" * 72)
    for n, st, dt in results:
        print(f"  {n}  {st:14} {dt:8.1f}s")
    print(f"  {'合计':<16} {total:8.1f}s")

    failed = [r for r in results if r[1] != "通过"]
    if failed:
        print(f"\n有 {len(failed)} 步失败："
              f"{', '.join(n for n, _, _ in failed)}", file=sys.stderr)
        return 1

    if len(results) == len(STEPS):
        print("\n" + "=" * 72)
        print("头条数字（直接读自 artifacts/metrics/*.json，未重算）")
        print("=" * 72)
        for k, v in _summary(args.scale):
            print(f"  {k:12} {v}")

    print(f"\n端到端跑通（{len(results)}/{len(STEPS)} 步，{total:.1f}s）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
