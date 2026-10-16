"""04 双塔召回：2000 素材 → Top-50，为排序层挡掉 97.5% 的候选。

产出
----
``artifacts/models/two_tower_{scale}.pkl``     双塔模型
``artifacts/data/tower_pairs_{scale}.pkl``     训练用的 (用户, 正素材, 5 难负)
``artifacts/metrics/04_recall_{scale}.json``

**这一步真正要回答的问题**

Recall@50 本身不是重点。mini 档素材库只有 400 个，Top-50 已经覆盖
12.5% 的全库 —— 随机基线就有 0.125，热门零模型能到 0.33。
所以「Recall@50 = 0.35」这个数字单独看毫无意义。

有意义的是两个**约束**，它们才是双塔这个结构存在的理由：

1. **user tower 的输入里不能有任何跨域特征**。一旦吃进
   ``user_x_category_ctr`` 这类同时依赖用户和素材的列，用户向量就依赖
   具体素材，**离线预算 item 向量这件事直接失效** —— 双塔退化成一个
   更差的排序模型。而它照样能训、指标还更好看，没有任何报错。
2. **两个塔的输出必须是 L2 归一化的单位向量**。少了这一步，
   内积退化成「谁的模长大谁赢」，模型会去学「把热门素材的向量整体放大」
   这种平凡解。

指标那一侧只留「必须显著打赢热门零模型」—— 一个个性化的召回模型如果
打不过「给所有人推最热的 50 个」，它的价值就只剩「在线 O(1) 打分」。
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
from src.data import schema as S                            # noqa: E402
from src.models import dataset as DS                        # noqa: E402
from src.models.two_tower import (TwoTower, TwoTowerConfig,  # noqa: E402
                                  recall_at_k, train_two_tower)

# 双塔的超参独立一套。学习率比排序模型大一个量级（3e-3 vs 1e-3）：
# 召回任务的梯度信号稠密得多（每个 batch 有 batch_size 个正样本参与
# 竞争，而不是 batch_size × 3% 个），而且 L2 归一化之后内积尺度很小。
TOWER_EPOCHS = 6
TOWER_BATCH = 1024
TOWER_LR = 3e-3
TOWER_DIM = 32


def cross_domain_names() -> set[str]:
    """跨域特征列名 —— 同时依赖「用户」和「素材」两边的那些列。"""
    return set(S.IMPRESSION_AD_CONTEXT) | set(S.CROSS_COUNT_NAMES)


def user_tower_audit(bundle, tower: TwoTower) -> dict:
    """检查 user tower 到底吃进了哪些字段。

    判据不是「代码里没写」而是**从构造好的模型上把字段名读回来** ——
    这样才能拦住「有人在 ``user_cat_fields`` 里偷偷加了一列」。
    """
    cross = cross_domain_names()
    # 从模型上读，而不是从 DS.user_cat_fields 读：
    # 后者是「本该是什么」，前者是「实际建成了什么」。
    u_names = [n for n, _ in tower.user_tower.bank.cat_fields]
    leaked = sorted(set(u_names) & cross)
    return {
        "user_fields": u_names,
        "n_user_fields": len(u_names),
        "cross_domain_columns": sorted(cross),
        "leaked": leaked,
    }


def l2_audit(tower: TwoTower, bundle) -> dict:
    """两个塔的输出必须是单位向量。"""
    u_cat = torch.from_numpy(bundle.user_cat[:64]).long()
    u_dense = torch.from_numpy(bundle.user_dense[:64]).float()
    i_cat = torch.from_numpy(bundle.item_cat[:64]).long()
    i_dense = torch.from_numpy(bundle.item_dense[:64]).float()
    tower.eval()
    with torch.no_grad():
        # encode_* 返回 (向量, 主效应标量) 两元组，范数只对向量那一项取
        hu, _ = tower.encode_users(u_cat, u_dense)
        hi, _ = tower.encode_items(i_cat, i_dense)
    nu = hu.norm(dim=1).numpy()
    ni = hi.norm(dim=1).numpy()
    return {
        "user_norm_err": float(np.abs(nu - 1.0).max()),
        "item_norm_err": float(np.abs(ni - 1.0).max()),
        "tower_dim": int(hu.shape[1]),
    }


def popularity_top_k(imp, split, n_items: int, k: int) -> np.ndarray:
    """零模型：把 train 折里点击最多的 K 个素材推给**所有人**。

    它是召回真正的对手。用 train 折统计（不能用全表 —— 那是偷看未来）。
    """
    tr = imp.iloc[split.train]
    pop = (tr.groupby("ad_id")["y_click"].sum()
           .reindex(range(n_items), fill_value=0).to_numpy())
    return np.argsort(-pop)[:k]


def main() -> int:
    scale, _ = parse_scale(__doc__.splitlines()[0])
    cfg = cfg_for(scale)
    set_seed(cfg.seed)
    torch.set_num_threads(1)

    data = load_data(cfg)
    imp = data["impressions"]
    split = load_split(cfg)
    bundle = load_bundle(cfg)

    with Timer(f"04 构造训练对（{scale}）"):
        pairs = DS.build_pairs(bundle, imp, split.train,
                               n_neg=cfg.neg_pos_ratio, seed=cfg.seed)
        save_artifact(pairs, DATA_DIR / "tower_pairs.pkl", cfg)

    tower_cfg = TwoTowerConfig(emb_dim=cfg.emb_dim, tower_hidden=(64, 32),
                               tower_dim=TOWER_DIM, dropout=0.1)
    torch.manual_seed(cfg.seed)
    tower = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                     DS.item_cat_fields(bundle), bundle.item_dense_dim,
                     tower_cfg)

    with Timer(f"04 训练双塔（{scale}）"):
        hist = train_two_tower(tower, pairs, epochs=TOWER_EPOCHS,
                               batch_size=TOWER_BATCH, lr=TOWER_LR,
                               seed=cfg.seed)
        save_artifact(tower, MODEL_DIR / "two_tower.pkl", cfg)

    n_items = bundle.item_cat.shape[0]
    n_cand = min(50, n_items)
    K = 50

    banner("训练")
    log.info("正样本对 %d 条 × (1 正 + %d 难负)，loss %.4f → %.4f，温度 τ=%.4f",
             pairs["u_cat"].shape[0], cfg.neg_pos_ratio,
             hist["train_loss"][0], hist["train_loss"][-1], hist["temperature"])

    # ---------------- 结构约束 ----------------
    audit = user_tower_audit(bundle, tower)
    norm = l2_audit(tower, bundle)
    banner("结构约束")
    log.info("user tower 吃进 %d 个字段；跨域列 %d 个，泄漏进去的：%s",
             audit["n_user_fields"], len(audit["cross_domain_columns"]),
             audit["leaked"] or "无")
    log.info("塔输出 L2 范数误差：user %.2e / item %.2e（维度 %d）",
             norm["user_norm_err"], norm["item_norm_err"], norm["tower_dim"])

    # ---------------- 召回质量 ----------------
    rows = []
    for name, idx in (("train", split.train), ("valid", split.valid),
                      ("test", split.test)):
        t, targets = DS.recall_targets(bundle, imp, idx, seed=cfg.seed)
        r = recall_at_k(tower, t["u_cat"], t["u_dense"],
                        t["i_cat"], t["i_dense"], targets, k=K)
        rows.append({"split": name, "n_pairs": len(targets), f"recall@{K}": r,
                     "random": n_cand / n_items})

    pop_top = popularity_top_k(imp, split, n_items, K)
    ev_t, ev_targets = DS.recall_targets(bundle, imp, split.valid, seed=cfg.seed)
    pop_recall = float(np.isin(ev_targets.numpy(), pop_top).mean())
    model_recall = rows[1][f"recall@{K}"]

    banner(f"Recall@{K}（候选池 {n_items} 个素材，Top-{K} 覆盖 "
           f"{n_cand / n_items:.1%} 的全库）")
    show_table(rows, ["split", "n_pairs", f"recall@{K}", "random"], "分折")
    log.info("热门零模型（train 折点击最多的 %d 个推给所有人）= %.4f",
             K, pop_recall)
    log.info("双塔 = %.4f，相对热门 %+.4f。"
             "⚠️ mini 档这个任务的可达上限就在热门基线附近："
             "素材只有 %d 个、每用户约百条曝光，而且**召回阶段看不到上下文**，"
             "而生成器把大量点击信号放在了时段/页面位置的高阶项里",
             model_recall, model_recall - pop_recall, n_items)

    # ---------------- 门槛 ----------------
    problems = []
    if audit["leaked"]:
        problems.append(f"user tower 吃进了跨域特征：{audit['leaked']} —— "
                        f"离线预算 item 向量的前提直接失效")
    if norm["user_norm_err"] > 1e-4 or norm["item_norm_err"] > 1e-4:
        problems.append(f"塔输出没有 L2 归一化（误差 "
                        f"{norm['user_norm_err']:.2e}/{norm['item_norm_err']:.2e}）")
    if not model_recall > 0.30:
        problems.append(f"Recall@{K} = {model_recall:.4f} ≤ 0.30")
    if not model_recall > pop_recall + 0.01:
        problems.append(f"Recall@{K} = {model_recall:.4f} 没有超过热门零模型 "
                        f"{pop_recall:.4f} —— 个性化的价值不成立")
    if not hist["train_loss"][-1] < hist["train_loss"][0]:
        problems.append("训练损失没有下降，模型根本没在学")

    metrics = {
        "scale": scale,
        "train": {"n_pairs": int(pairs["u_cat"].shape[0]),
                  "n_neg": int(cfg.neg_pos_ratio),
                  "epochs": TOWER_EPOCHS,
                  "loss_first": hist["train_loss"][0],
                  "loss_last": hist["train_loss"][-1],
                  "temperature": hist["temperature"]},
        "structure": {"n_user_fields": audit["n_user_fields"],
                      "leaked_cross_columns": audit["leaked"],
                      "user_norm_err": norm["user_norm_err"],
                      "item_norm_err": norm["item_norm_err"],
                      "tower_dim": norm["tower_dim"]},
        "recall": {"k": K, "n_items": n_items,
                   "per_split": rows,
                   "popularity_baseline": pop_recall},
        "problems": problems,
    }
    save_metrics("04_recall", metrics, cfg)

    if problems:
        for p in problems:
            log.error("体检不通过：%s", p)
        return 1
    log.info("体检通过：user tower 无跨域特征、塔输出已归一化，"
             "Recall@%d = %.4f > 热门 %.4f", K, model_recall, pop_recall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
