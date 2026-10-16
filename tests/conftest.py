"""共享 fixture。

三个提速手段
------------
1. **统一用 mini 档**：2000 用户 / 15 万曝光，生成不到 1 秒。
2. **session 级缓存**：整个测试会话只生成一次数据、只跑一次 PCA/KMeans，
   所有测试文件复用同一份产物。
3. **单线程 torch**：小数据下单线程反而更快，而且**保证逐位可复现** ——
   多线程会让 ``nn.Linear`` 的归约顺序非确定，loss 从小数第 6 位开始漂移，
   任何精确断言都会变成随机失败。
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from config import get_config
from src.data.generator import DataGenerator
from src.data.splits import time_split
from src.data import schema as S
from src.features import pipeline as P, pca as PCA, kmeans as KM
from src.models import dataset as DS
from src.models.deepfm import DeepFMConfig, FactorizationMachineRanker
from src.models.sampling import downsample_negatives, prior_shift
from src.models.train import train_model
from src.models.two_tower import TwoTower, TwoTowerConfig, train_two_tower
from src.models.ziln import ZILN
from src.debias.dfm import DFMModel, ElapsedFeatureModel

warnings.filterwarnings("ignore")

# 测试一律单线程，保证可复现
torch.set_num_threads(1)

_CACHE: dict = {}

# 消融实验统一用的训练配置。三个变体**完全共用**这套超参 ——
# 任何一处不一样，差值就不再只反映模型结构，测试也就失去意义。
ABLATION_EPOCHS = 16
ABLATION_BATCH = 512
ABLATION_LR = 1e-3
ABLATION_EMB = 8
ABLATION_HIDDEN = (64, 32)

_VARIANT_CFG = {"lr": DeepFMConfig.as_lr, "fm": DeepFMConfig.as_fm,
                "deepfm": DeepFMConfig.as_deepfm}


def train_ranker(variant: str, bundle, tensors, train_rows, valid_idx, cfg,
                 *, epochs: int = ABLATION_EPOCHS):
    """按变体名训练一个排序模型，超参与随机种子全局统一。

    早停看 **CTR** 的验证 AUC，不看 CTCVR —— 这不是随手选的：
    mini 档验证集里 CTR 正样本约 3300 个（AUC 标准误 ≈0.005），
    CTCVR 正样本只有约 140 个（标准误 ≈0.02）。用后者选 epoch
    等于在每个 epoch 之间随机抽一个，实测同一份数据换个种子
    AUC 能差 0.05，而这跟模型好坏毫无关系。
    """
    torch.manual_seed(cfg.seed)
    mcfg = _VARIANT_CFG[variant](cfg)
    mcfg.emb_dim = ABLATION_EMB
    mcfg.dnn_hidden = ABLATION_HIDDEN
    model = FactorizationMachineRanker(
        bundle.cat_fields, bundle.imp_dense.shape[1], mcfg)
    hist = train_model(model, tensors, train_rows, valid_idx, epochs=epochs,
                       batch_size=ABLATION_BATCH, lr=ABLATION_LR,
                       weight_decay=cfg.weight_decay,
                       auc_key="y_click", prob_key="p_ctr", seed=cfg.seed)
    return model, hist


@pytest.fixture(scope="session")
def cfg():
    return get_config("mini")


@pytest.fixture(scope="session")
def data(cfg):
    """原始模拟数据（users / items / impressions）。"""
    if "data" not in _CACHE:
        _CACHE["data"] = DataGenerator(cfg).generate()
    return _CACHE["data"]


@pytest.fixture(scope="session")
def users(data):
    return data["users"]


@pytest.fixture(scope="session")
def items(data):
    return data["items"]


@pytest.fixture(scope="session")
def impressions(data):
    return data["impressions"]


@pytest.fixture(scope="session")
def profile(cfg, users):
    """用户画像宽表。"""
    if "profile" not in _CACHE:
        rng = np.random.default_rng([cfg.seed, 77])
        _CACHE["profile"] = P.build_user_profile(users, cfg, rng)
    return _CACHE["profile"]


@pytest.fixture(scope="session")
def pca_result(cfg, profile):
    if "pca" not in _CACHE:
        X = P._safe_float32(profile[0], S.USER_PROFILE_NUMERIC)
        _CACHE["pca"] = PCA.fit_pca(X, cfg)
        _CACHE["pca_X"] = X
    return _CACHE["pca"]


@pytest.fixture(scope="session")
def kmeans_result(cfg, pca_result):
    if "km" not in _CACHE:
        Xpca = pca_result.transform(_CACHE["pca_X"])
        _CACHE["km"] = KM.fit_kmeans(Xpca, cfg)
        _CACHE["pca_space"] = Xpca
    return _CACHE["km"]


@pytest.fixture(scope="session")
def pca_space(pca_result, kmeans_result):
    return _CACHE["pca_space"]


@pytest.fixture(scope="session")
def bundle(cfg, data, pca_result, kmeans_result):
    """特征打包结果（类别矩阵 / 数值矩阵 / 字段定义）。"""
    if "bundle" not in _CACHE:
        _CACHE["bundle"] = P.build_feature_bundle(data, pca_result, kmeans_result, cfg)
    return _CACHE["bundle"]


# ============================================================ 模型
@pytest.fixture(scope="session")
def split(impressions, cfg):
    return time_split(impressions, cfg)


@pytest.fixture(scope="session")
def ranker_tensors(bundle, impressions, cfg):
    return DS.ranker_tensors(bundle, impressions, cfg)


@pytest.fixture(scope="session")
def sampled_train(impressions, split, cfg):
    """1:5 负下采样后的训练行 + 先验校正系数 β。

    ⚠️ 下采样**只在 train 折内**做，β 也相对 train 折算。
       先在全表上下采样再和 train 取交集，β 会把「train 占全表的比例」
       一起吸进去（实测 0.176 而不是 0.262），pCTR 从 0.050 校正成
       0.021 —— 不报错，只是 eCPM 系统性偏低。
    """
    y = impressions["y_click"].to_numpy()[split.train]
    keep = downsample_negatives(y, neg_pos_ratio=cfg.neg_pos_ratio, seed=cfg.seed)
    return split.train[keep], prior_shift(y, keep)


@pytest.fixture(scope="session")
def model_and_batch(bundle, ranker_tensors):
    """一个不训练的 DeepFM 在真实特征上的 forward 输出。

    给「结构性等式」类的断言用（``pCTCVR = pCTR × pCVR``、pCVR ≤ 1）。
    这些性质由**参数化方式**决定，不需要训练就能验证 ——
    也正因如此，坏了很容易漏掉：训练后的 AUC 照样正常。
    """
    mcfg = DeepFMConfig.as_deepfm(DeepFMConfig())
    mcfg.emb_dim, mcfg.dnn_hidden, mcfg.dropout = 8, (16, 8), 0.0
    torch.manual_seed(0)
    model = FactorizationMachineRanker(
        bundle.cat_fields, bundle.imp_dense.shape[1], mcfg)
    model.eval()
    with torch.no_grad():
        return model(ranker_tensors["cat"][:512], ranker_tensors["dense"][:512])


@pytest.fixture(scope="session")
def rankers(bundle, ranker_tensors, sampled_train, split, cfg):
    """三个消融变体各训一次，整个测试会话复用。

    一次训练约 6 分钟（DeepFM 在 40 万行上跑 16 个 epoch），
    所以务必是 session 级 —— 每个断言各训一遍会让测试套件失控。
    """
    if "rankers" not in _CACHE:
        rows, _ = sampled_train
        _CACHE["rankers"] = {
            v: train_ranker(v, bundle, ranker_tensors, rows, split.valid, cfg)
            for v in ("lr", "fm", "deepfm")
        }
    return _CACHE["rankers"]


# ============================================================ 双塔召回
# 双塔的超参也全局统一。tower_dim 从 cfg（64）降到 32、epochs 降到 6，
# 只是为了让测试能跑完 —— 召回任务的收敛比排序慢得多，门槛也相应放宽，
# 见 test_04_recall.py 里对门槛的说明。
TOWER_EPOCHS = 6
TOWER_BATCH = 1024
TOWER_LR = 3e-3
TOWER_DIM = 32


@pytest.fixture(scope="session")
def tower_cfg(cfg) -> TwoTowerConfig:
    return TwoTowerConfig(emb_dim=ABLATION_EMB, tower_hidden=(64, 32),
                          tower_dim=TOWER_DIM, dropout=0.1)


@pytest.fixture(scope="session")
def tower_pairs(bundle, impressions, split, cfg):
    """训练用 (用户, 正素材, 5 个难负素材) 对。只在 train 折内构造。"""
    return DS.build_pairs(bundle, impressions, split.train,
                          n_neg=cfg.neg_pos_ratio, seed=cfg.seed)


@pytest.fixture(scope="session")
def recall_eval(bundle, impressions, split):
    """评测用：valid 折上去重后的 (用户, 点过的素材)。

    ⚠️ 用 valid 而不是 train —— train 上的正样本对模型是「见过的」，
       Recall@50 会虚高到 0.9 以上，那个数字没有任何意义。
    """
    return DS.recall_targets(bundle, impressions, split.valid)


@pytest.fixture(scope="session")
def trained_tower(bundle, tower_pairs, tower_cfg, cfg):
    if "tower" not in _CACHE:
        torch.manual_seed(cfg.seed)
        model = TwoTower(DS.user_cat_fields(bundle), bundle.user_dense_dim,
                         DS.item_cat_fields(bundle), bundle.item_dense_dim,
                         tower_cfg)
        hist = train_two_tower(model, tower_pairs, epochs=TOWER_EPOCHS,
                               batch_size=TOWER_BATCH, lr=TOWER_LR,
                               seed=cfg.seed)
        _CACHE["tower"] = (model, hist)
    return _CACHE["tower"]


# ============================================================ ZILN（用户级）
# ⚠️ 这三个值覆盖了 cfg 里的默认值，理由见 ``trained_ziln`` 的注释：
#    cfg 的 10 epoch × batch 512 在 mini 档只有 60 步，ZILN 完全没训起来。
ZILN_EPOCHS = 60
ZILN_BATCH = 256
ZILN_LR = 3e-3


def _user_split(n_users: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """用户级的随机切分。

    ⚠️ 用户表上**没有时间列** —— LTV 是「这个人一生会付多少钱」，
       不是「这周会付多少」，本来就没有时间轴。所以这里用随机切分，
       与曝光级的 ``time_split`` 不是一回事，不要混用。

       但同一个用户在 train 和 valid 里**不能同时出现**：ZILN 一行一个
       用户，重复出现的那个用户会被模型记住，验证指标虚高。
    """
    rng = np.random.default_rng([seed, 4242])
    perm = rng.permutation(n_users)
    n_valid = max(1, int(round(n_users * 0.25)))
    return perm[n_valid:], perm[:n_valid]


@pytest.fixture(scope="session")
def ziln_data(bundle, users, cfg):
    """用户级张量 + 切分下标。"""
    if "ziln" not in _CACHE:
        n_users = len(bundle.user_cat)
        tr, va = _user_split(n_users, cfg.seed)
        _CACHE["ziln"] = {
            "train": DS.ziln_user_tensors(bundle, users, tr, cfg),
            "valid": DS.ziln_user_tensors(bundle, users, va, cfg),
            "train_idx": tr,
            "valid_idx": va,
        }
    return _CACHE["ziln"]


@pytest.fixture(scope="session")
def trained_ziln(ziln_data, cfg):
    """训好的 ZILN 在 valid 上的输出 + 真实金额。

    训练循环单独写在这里而不是复用 ``train_model``：后者是二分类的
    （一个 logit、一个标签），而 ZILN 一次要回传三个头、标签是连续金额。

    ⚠️ 超参覆盖了 ``cfg`` 里的默认值（10 epoch × batch 512 = **60 步**）。
       60 步下 ZILN 的付费分类 AUC 只有 0.52 —— 和瞎猜一样，
       但 loss 曲线看起来很正常（在降），所以很容易以为「ZILN 就是这个
       水平」。实测把它训到 720 步（60 epoch × batch 256）之后 AUC 才到
       0.63，**接近理论上限 0.70**（上限是把生成器的真实付费概率
       ``_true_pay_prob`` 直接拿来排序得到的，见 ``test_payment_flag_auc``）。

       用户级样本量小、batch 又大，每 epoch 只有十几个 batch ——
       这类「用户级、小样本、多塔」的模型必须按**步数**而不是 epoch 数
       来配预算。
    """
    if "ziln_model" not in _CACHE:
        tr, va = ziln_data["train"], ziln_data["valid"]
        torch.manual_seed(cfg.seed)
        model = ZILN(tr["_fields"], tr["dense"].shape[1], cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=ZILN_LR,
                                weight_decay=cfg.weight_decay)
        n = tr["cat"].shape[0]
        losses: list[float] = []
        for ep in range(ZILN_EPOCHS):
            model.train()
            g = torch.Generator().manual_seed(cfg.seed * 100 + ep)
            perm = torch.randperm(n, generator=g)
            ep_loss = []
            for s in range(0, n, ZILN_BATCH):
                rows = perm[s:s + ZILN_BATCH]
                out = model(tr["cat"][rows], tr["dense"][rows])
                loss, _ = model.loss(out, tr["y"][rows])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss.append(float(loss.detach()))
            losses.append(float(np.mean(ep_loss)))

        model.eval()
        with torch.no_grad():
            out = model(va["cat"], va["dense"])
        _CACHE["ziln_model"] = (model, out, va["y"].numpy().ravel(), losses)
    return _CACHE["ziln_model"]


# ============================================================ 延迟反馈（DFM）
DFM_EPOCHS = 6
DFM_BATCH = 1024
DFM_LR = 1e-3


def _dfm_train(model, tensors, rows, cfg, *, epochs=DFM_EPOCHS, seed=None,
               zero_elapsed: bool = False):
    """DFM / Elapsed-feature / vanilla 共用的训练循环。

    三个模型的 forward 签名一致（``cat, dense, elapsed_h``），
    所以能共用 —— 这也保证了对比的公平性：**唯一的差别只有损失函数**
    （以及 ``zero_elapsed`` 时那个恒为 0 的输入列）。

    ``zero_elapsed=True`` 就是 vanilla：架构和 elapsed 版一模一样，
    只是永远看不到「已经过了多久」这条信息，因此完全无法察觉标签是假的。
    用它而不是另写一个模型，是为了让「有偏标签造成的偏差」这件事
    只由**信息**差异引起，不掺进任何结构差异。
    """
    seed = cfg.seed if seed is None else seed
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=DFM_LR,
                            weight_decay=cfg.weight_decay)
    if zero_elapsed:
        elapsed = torch.zeros_like(tensors["elapsed_h"])
    else:
        elapsed = tensors["elapsed_h"]
    n = len(rows)
    hist = []
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)
        perm = rows[torch.randperm(n, generator=g).numpy()]
        ep_loss = []
        for s in range(0, n, DFM_BATCH):
            b = perm[s:s + DFM_BATCH]
            out = model(tensors["cat"][b], tensors["dense"][b], elapsed[b])
            loss, _ = model.loss(out, tensors["y_conv"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(float(loss.detach()))
        hist.append(float(np.mean(ep_loss)))
    return hist


@pytest.fixture(scope="session")
def dfm_tensors(bundle, impressions, cfg):
    """曝光级张量，带 ``elapsed_h``。标签是**有偏的 24h 窗口标签**。"""
    if "dfm_t" not in _CACHE:
        _CACHE["dfm_t"] = DS.ranker_tensors(bundle, impressions, cfg,
                                            label_col="y_conv",
                                            with_elapsed=True)
    return _CACHE["dfm_t"]


@pytest.fixture(scope="session")
def trained_dfm(bundle, dfm_tensors, split, cfg):
    """三个模型各训一次：DFM / elapsed-as-feature / vanilla。

    vanilla 就是「什么都不做」—— 直接拿有偏标签训一个普通 CVR 模型。
    它是 DFM 存在的理由：如果 vanilla 的偏差本来就不大，那 DFM 这套
    额外复杂度就白加了。
    """
    if "dfm_models" not in _CACHE:
        fields = bundle.cat_fields
        dense_dim = bundle.imp_dense.shape[1]
        rows = torch.from_numpy(split.train.copy())
        out: dict = {}
        for name, ctor in [("dfm", DFMModel),
                           ("elapsed", ElapsedFeatureModel)]:
            torch.manual_seed(cfg.seed)
            m = ctor(fields, dense_dim, cfg)
            out[name] = (m, _dfm_train(m, dfm_tensors, rows, cfg))
        # vanilla：同一个架构，但 elapsed 恒为 0 —— 即「完全不修正」
        torch.manual_seed(cfg.seed)
        van = ElapsedFeatureModel(fields, dense_dim, cfg)
        hist = _dfm_train(van, dfm_tensors, rows, cfg, zero_elapsed=True)
        out["vanilla"] = (van, hist)
        _CACHE["dfm_models"] = out
    return _CACHE["dfm_models"]
