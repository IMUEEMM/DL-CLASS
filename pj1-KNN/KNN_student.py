# mnist_knn_skeleton_local.py
# 目标：给深度学习的同学的“本地数据+可运行+逐步补全”框架
# - 本地加载 MNIST（支持 IDX+gz 或 mnist.npz）
# - 跑通 sklearn KNN baseline
# - 按 TODO 逐步补全：从零实现 KNN、PCA、Top-k 精度、混淆矩阵、计时对比

import os
import gzip
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    top_k_accuracy_score,
)

import os, gzip, struct, time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score

# ========= 本地数据目录（改成你的路径） =========
DATA_DIR = r"D:\desk\100-work-工作\内蒙古大学\教学\2025秋季\week2\KNN"

# ========= 可调参数 =========
RANDOM_STATE = 42
TRAIN_N = 60000   # 设小一点（如 12000）可更快
TEST_N  = 10000   # 设小一点（如 2000）可更快
USE_PCA = False
N_COMP  = 50
K_LIST  = (1,3,5,7,9)
TOPK    = 5  # None 则跳过 top-k

# ========= 工具 =========
def summarize(results, title):
    print("\n" + title)
    print(f"k\tacc\t\tfit_time(s)\tpredict_time(s)\ttop{TOPK if TOPK else 'N'}_acc")
    for r in results:
        print(f"{r['k']}\t{r['acc']:.4f}\t\t{r['fit_time_s']:.2f}\t\t{r['pred_time_s']:.2f}\t\t{r.get(f'top{TOPK}_acc')}")

def _open_auto(path):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")

def _read_idx_images(path):
    with _open_auto(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051, f"Images magic wrong: {magic}"
        n, rows, cols = struct.unpack(">III", f.read(12))
        data = np.frombuffer(f.read(n*rows*cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)

def _read_idx_labels(path):
    with _open_auto(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049, f"Labels magic wrong: {magic}"
        n = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(n), dtype=np.uint8)
        return data

def _find_first_exist_recursive(root, names):
    # 在 root 下递归寻找 names 列表中的任意一个文件名，找到就返回完整路径
    for dirpath, _, filenames in os.walk(root):
        fl = set(filenames)
        for name in names:
            if name in fl:
                return os.path.join(dirpath, name)
    return None

def _load_idx_folder(root):
    # 支持常见命名（带/不带 .gz，dash 或 dot）
    train_img_names = ["train-images-idx3-ubyte.gz", "train-images-idx3-ubyte",
                       "train-images.idx3-ubyte.gz", "train-images.idx3-ubyte"]
    train_lab_names = ["train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte",
                       "train-labels.idx1-ubyte.gz", "train-labels.idx1-ubyte"]
    test_img_names  = ["t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte",
                       "t10k-images.idx3-ubyte.gz", "t10k-images.idx3-ubyte"]
    test_lab_names  = ["t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte",
                       "t10k-labels.idx1-ubyte.gz", "t10k-labels.idx1-ubyte"]

    p_tr_img = _find_first_exist_recursive(root, train_img_names)
    p_tr_lab = _find_first_exist_recursive(root, train_lab_names)
    p_te_img = _find_first_exist_recursive(root, test_img_names)
    p_te_lab = _find_first_exist_recursive(root, test_lab_names)
    if not all([p_tr_img, p_tr_lab, p_te_img, p_te_lab]):
        return None

    Xtr = _read_idx_images(p_tr_img).astype(np.float32)/255.0
    ytr = _read_idx_labels(p_tr_lab).astype(np.int64)
    Xte = _read_idx_images(p_te_img).astype(np.float32)/255.0
    yte = _read_idx_labels(p_te_lab).astype(np.int64)
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xte = Xte.reshape(len(Xte), -1)
    return Xtr, ytr, Xte, yte

def _load_npz(root):
    # 优先找 mnist.npz；找不到就找任意 .npz 里包含所需 key 的
    candidate = _find_first_exist_recursive(root, ["mnist.npz"])
    if candidate is None:
        # 搜索任何 .npz
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".npz"):
                    path = os.path.join(dirpath, fn)
                    try:
                        with np.load(path) as d:
                            if all(k in d for k in ["x_train","y_train","x_test","y_test"]):
                                candidate = path
                                break
                    except Exception:
                        pass
            if candidate: break
    if candidate is None:
        return None

    with np.load(candidate) as d:
        Xtr, ytr = d["x_train"], d["y_train"]
        Xte, yte = d["x_test"],  d["y_test"]
    if Xtr.ndim == 4:  # 处理通道维
        Xtr = Xtr[..., 0]
        Xte = Xte[..., 0]
    Xtr = (Xtr.astype(np.float32)/255.0).reshape(len(Xtr), -1)
    Xte = (Xte.astype(np.float32)/255.0).reshape(len(Xte), -1)
    ytr = ytr.astype(np.int64)
    yte = yte.astype(np.int64)
    return Xtr, ytr, Xte, yte

def load_mnist_local(root, train_n=TRAIN_N, test_n=TEST_N):
    # 优先用 npz，找不到再尝试 IDX
    data = _load_npz(root)
    src = "npz"
    if data is None:
        data = _load_idx_folder(root)
        src = "idx"
    if data is None:
        raise FileNotFoundError(
            f"未在 {root} 找到 mnist.npz 或 4 个 IDX 文件（train/test 的 images/labels，支持.gz）。"
        )
    Xtr, ytr, Xte, yte = data
    print(f"[load] source={src} | Train raw={Xtr.shape} Test raw={Xte.shape}")

    # 子采样
    rs = np.random.RandomState(RANDOM_STATE)
    if train_n is not None and train_n < len(Xtr):
        idx = rs.choice(len(Xtr), size=train_n, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if test_n is not None and test_n < len(Xte):
        idx = rs.choice(len(Xte), size=test_n, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    print(f"[load] after subsample -> Train={Xtr.shape} Test={Xte.shape}")
    return Xtr, ytr, Xte, yte

# ========= （可选）PCA =========
def pca_reduce(Xtr, Xte, n_components=50):
    """
    TODO（学生完成）：
      - 自己实现最简 PCA（中心化 + SVD），不要直接用 sklearn.PCA
      - 返回 Xtr_pca, Xte_pca（以及解释方差比，可选）
    """
    # 占位：先原样返回，保证可运行；完成后请替换为你的 PCA 实现
    return Xtr, Xte, None

# ========= sklearn 的 KNN baseline =========
def run_sklearn_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9)):
    """
    TODO（学生可选）：
      - 比较 metric/p/weights
      - 计时 fit/predict
      - 计算 top-k 精度（TOPK不为None）
    """
    from sklearn.neighbors import KNeighborsClassifier

    rows = []
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, algorithm="auto")  # TODO: metric='minkowski', p=1/2, weights
        t0 = timer()
        clf.fit(Xtr, ytr)
        fit_t = timer() - t0

        t0 = timer()
        y_pred = clf.predict(Xte)
        pred_t = timer() - t0

        acc = accuracy_score(yte, y_pred)

        topk_acc = None
        if TOPK is not None:
            try:
                proba = clf.predict_proba(Xte)
                topk_acc = top_k_accuracy_score(yte, proba, k=TOPK, labels=clf.classes_)
            except Exception:
                topk_acc = None

        rows.append([f"k={k}", f"acc={acc:.4f}", f"fit={fit_t:.2f}s", f"pred={pred_t:.2f}s", f"top{TOPK}={topk_acc}"])
    summarize("== sklearn KNN summary ==", rows, header="k\tacc\tfit\tpred\ttopk")
    return

# ========= 从零实现的 KNN（学生核心练习）=========
class MyKNN:
    """最小可用 KNN（L2 距离；多数票）"""

    def __init__(self, k=5):
        self.k = k
        self.Xtr = None
        self.ytr = None

    def fit(self, X, y):
        self.Xtr = X
        self.ytr = y

    def _pairwise_distances(self, X):
        """
        TODO（学生完成）：返回 (Nt, N) 的 L2平方距离矩阵
          提示：使用向量化：(a-b)^2 = a^2 + b^2 - 2ab
        """
        # 占位：低效写法，保证可运行；完成后请改成向量化
        Nt = X.shape[0]
        N  = self.Xtr.shape[0]
        dist2 = np.zeros((Nt, N), dtype=np.float32)
        for i in range(Nt):
            diff = self.Xtr - X[i]
            dist2[i] = np.sum(diff * diff, axis=1)
        return dist2

    def _majority_vote(self, labels_1d):
        """
        TODO（学生完成）：多数票；平票时取“票数大且类别索引小”的
        """
        cnt = Counter(labels_1d.tolist())
        return sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[0][0]

    def predict(self, X):
        """
        TODO（学生完成）：
          - 用 np.argpartition 取每行前k小索引
          - 多数票得到预测标签
        """
        dist2 = self._pairwise_distances(X)
        idx = np.argpartition(dist2, kth=range(self.k), axis=1)[:, : self.k]
        neigh_labels = self.ytr[idx]
        y_pred = np.empty(neigh_labels.shape[0], dtype=self.ytr.dtype)
        for i in range(neigh_labels.shape[0]):
            y_pred[i] = self._majority_vote(neigh_labels[i])
        return y_pred

def run_my_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9)):
    """
    TODO（学生完成）：
      - 计时预测
      - 比较不同 k 的准确率
      - （可选）Top-k 精度
    """
    rows = []
    for k in k_list:
        model = MyKNN(k=k)
        model.fit(Xtr, ytr)

        t0 = timer()
        y_pred = model.predict(Xte)
        pred_t = timer() - t0

        acc = accuracy_score(yte, y_pred)
        rows.append([f"k={k}", f"acc={acc:.4f}", f"pred={pred_t:.2f}s"])
    summarize("== MyKNN summary ==", rows, header="k\tacc\tpred")
    return

# ========= 混淆矩阵 =========
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    TODO（学生完成）：
      - 用 confusion_matrix / ConfusionMatrixDisplay 画图
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print("Saved:", save_path)
    else:
        plt.show()

# ========= 主流程 =========
def main():
    # 1) 本地数据
    Xtr, ytr, Xte, yte = load_mnist_local()

    # 2) （可选）PCA
    if USE_PCA:
        Xtr, Xte, _ = pca_reduce(Xtr, Xte, n_components=50)

    # 3) 跑 baseline 或 走手写
    if USE_SKLEARN_BASELINE:
        print("\n=== Running sklearn KNN baseline ===")
        run_sklearn_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9))

        # 混淆矩阵（k=5示例）
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        plot_confusion_matrix(yte, y_pred, title="MNIST KNN (sklearn, k=5)", save_path="cm_sklearn_k5.png")
    else:
        print("\n=== Running your MyKNN (from scratch) ===")
        run_my_knn(Xtr, ytr, Xte, yte, k_list=(1,3,5,7,9))

        model = MyKNN(k=5)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        plot_confusion_matrix(yte, y_pred, title="MNIST KNN (MyKNN, k=5)", save_path="cm_mykNN_k5.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
