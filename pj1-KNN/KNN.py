# mnist_knn_compare_local.py
# 运行：python mnist_knn_compare_local.py
# 依赖：numpy scikit-learn matplotlib

import os, gzip, struct, time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score

# ========= 本地数据目录（改成你的路径；已填入你给的路径） =========
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
def do_pca(X_train, X_test, n_components=N_COMP):
    print(f"Fitting PCA(n_components={n_components}) ...")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE, svd_solver="randomized")
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    print(f"[PCA] time={time.time()-t0:.2f}s | explained_var_sum={pca.explained_variance_ratio_.sum():.4f}")
    return Xtr_pca, Xte_pca

# ========= sklearn KNN 对比 =========
def eval_sklearn_knn(Xtr, ytr, Xte, yte, k_list=K_LIST, metric="minkowski", p=2, weights="uniform", topk=TOPK):
    results = []
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p, weights=weights, algorithm="auto")
        t0 = time.time(); clf.fit(Xtr, ytr); fit_t = time.time()-t0
        t0 = time.time(); y_pred = clf.predict(Xte); pred_t = time.time()-t0
        acc = accuracy_score(yte, y_pred)

        topk_acc = None
        if topk is not None:
            try:
                proba = clf.predict_proba(Xte)
                topk_acc = top_k_accuracy_score(yte, proba, k=topk, labels=clf.classes_)
            except Exception:
                topk_acc = None

        results.append({"k":k, "acc":acc, "fit_time_s":fit_t, "pred_time_s":pred_t, f"top{topk}_acc":topk_acc})
        print(f"[sklearn] k={k:2d} | acc={acc:.4f} | fit={fit_t:.2f}s | pred={pred_t:.2f}s | top{topk}={topk_acc}")
    return results

# ========= 手写 NumPy KNN（子采样+分批距离）=========
def knn_predict_numpy(Xtr, ytr, Xte, k=5, batch_size=500):
    Ntr = Xtr.shape[0]; Nte = Xte.shape[0]
    y_pred = np.zeros(Nte, dtype=np.int64)
    tr_sq = np.sum(Xtr*Xtr, axis=1)
    for start in range(0, Nte, batch_size):
        end = min(start+batch_size, Nte)
        Xb = Xte[start:end]
        te_sq = np.sum(Xb*Xb, axis=1, keepdims=True)
        d2 = te_sq + tr_sq[None,:] - 2.0*(Xb @ Xtr.T)
        idx = np.argpartition(d2, kth=range(k), axis=1)[:, :k]
        neigh = ytr[idx]
        for i in range(neigh.shape[0]):
            cnt = Counter(neigh[i].tolist())
            y_pred[start+i] = sorted(cnt.items(), key=lambda t:(-t[1], t[0]))[0][0]
    return y_pred

def main():
    # 1) 加载本地数据
    X_train, y_train, X_test, y_test = load_mnist_local(DATA_DIR, TRAIN_N, TEST_N)

    # 2) sklearn @ 原始像素
    print("\n=== scikit-learn KNN @ RAW PIXELS ===")
    res_raw = eval_sklearn_knn(X_train, y_train, X_test, y_test)

    # 3) sklearn @ PCA 特征（可选）
    res_pca = []
    if USE_PCA:
        Xtr_pca, Xte_pca = do_pca(X_train, X_test, N_COMP)
        print("\n=== scikit-learn KNN @ PCA FEATURES ===")
        res_pca = eval_sklearn_knn(Xtr_pca, y_train, Xte_pca, y_test)

    # 4) 手写 NumPy KNN（再子采样以便更快对比）
    np.random.seed(RANDOM_STATE)
    tr_sub = min(8000, len(X_train))
    te_sub = min(2000, len(X_test))
    sel_tr = np.random.choice(len(X_train), size=tr_sub, replace=False)
    sel_te = np.random.choice(len(X_test),  size=te_sub, replace=False)
    Xtr_sub, ytr_sub = X_train[sel_tr], y_train[sel_tr]
    Xte_sub, yte_sub = X_test[sel_te],  y_test[sel_te]

    print("\n=== NumPy 手写 KNN（子采样集） ===")
    for k in K_LIST:
        t0 = time.time()
        y_pred_np = knn_predict_numpy(Xtr_sub, ytr_sub, Xte_sub, k=k, batch_size=500)
        pred_t = time.time() - t0
        acc = accuracy_score(yte_sub, y_pred_np)
        print(f"[numpy]  k={k:2d} | acc={acc:.4f} | pred_time={pred_t:.2f}s (train={tr_sub}, test={te_sub})")

    # 5) 混淆矩阵（sklearn k=5 原始像素）
    print("\nPlotting confusion matrix for sklearn KNN (k=5, RAW PIXELS) ...")
    clf = KNeighborsClassifier(n_neighbors=5, algorithm="auto")
    clf.fit(X_train, y_train)
    y_pred_best = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("MNIST KNN (sklearn, k=5) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("mnist_knn_confusion_matrix.png", dpi=150)
    print("Saved: mnist_knn_confusion_matrix.png")

    # 6) 汇总
    summarize(res_raw, "== Summary: sklearn KNN on RAW PIXELS ==")
    if res_pca:
        summarize(res_pca, "== Summary: sklearn KNN on PCA FEATURES ==")
    print("\nDone.")

if __name__ == "__main__":
    main()
