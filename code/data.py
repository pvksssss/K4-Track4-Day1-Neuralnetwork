"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
N_TRAIN_TOTAL = 464_809
N_EVAL_TOTAL = 116_203


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    d_tr = np.load(f"{processed_dir}/train.npz", allow_pickle=False)
    d_ev = np.load(f"{processed_dir}/eval.npz", allow_pickle=False)

    X_train_full, y_train_full = d_tr["X"], d_tr["y"]
    X_eval, y_eval, eval_row_id = d_ev["X"], d_ev["y"], d_ev["row_id"]

    # ---- kiểm tra tính toàn vẹn theo quy ước ở đầu file
    assert X_train_full.shape == (N_TRAIN_TOTAL, N_FEATURES), X_train_full.shape
    assert X_eval.shape == (N_EVAL_TOTAL, N_FEATURES), X_eval.shape
    assert X_train_full.dtype == np.float32 and X_eval.dtype == np.float32
    assert y_train_full.dtype == np.int64 and y_eval.dtype == np.int64
    assert eval_row_id.dtype == np.int64 and eval_row_id.shape == (N_EVAL_TOTAL,)
    for y in (y_train_full, y_eval):
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1, "nhãn phải nằm trong 0..6 (đã trừ 1)"
    assert len(np.unique(eval_row_id)) == N_EVAL_TOTAL, "row_id của eval phải khác nhau"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    # train_test_split trả về bản sao không đảo chiều -> ép lại float32/int64 cho chắc
    return (
        np.ascontiguousarray(X_tr, dtype=np.float32),
        np.ascontiguousarray(y_tr, dtype=np.int64),
        np.ascontiguousarray(X_val, dtype=np.float32),
        np.ascontiguousarray(y_val, dtype=np.int64),
    )


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))

    Vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
      Vì mean/std là thống kê của tập huấn luyện. Dùng thống kê của val/eval để chuẩn hoá
      là rò rỉ thông tin (information leakage): thông tin của tập đánh giá chảy ngược
      vào quá trình huấn luyện, làm val/eval lạc quan và đánh giá không còn trung thực.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)  # float64 để mean/std chính xác hơn
    mean = num.mean(axis=0)
    std = num.std(axis=0)                        # ddof=0: khớp với định nghĩa chuẩn hoá
    std = np.where(std < 1e-12, 1.0, std)        # tránh chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X = np.array(X, dtype=np.float32, copy=True)
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    # 1. nạp -> tách val -> chuẩn hoá bằng thống kê của X_tr
    X_train_full, y_train_full, X_eval_np, y_eval_np, eval_row_id = load_split(processed_dir)
    X_tr_np, y_tr_np, X_val_np, y_val_np = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr_np)          # CHỈ trên phần train còn lại
    X_tr_np = apply_standardizer(X_tr_np, mean, std)
    X_val_np = apply_standardizer(X_val_np, mean, std)
    X_eval_np = apply_standardizer(X_eval_np, mean, std)   # cùng mean/std, KHÔNG tính lại

    # 2. lên thiết bị một lần duy nhất
    dev = torch.device(device)
    data = dict(
        X_tr=torch.from_numpy(X_tr_np).to(dev),
        y_tr=torch.from_numpy(y_tr_np).to(dev),
        X_val=torch.from_numpy(X_val_np).to(dev),
        y_val=torch.from_numpy(y_val_np).to(dev),
        X_eval=torch.from_numpy(X_eval_np).to(dev),
        y_eval=torch.from_numpy(y_eval_np).to(dev),
        eval_row_id=eval_row_id,
        mean=mean, std=std,
        val_fraction=val_fraction, split_seed=seed,
    )

    # 3. báo cáo + mốc thấp nhất mà mô hình phải vượt
    n_tr, n_val, n_ev = len(data["X_tr"]), len(data["X_val"]), len(data["X_eval"])
    print(f"thiết bị: {dev}")
    print(f"X_tr  {tuple(data['X_tr'].shape)}  |  X_val {tuple(data['X_val'].shape)}"
          f"  |  X_eval {tuple(data['X_eval'].shape)}")
    print(f"số mẫu: train {n_tr} / val {n_val} / eval {n_ev}  "
          f"(kỳ vọng {int(round(N_TRAIN_TOTAL*(1-val_fraction)))} / "
          f"{N_TRAIN_TOTAL-n_tr} / {n_ev})")
    assert n_tr + n_val == N_TRAIN_TOTAL

    cnt_val = torch.bincount(data["y_val"], minlength=N_CLASSES).float()
    majority = int(cnt_val.argmax())
    acc_major = float((data["y_val"] == majority).float().mean())
    print(f"lớp đa số trên val = {majority} ({100*float(cnt_val[majority]/cnt_val.sum()):.2f}% mẫu); "
          f"chiến lược 'luôn đoán lớp đa số' cho val acc = {acc_major:.4f}  <- mốc phải vượt")
    data["majority_class"] = majority
    data["majority_acc_val"] = acc_major

    # 4. kiểm tra chuẩn hoá: 10 cột đầu trên X_tr phải có mean ~= 0, std ~= 1
    mu = data["X_tr"][:, :N_NUMERIC].mean(0)
    sd = data["X_tr"][:, :N_NUMERIC].std(0, unbiased=False)
    print(f"X_tr 10 cột đầu sau chuẩn hoá: |mean|max = {float(mu.abs().max()):.2e}, "
          f"|std-1|max = {float((sd-1).abs().max()):.2e}")
    binary = data["X_tr"][:, N_NUMERIC:]
    # 4 cột Wilderness + 40 cột Soil: mỗi khối đều là one-hot nên mỗi hàng có đúng 8 số 1
    w_ok = bool((binary[:, :4].sum(1) == 1).all())
    s_ok = bool((binary[:, 4:].sum(1) == 1).all())
    print(f"44 cột nhị phân giữ nguyên: giá trị ∈ {{{float(binary.min()):.0f}, {float(binary.max()):.0f}}}, "
          f"mỗi hàng có 8 số 1: {bool((binary.sum(1) == 8).all())} | "
          f"one-hot Wilderness ok: {w_ok} | one-hot Soil ok: {s_ok}")
    assert w_ok and s_ok, "44 cột sau phải là 2 khối one-hot (4 + 40)"
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Xáo bằng `torch.randperm(N)` rồi cắt lô. Lô cuối có thể nhỏ hơn batch_size;
    ta GIỮ NGUYÊN lô cuối (không bỏ mẫu nào) và gradient được trung bình theo số mẫu
    trong lô nên lô nhỏ hơa có đóng góp nhỏ hơn trên mỗi mẫu.
    """
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]