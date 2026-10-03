"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import copy
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
LN_C = math.log(N_CLASSES)          # 1.9459 — loss bước 0 kỳ vọng với 7 lớp
N_TRAIN_EVAL_SUBSET = 50_000        # tập con CỐ ĐỊNH để đo train loss (xem _train_subset)

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val ở Part 2.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # chọn bằng VAL ở Part 2 (quét 0.01 / 0.03 / 0.1)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "step"
    seed=1,
    notes="",
)


# --------------------------------------------------------------------------------------
# tiện ích
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Mạng nhỏ, chi phí thấp: ưu tiên chạy lại cho ra cùng kết quả để đo nhiễu seed có ý nghĩa.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom_p = tp + fp
    denom_r = tp + fn
    prec = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
    rec = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)
    denom_f1 = prec + rec
    f1 = np.divide(2 * prec * rec, denom_f1, out=np.zeros_like(tp), where=denom_f1 > 0)
    return float(f1.mean())


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.

    Quy ước MSE: dùng đúng `nn.MSELoss` — KHÔNG có hệ số 1/2, và mặc định lấy trung bình
    trên TẤT CẢ phần tử (B x 7). Với reduction="sum" ta trả tổng của trung bình trên 7 lớp
    của từng mẫu, để `evaluate` chia cho N cho ra cùng thang đo với reduction="mean".
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        onehot = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        sq = (logits - onehot) ** 2
        if reduction == "mean":
            return sq.mean()
        if reduction == "sum":
            return sq.sum() / logits.shape[1]
        return sq.mean(dim=1).sum()
    raise ValueError(f"loss không hợp lệ: {loss_name!r}, chỉ nhận 'ce' hoặc 'mse'")


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(model(X[i:i + batch_size]).argmax(dim=1))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    n = X.shape[0]
    total_loss = torch.zeros((), dtype=torch.float64, device=X.device)
    cm = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64, device=X.device)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total_loss += compute_loss(logits, yb, loss_name, reduction="sum").double()
        pred = logits.argmax(dim=1)
        cm.index_put_((yb, pred), torch.ones_like(pred), accumulate=True)
    cm_np = cm.cpu().numpy()
    return dict(
        loss=float(total_loss / max(n, 1)),
        acc=float(np.trace(cm_np) / max(cm_np.sum(), 1)),
        macro_f1=macro_f1_from_confusion(cm_np),
        confusion=cm_np,
    )


def _train_subset(data: dict, n_sub: int = N_TRAIN_EVAL_SUBSET) -> torch.Tensor:
    """Chỉ số của tập con train CỐ ĐỊNH để đo train loss.

    Cố định (seed 12345, không phụ thuộc seed của thí nghiệm) để mọi đường train loss
    so sánh được với nhau: cùng một tập 50 000 mẫu cho mọi lần chạy.
    """
    if "_train_subset_idx" in data:
        return data["_train_subset_idx"]
    X_tr = data["X_tr"]
    n = X_tr.shape[0]
    g = torch.Generator(device=X_tr.device)
    g.manual_seed(12345)
    idx = torch.randperm(n, generator=g, device=X_tr.device)[:min(n_sub, n)]
    data["_train_subset_idx"] = idx
    return idx


def _amp_context(cfg: dict, device: torch.device):
    """Trả về context cho phép mixed precision, hoặc nullcontext nếu fp32 / không có GPU."""
    prec = cfg.get("precision", "fp32")
    if prec not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {prec!r}")
    if prec == "fp32" or device.type == "cpu":
        return contextlib.nullcontext(), False
    dtype = torch.float16 if prec == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype), True


# --------------------------------------------------------------------------------------
# vòng huấn luyện
# --------------------------------------------------------------------------------------
def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data

    Trả về dict: {"cfg", "history", "summary", "best_state"} (xem docstring đầu hàm).
    TUYỆT ĐỐI không dùng X_eval trong hàm này để chọn epoch hay cấu hình.
    """
    cfg = dict(cfg)
    hidden = tuple(cfg["hidden"])
    loss_name = cfg["loss"]
    set_seed(cfg["seed"])

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device

    # ---- 0. model / optimizer / scaler
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS[hidden], (
        f"{hidden} có {n_params} tham số, quy định là {EXPECTED_PARAMS[hidden]}")
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(X_tr.shape[0] / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"),
                                cfg["epochs"] * steps_per_epoch)
    amp_ctx, amp_on = _amp_context(cfg, device)
    scaler = torch.amp.GradScaler(device.type) if (amp_on and cfg["precision"] == "fp16") else None

    # ---- 1. loss bước 0 TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7)
    step0 = evaluate(model, X_val, y_val, loss_name)
    sub_idx = _train_subset(data)

    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                               "val_macro_f1", "grad_norm", "grad_norm_min", "grad_norm_max",
                               "epoch_time_s", "lr")}
    best = dict(val_loss=float("inf"), epoch=0, state=None, acc=0.0, macro_f1=0.0)
    diverged = False
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        _sync(device)
        t0 = time.time()
        model.train()
        gn_sum, gn_lo, gn_hi, n_steps = 0.0, float("inf"), 0.0, 0

        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with amp_ctx:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                # bỏ hệ số scale TRƯỚC khi đo/cắt gradient, để grad_norm là giá trị thật
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if not torch.isfinite(loss):
                diverged = True
                break
            gn_sum += gn
            gn_lo = min(gn_lo, gn)
            gn_hi = max(gn_hi, gn)
            n_steps += 1

        if diverged:
            print(f"  [{cfg['exp_id']}] DIVERGE ở epoch {epoch}: loss = {loss.detach().item():.3e} -> dừng sớm")
            break

        # ---- cuối epoch: đo lại ở chế độ eval() để train loss và val loss cùng thang đo
        tr = evaluate(model, X_tr[sub_idx], y_tr[sub_idx], loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        _sync(device)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(gn_sum / max(n_steps, 1))
        history["grad_norm_min"].append(gn_lo if n_steps else float("nan"))
        history["grad_norm_max"].append(gn_hi if n_steps else float("nan"))
        history["epoch_time_s"].append(dt)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch, acc=va["acc"], macro_f1=va["macro_f1"],
                        state=copy.deepcopy(model.state_dict()))

        print(f"  [{cfg['exp_id']}] ep {epoch:>3d}/{cfg['epochs']}  "
              f"train_loss {tr['loss']:.4f}  val_loss {va['loss']:.4f}  "
              f"val_acc {va['acc']:.4f}  val_f1 {va['macro_f1']:.4f}  "
              f"gbar {gn_sum/max(n_steps,1):.3e}  {dt:.1f}s")

    _sync(device)
    total_time = time.time() - total_t0
    peak_mem = float(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else 0.0

    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best["val_loss"],
        best_epoch=best["epoch"],
        final_train_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else float("nan"),
        val_acc=best["acc"],
        val_macro_f1=best["macro_f1"],
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else float("nan"),
        peak_mem_MB=peak_mem,
        diverged=diverged,
        # thông tin phụ (không bắt buộc trong bảng)
        n_params=n_params,
        total_time_s=total_time,
        n_epochs_done=len(history["epoch"]),
        amp=bool(amp_on),
        grad_norm_max=float(np.max(history["grad_norm"])) if history["grad_norm"] else float("nan"),
        grad_norm_median=float(np.median(history["grad_norm"])) if history["grad_norm"] else float("nan"),
    )
    if cfg.get("notes"):
        summary["notes"] = cfg["notes"]
    return dict(cfg=cfg, history=history, summary=summary, best_state=best["state"])


# --------------------------------------------------------------------------------------
# dự đoán + ghi file nộp
# --------------------------------------------------------------------------------------
def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert row_id.shape == preds.shape, (row_id.shape, preds.shape)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    assert len(np.unique(row_id)) == len(row_id), "mỗi row_id phải xuất hiện đúng một lần"
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)
    print(f"đã ghi {path}  ({len(row_id)} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> torch.Tensor:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và cho baseline): nạp best_state, dự đoán eval.

    best_state là epoch có val_loss thấp nhất (chọn bằng val, không dùng eval).
    Dự đoán ở chế độ eval() và fp32.
    """
    cfg = dict(cfg)
    device = data["X_tr"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    if result.get("best_state") is None:
        raise RuntimeError(f"không có best_state cho {cfg['exp_id']} (mọi epoch đều lỗi?)")
    model.load_state_dict(result["best_state"])
    model.eval()
    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    print(f"đã dự đoán eval bằng {cfg['exp_id']} (best_epoch = {result['summary']['best_epoch']}, "
          f"val macro-F1 = {result['summary']['val_macro_f1']:.4f})")
    return torch.from_numpy(preds)