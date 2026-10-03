"""optimizer.py — chọn bộ tối ưu, lập lịch lr và cắt gradient.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "cosine", "step")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Lưu ý: weight_decay của Adam (L2 trộn vào gradient, còn gọi là L2 regularization) khác
    weight_decay của AdamW (suy giảm trọng số tách riêng khỏi gradient). Với weight_decay = 0,
    Adam và AdamW cho cùng một quỹ đạo.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer không hợp lệ: {name!r}, chỉ nhận {OPTIMIZERS}")
    params = list(params)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch tốc độ học theo BƯỚC (total_steps = tổng số bước cập nhật của lần chạy).

    "cosine": CosineAnnealingLR với T_max = total_steps (cần scheduler.step() sau MỖI bước).
    Trả về None nếu name là None.
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(total_steps), 1), eta_min=kwargs.get("eta_min", 0.0))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(int(total_steps) // 3, 1), gamma=kwargs.get("gamma", 0.1))
    raise ValueError(f"scheduler không hợp lệ: {name!r}, chỉ nhận {SCHEDULERS}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None: chỉ đo chuẩn, không cắt (dùng max_norm = inf nên hệ số scale = 1).
    Giá trị trả về chính là `grad_norm` ghi vào lịch sử mỗi bước (để thấy "gai" gradient).

    Với mixed precision FP16 + GradScaler: phải gọi scaler.unscale_(optimizer) TRƯỚC hàm này,
    nếu không chuẩn đo được còn nhân với hệ số scale của GradScaler.
    """
    params = list(params)
    if len(params) == 0:
        return 0.0
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
    return float(total_norm)