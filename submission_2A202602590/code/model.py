"""model.py — MLP tự định nghĩa cho bài toán 7 lớp, đúng shape cố định của lab.

Model (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden = tuple(hidden)
        self.dropout_p = float(dropout)

        layers: list[nn.Module] = []
        d_in = in_features
        for h in self.hidden:
            layers.append(nn.Linear(d_in, h))        # bias mặc định = True
            layers.append(nn.ReLU())
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))   # chỉ sau ReLU của lớp ẩn
            d_in = h
        layers.append(nn.Linear(d_in, num_classes))  # logit thô, không softmax
        self.net = nn.Sequential(*layers)

        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : giữ khởi tạo mặc định của nn.Linear (KHÔNG phải He)
    """
    if init == "default":
        return                                   # không làm gì: giữ mặc định nn.Linear
    if init not in ("zeros", "normal", "xavier", "he"):
        raise ValueError(f"init không hợp lệ: {init!r}")

    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            if init == "zeros":
                nn.init.zeros_(w)
            elif init == "normal":
                nn.init.normal_(w, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(w)         # Var = 2/(n_in + n_out)
            else:  # "he"
                nn.init.kaiming_normal_(w, nonlinearity="relu")   # Var = 2/n_in
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau MỖI lớp, ở bước 0, trên một lô val.

    Ghi lại std sau mỗi ReLU (tức là std của kích hoạt lớp ẩn) và std của logits ở lớp cuối.
    Trả về danh sách có độ dài = số lớp ẩn + 1.
    """
    was_training = model.training
    model.eval()
    h = x
    stds: list[float] = []
    n_out = getattr(model, "num_classes", None)
    for m in model.net:
        h = m(h)
        if isinstance(m, nn.ReLU) or (isinstance(m, nn.Linear) and m.out_features == n_out):
            stds.append(float(h.std()))
    if was_training:
        model.train()
    return stds