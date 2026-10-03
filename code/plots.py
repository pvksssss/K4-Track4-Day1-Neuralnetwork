"""plots.py — vẽ biểu đồ cho từng thí nghiệm và ảnh chồng theo nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt

_METRIC_LABEL = {
    "val_loss": "val loss",
    "train_loss": "train loss (eval mode)",
    "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1",
    "grad_norm": "grad norm (trước clip)",
    "train_acc": "train accuracy (eval mode)",
}


def cfg_caption(cfg: dict) -> str:
    """Dòng mô tả cấu hình ngắn gọn để đặt trên tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = "none" if cfg.get("clip_norm") is None else f"{cfg['clip_norm']:g}"
    return (f"loss={cfg['loss']} | {cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | "
            f"batch={cfg['batch']} epochs={cfg['epochs']} | hidden={hidden} do={cfg['dropout']:g} | "
            f"clip={clip} prec={cfg['precision']} init={cfg['init']} seed={cfg['seed']}")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg, hist, summ = result["cfg"], result["history"], result["summary"]
    ep = hist["epoch"]
    diverged = summ.get("diverged")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    ax = axes[0]
    ax.plot(ep, hist["train_loss"], "o-", ms=3.5, label="train loss (eval mode)")
    ax.plot(ep, hist["val_loss"], "s-", ms=3.5, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Train / val loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ep, hist["val_acc"], "o-", ms=3.5, label="val accuracy")
    ax.plot(ep, hist["val_macro_f1"], "s-", ms=3.5, label="val macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Val accuracy / macro-F1")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(ep, hist["grad_norm"], "o-", ms=3.5, color="tab:purple", label="mean grad_norm / epoch")
    c = cfg.get("clip_norm")
    if c:
        ax.axhline(float(c), ls="--", color="tab:red", label=f"ngưỡng clip c = {c:g}")
    ax.set_xlabel("epoch"); ax.set_ylabel("L2 global norm")
    ax.set_title("Gradient norm (trước khi cắt)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    be = summ.get("best_epoch") or 0
    for ax in axes:
        if be:
            ax.axvline(be, ls=":", color="gray", lw=1.2)
        ax.set_xlim(left=0)

    flag = "  [DIVERGED]" if diverged else ""
    fig.suptitle(f"{cfg['exp_id']} — {cfg.get('description','')}\n{cfg_caption(cfg)}{flag}",
                 fontsize=9.5, y=1.04)
    fig.text(0.5, -0.02,
             f"step0_loss = {summ['step0_loss']:.4f} (ln7 = 1.9459)   |   "
             f"best_epoch = {be}   |   best_val_loss = {summ['best_val_loss']:.4f}   |   "
             f"val_acc = {summ['val_acc']:.4f}   |   val_macro_f1 = {summ['val_macro_f1']:.4f}   |   "
             f"{summ['time_per_epoch_s']:.1f}s/epoch, peak {summ['peak_mem_MB']:.0f} MB",
             ha="center", fontsize=8.5, color="dimgray")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục, chú thích bằng exp_id."""
    if metric not in _METRIC_LABEL:
        raise ValueError(f"metric chưa hỗ trợ: {metric!r}, chọn trong {tuple(_METRIC_LABEL)}")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for r in results:
        ep, ys = r["history"]["epoch"], r["history"][metric]
        if not ys:
            continue
        label = r["cfg"]["exp_id"]
        if r["cfg"].get("lr") is not None:
            label += f"  (lr={r['cfg']['lr']:g})"
        ax.plot(ep, ys, "o-", ms=3, lw=1.4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(_METRIC_LABEL[metric])
    if metric == "grad_norm":
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title or f"So sánh {_METRIC_LABEL[metric]} theo nhóm thí nghiệm", fontsize=11)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)