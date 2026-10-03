# REPORT.md — bản nháp có sẵn phần cơ chế (chép thành `submission_<MSSV>/REPORT.md` rồi điền số)

> **Cách dùng.** Copy file này thành `submission_<MSSV>/REPORT.md`, xoá các dòng `>` hướng dẫn,
> rồi điền số từ ô **"BẢNG SỐ LIỆU CHO REPORT.md"** (ô cuối của `lab.ipynb`) và các bảng in ra ở
> Part 3. Phần **cơ chế** đã viết sẵn — chỉ cần chèn số, `exp_id` và đường dẫn ảnh.
>
> Mọi dòng `SỐ CẦN ĐIỀN: ...` phải được thay bằng số đo thật của bạn.
> Mọi kết luận "A hơn B" phải kèm so sánh với **2σ = ___** (mục 2).

---

# Báo cáo Lab Day 1 — <Họ tên> — <MSSV>

## 1. Thiết lập

- **Môi trường:** Kaggle, GPU <T4 x2>, PyTorch <2.x.y>, CUDA <...>.
- **Dữ liệu:** Forest CoverType. `train` 464 809 / `eval` 116 203 theo `data/split_metadata.csv`.
  Validation: 20% của train, phân tầng theo nhãn, seed 42 → **371 847 train / 92 962 val**.
- **Model:** `M-base` = 54 → 256 → 128 → 7, **47 879 tham số**, ReLU, dropout chỉ sau ReLU lớp ẩn,
  lớp cuối trả **logit thô** (softmax nằm trong loss).
- **Baseline:** cross-entropy · SGD + momentum 0,9 · `lr` = ___ (chọn bằng val ở mục 2) ·
  batch 512 · 20 epoch · khởi tạo He · không clip · FP32.
- **Mốc tham chiếu:** chiến lược "luôn đoán lớp đa số" (lớp 1) cho val accuracy = **0,4876**.
- **7 chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping
  ☑ mixed precision ☑ khởi tạo

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả | Nhận xét |
|---|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) | khớp `EXPECTED_PARAMS`; `assert` trong `run_experiment` |
| Loss bước 0 (ln 7 = 1,9459) | ___ | xem giải thích bên dưới |
| Quá khớp 20 mẫu: loss cuối | ___ (sau ___ bước) | về gần 0 ⇒ loss/optimizer/model đúng |
| Gradient khác 0 ở mọi tham số | ☐ có (6/6 nhóm) | nếu không thì lỗi `zero_grad` hoặc tham số không vào optimizer |
| Baseline, số seed | 3 (seed 1, 2, 3) | |
| Baseline: val accuracy | ___ ± ___ | |
| Baseline: val macro-F1 | ___ ± ___ | |

**Ngưỡng nhiễu dùng trong báo cáo: 2σ = ___ (val macro-F1).**
Chênh lệch nhỏ hơn 2σ được coi là **chưa kết luận được**.

**Về loss bước 0.** Với khởi tạo He áp cho *mọi* lớp kể cả lớp cuối, loss bước 0 đo được là **___**,
lệch **+___** so với `ln 7`. Điều này **không phải lỗi**: CE bằng `ln 7` chỉ khi logit chia đều 7 lớp;
He cho logit có phương sai lớn hơn nên loss cao hơn một chút. Bảng ở Part 1 cho thấy cùng kiến trúc
nhưng khởi tạo `default` cho loss bước 0 = ___ (gần `ln 7` vì logit nhỏ), `zeros` cho đúng 1,9459
(logit bằng 0 ⇒ phân bố đều). Vì vậy tiêu chí đúng là **loss bước 0 ổn định và không phụ thuộc mẫu
đầu tiên**, chứ không phải "bằng đúng ln 7".

## 3. Kết quả theo chủ đề

> Mỗi mục: (a) dự đoán trước, (b) kết quả + `exp_id` + ảnh, (c) cơ chế, (d) có vượt 2σ không.
> Toàn bộ dựa trên **val**; tập eval chỉ dùng ở mục 4.

### 3.1 Hàm mất mát — CE vs MSE

- **Dự đoán:** MSE trên nhãn one-hot cho macro-F1 **thấp hơn** CE ở cùng `lr`.
- **Kết quả:** `loss-mse` (MSE) đạt val macro-F1 = **___**, thấp hơn baseline `base-s1` = **___**
  một lượng **___**, tức **vượt / không vượt** 2σ = ___. Ảnh: `figures/compare_loss.png`,
  `figures/loss-mse.png`.
- **Cơ chế:** gradient của CE theo logit là `p − y` (với `p = softmax(z)`): sai ở đâu phạt nặng
  đúng chỗ đó, và **không bão hoà** — tín hiệu lớp hiếm vẫn đủ lớn để đẩy mô hình học.
  Gradient của MSE là `2/N · (z − y) · σ(z)`: bị nhân với đạo hàm sigmoid nên khi `|z|` lớn thì
  `σ' → 0`, **bão hoà**, mô hình ngừng học dù còn sai; thêm nữa tín hiệu được **chia trung bình trên
  cả 7 thành phần one-hot** nên lớp chỉ 0,5% bị loãng gần như tan.
- **Lưu ý:** loss CE và MSE **khác thang đo** (CE ≈ 1–2, MSE ≈ 0,0x vì `nn.MSELoss` không có hệ số
  1/2 và trung bình trên mọi phần tử). Chỉ so **accuracy / macro-F1 / tốc độ hội tụ**, không so loss.

### 3.2 Bộ tối ưu hoá

- **Dự đoán:** mỗi bộ được chỉnh `lr` riêng thì **Adam/AdamW thắng** ở ngân sách 20 epoch;
  **SGD nhạy `lr` nhất**; Adam và AdamW phải cho **cùng** kết quả khi `weight_decay = 0`.

Bảng nhỏ — mỗi bộ ở `lr` tốt nhất của chính nó:

| Bộ tối ưu | `lr` tốt nhất | `exp_id` | val macro-F1 | val acc | best epoch |
|---|---|---|---|---|---|
| SGD | ___ | `opt-sgd-lr___` | ___ | ___ | ___ |
| SGD + momentum 0,9 | ___ | `opt-sgdm-lr___` | ___ | ___ | ___ |
| Adam | ___ | `opt-adam-lr___` | ___ | ___ | ___ |
| AdamW (wd = 0,01) | ___ | `opt-adamw-lr___` | ___ | ___ | ___ |

- **Độ nhạy với `lr`:** biên độ val macro-F1 giữa `lr` tệ nhất và tốt nhất — SGD: **___**,
  SGD+mom: **___**, Adam: **___**, AdamW: **___**. Bộ ổn định nhất: **___**.
  Ảnh chồng: `figures/compare_opt_*.png`, `figures/compare_optimizer.png`.
- **Kiểm chứng Adam ≡ AdamW khi wd = 0:** `opt-adamw-wd0-lr0.001` cho val macro-F1 = **___**
  so với `opt-adam-lr0.001` = **___** ⇒ khớp trong sai số số học, đúng như công thức
  (`w ← w − ηλw − η·m̂/(√v̂+ε)` với `λ = 0` rút về đúng công thức Adam).
- **Cơ chế:** SGD dùng **phương sai** của gradient nên cần `lr` nhỏ và phụ thuộc mạnh vào tỉ lệ giữa
  các lớp → nhạy `lr`, hội tụ chậm. Momentum giảm dao động ngang nên chịu `lr` lớn hơn.
  Adam/AdamW chuẩn hoá **theo phương sai giới hạn tích cực** (`m̂/(√v̂+ε)`), nên mọi tham số đều có
  bước đi gần như cùng thang ⇒ nhạy `lr` thấp hơn, hội tụ nhanh ở ngân sách epoch ngắn.
  Ở đây best epoch của các bộ là **___** ⇒ (converged / chưa hội tụ; nếu chưa thì lợi thế
  "hội tụ nhanh" của Adam là lý do thắng, và khoảng cách sẽ thu hẹp khi huấn luyện lâu hơn).

### 3.3 Hyper-parameter

| Thay đổi | `exp_id` | val macro-F1 | Δ so với baseline | vượt 2σ? | ghi chú |
|---|---|---|---|---|---|
| batch 128 | `hparam-batch128` | ___ | ___ | ___ | ___ bước/epoch |
| batch 2048 | `hparam-batch2048` | ___ | ___ | ___ | ___ bước/epoch |
| `weight_decay` 1e-4 | `hparam-wd1e-4` | ___ | ___ | ___ | |
| cosine annealing | `hparam-cosine` | ___ | ___ | ___ | |
| `M-wide` (512-256) | `arch-wide` | ___ | ___ | ___ | 161 287 tham số |
| `M-deep` (256-128-64) | `arch-deep` | ___ | ___ | ___ | 55 687 tham số |
| ___ epoch | `hparam-epoch___` | ___ | ___ | ___ | |

- **Số bước cập nhật:** 371 847 mẫu / batch 512 = **726** bước/epoch; batch 128 → **2 904**;
  batch 2048 → **182**. Cùng 20 epoch nhưng khác số bước 16 lần, nên phải đọc cùng `time_per_epoch_s`.
- **Cơ chế:** batch lớn ⇒ gradient ít nhiễu nhưng **ít bước cập nhật hơn**, ở ngân sách epoch cố định
  nên chậm hơn; batch nhỏ ⇒ nhiều bước hơn nhưng nhiễu lớn hơn và tốn thời gian/epoch.
  Ta **giữ nguyên `lr`** để cô lập đúng một yếu tố; nếu muốn tối ưu batch 2048 thì phải tăng `lr`
  theo quy tắc lô ×k thì η ×k (có khởi động).
- **Kiến trúc:** `M-wide` thêm năng lực → ___ ; `M-deep` thêm ít tham số → ___ .
- **Số epoch:** best epoch của `hparam-epoch___` là **___** (còn là epoch cuối không?) và
  `val_loss − train_loss` = **___** ⇒ mô hình **đã / chưa** hội tụ. Đây là căn cứ để chọn số epoch
  cho cấu hình cuối cùng.

### 3.4 Dropout

| `q` | `exp_id` | val macro-F1 | Δ | vượt 2σ? | `val_loss − train_loss` |
|---|---|---|---|---|---|
| 0 (baseline) | `base-s1` | ___ | — | — | ___ |
| 0,1 | `drop-0.1` | ___ | ___ | ___ | ___ |
| 0,2 | `drop-0.2` | ___ | ___ | ___ | ___ |
| 0,5 | `drop-0.5` | ___ | ___ | ___ | ___ |

Ảnh: `figures/compare_dropout.png`.

- **Cơ chế:** dropout là **thuốc cho quá khớp**. Ở đây khoảng cách `val_loss − train_loss` của
  baseline chỉ là **___** ⇒ mô hình **chưa quá khớp**, nên dropout không có "bệnh" để chữa và chỉ
  làm loãng tín hiệu ⇒ giảm macro-F1. Khi nào nên dùng: khi `val_loss` bắt đầu **tăng** trong khi
  `train_loss` vẫn giảm (dấu hiệu quá khớp), hoặc khi tăng sức mạnh mô hình (rộng/sâu hơn,
  epoch nhiều hơn).
- **Lưu ý đo lường:** `train_loss` phải đo ở chế độ `eval()` (tắt dropout) thì mới so được với
  `val_loss`. Nếu đo trong lúc huấn luyện, dropout "làm tăng train loss" một cách giả.

### 3.5 Gradient clipping

- **`grad_norm` của baseline (đo trước khi clip):** min **___**, trung vị theo epoch **___**,
  max **___**. Chọn `c` = **___**.
  Ảnh: `figures/base-s1.png` (ô 3), `figures/compare_clipping.png` (thang log).

| Cấu hình | `exp_id` | val macro-F1 | Δ | vượt 2σ? | `diverged` |
|---|---|---|---|---|---|
| không clip (baseline) | `base-s1` | ___ | — | — | ___ |
| clip c = ___ ở `lr` thường | `clip-1.0` | ___ | ___ | ___ | ___ |
| `lr` = ___ , **không** clip | `clip-hilr-noclip` | ___ | ___ | ___ | ___ |
| `lr` = ___ , **có** clip | `clip-hilr-clip1.0` | ___ | ___ | ___ | ___ |

- **Ở `lr` thường:** clipping kích hoạt ở **___%** các epoch (epoch có `grad_norm_min < c < grad_norm_max`).
  Chênh lệch so với baseline là **___** ⇒ **vượt / không vượt** 2σ. Kết luận phải là
  *"clipping không giúp ở `lr` này vì gradient vốn đã nằm dưới ngưỡng"*, **không phải**
  *"clipping vô dụng"*.
- **Ở `lr` cao (thí nghiệm phản chứng):** không clip thì val macro-F1 = **___**
  (lệch **___** so với baseline), có clip thì = **___** (cải thiện **___** so với ca không clip).
  ⇒ clipping **cứu được** huấn luyện ở `lr` cao.
- **Cơ chế:** tăng `lr` làm gradient nổ gai (`grad_norm` tăng vọt, thấy rõ ở thang log), SGD
  bước đi quá xa và dao động/NaN. Cắt gradient áp `g ← g · min(1, c/‖g‖)` giới hạn **độ dài bước
  cập nhật** theo chuẩn L2 toàn cục nên bước đi không bao giờ vượt `c·lr`; đây chính là tình huống
  clipping được thiết kế để xử lý.

### 3.6 Mixed precision

| precision | `exp_id` | s/epoch | so với FP32 | peak mem (MB) | so với FP32 | val macro-F1 | Δ |
|---|---|---|---|---|---|---|---|
| fp32 | `base-s1` | ___ | 1,00× | ___ | 1,00× | ___ | — |
| fp16 | `amp-fp16` | ___ | ___× | ___ | ___× | ___ | ___ |
| bf16 | `amp-bf16` | ___ | ___× | ___ | ___× | ___ | ___ |

Ảnh: `figures/compare_amp.png`. GPU có phần cứng BF16: **___**.

- **Kết luận:** mixed precision **không nhanh hơn** (nếu đúng) vì mạng này chỉ có 47 879 tham số —
  mỗi phép toán rất nhỏ nên thời gian bị **chi phối bởi chi phí gọi kernel và chuyển đổi dtype**,
  không phải FLOPs. Chỉ khi mô hình đủ lớn (vài trăm triệu tham số, GEMM dài) thì giảm độ rộng
  số và tăng băng thông mới thắng chi phí này. Chất lượng thay đổi **___** (trong 2σ?) nên coi là
  không đổi có ý nghĩa thực tế.
- **Cơ chế FP16 vs BF16:** FP16 có **11 bit mantissa** (~3–4 chữ số thập phân), giá trị nhỏ hơn
  ~6·10⁻⁵ chuyển thành 0 và lớn hơn 65 504 thành `inf` ⇒ gradient nhỏ bị **underflow** thành 0 và
  loss lớn bị tràn ⇒ phải nhân loss với hệ số scale (`GradScaler`) rồi bỏ scale trước khi cắt gradient
  và trước `optimizer.step()`. BF16 có **8 bit mantissa** nhưng **8 bit exponent** giống FP32 ⇒ dải
  số động rộng như FP32, nhỏ dưới 1e-38 vẫn bình thường ⇒ **không cần** `GradScaler`.

### 3.7 Khởi tạo tham số

| init | `std` kích hoạt tầng 1 | tầng 2 | logits | loss bước 0 | `exp_id` | val macro-F1 |
|---|---|---|---|---|---|---|
| he (baseline) | ___ | ___ | ___ | ___ | `base-s1` | ___ |
| xavier | ___ | ___ | ___ | ___ | `init-xavier` | ___ |
| default (`nn.Linear`) | ___ | ___ | ___ | ___ | `init-default` | ___ |
| normal (0,01) | ___ | ___ | ___ | ___ | `init-normal` | ___ |
| zeros | 0 | 0 | 0 | 1,9459 | `init-zeros` | ___ |

Ảnh: `figures/compare_init.png`, `figures/init-*.png`.

- **`zeros` hỏng vì sao.** Với `W = 0`, tất cả nơ-ron trong **cùng một tầng** nhận **cùng một**
  gradient (vì đầu vào và gradient phía sau giống hệt nhau), nên sau bước cập nhật chúng vẫn có
  trọng số **bằng nhau** → mạng bị **đối xứng** và hàm hoàn toàn tương đương **một nơ-ron duy nhất**,
  không có nhiều năng lực học. Thêm nữa `ReLU(0) = 0` nên gradient truyền ngược qua tầng ẩn bị
  chặn. Kết quả đo được: val macro-F1 = **___** ≈ mức "đoán đa số" (0,0936), val accuracy = **___**
  ≈ 0,4876 — mô hình **không học được** dù loss bước 0 = 1,9459 (đẹp!). Đây là ví dụ rõ nhất:
  **loss bước 0 đúng không bảo đảm mô hình học được** → phải kiểm tra cả gradient.
- **He khác Xavier ở đâu và khi nào quan trọng.** Cả hai đều giữ phương sai qua tầng nhưng tính theo
  hai công thức khác nhau: Xavier dùng `2/(n_in + n_out)` (xét cả chiều ra, giả định tầng có cả
  phương lớn lẫn nhỏ đều đóng góp), còn He dùng `2/n_in`. Với ReLU — vốn **giữ lại đúng một nửa**
  nơ-ron — He bù đúng phần bị tắt đó ⇒ phương sai kích hoạt **giữ gần như không đổi** qua các tầng
  (đo được: **___** → **___**), trong khi Xavier làm phương sai **giảm dần** (**___** → **___**) và
  `nn.Linear` mặc định (`Var = 1/(3·n_in)`) giảm mạnh hơn nữa (**___** → **___**). Vì vậy:
  **ReLU → He**, **tanh/sigmoid → Xavier**. Với mạng 2 tầp ẩn như lab này, sự suy giảm qua tầng còn
  nhỏ nên `he` và `xavier` chưa chênh lệch nhiều trên val macro-F1 (chênh **___**, có vượt 2σ không?);
  muốn thấy hiện tượng rõ như slide ("30 lớp ReLU") thì cần mạng sâu hơn nhiều.

### 3.8 Cấu hình cuối cùng (chọn bằng val)

Cấu hình cuối = **___** (ghi rõ từng thay đổi và `exp_id` làm bằng chứng).
Ba thay đổi so với baseline: optimizer + `lr` (mục 3.2, `exp_id` ___), dropout `q` = **___**
(mục 3.4, `exp_id` ___), số epoch = **___** (mục 3.3, `exp_id` ___).

Qua 3 seed: val macro-F1 = **___ ± ___**, val accuracy = **___ ± ___**.
So với baseline: Δ = **___**, 2σ = **___** ⇒ **VƯỢT / CHƯA VƯỢT** nhiễu.

---

## 4. Đánh giá cuối trên tập eval

> Cấu hình đã chọn **bằng val** ở mục 3.8 trước khi chạm vào eval. `scripts/evaluate.py` chạy
> **một lần cho baseline và một lần cho cấu hình cuối cùng**; số dưới đây lấy nguyên từ
> `eval_result.json`.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline | 1 | ___ | **___** | ___ |
| Cấu hình cuối cùng | 1 | ___ | **___** | ___ |

- **Cải thiện trên eval** = **___** (từ ___ lên ___). Ngưỡng nhiễu 2σ (đo trên val) = **___**.
  Kết luận: **VƯỢT / CHƯA VƯỢT** nhiễu seed. *(Nếu muốn có σ trên chính eval thì phải chạy cấu hình
  cuối nhiều seed và báo cáo trung bình ± σ; file nộp là dự đoán của **một** mô hình, seed ___.)*
- **Val và eval có gần nhau không?** Chênh lệch = **___**. Nếu lệch lớn thì do: val nhỏ hơn nên
  thống kê nhiễu hơn, hoặc `best_epoch` được chọn trên val nên val hơi lạc quan.

### 4.1 Phân tích lỗi theo lớp

*(chép nguyên văn từ `eval_result.json`, bảng được notebook in ra)*

| Lớp | Loại rừng | support | precision | recall | F1 |
|---|---|---|---|---|---|
| 0 | Spruce/Fir | | | | |
| 1 | Lodgepole Pine | | | | |
| 2 | Ponderosa Pine | | | | |
| 3 | Cottonwood/Willow | | | | |
| 4 | Aspen | | | | |
| 5 | Douglas-fir | | | | |
| 6 | Krummholz | | | | |

Ma trận nhầm lẫn: xem `figures/error_analysis_eval.png`.

- **Lớp khó nhất:** lớp **___** (F1 = **___**), có support = **___** mẫu.
  Nó bị nhầm nhiều nhất với lớp **___** (**___** mẫu, chiếm **___%** số mẫu của lớp ___).
- **Lý giải:**
  1. **Mất cân bằng:** lớp hiếm có ít mẫu (ví dụ lớp 3 chỉ ~0,5% ≈ **___** mẫu trong eval) nên
     recall thấp dù precision có thể cao — mô hình ở ranh giới nghiêng về lớp lớn.
  2. **Đặc trưng gần nhau:** hai lớp dùng chung dải cao độ/độ dốc và cùng nhóm đất (ví dụ
     Cottonwood/Willow và Aspen đều sống ở thấp, ven sông — `Elevation` thấp và
     `Horizontal_Distance_To_Hydrology` nhỏ) nên phân biệt chủ yếu bằng vài mẫu đất hiếm.
     Chiều nhầm ngược (lớp A bị dự đoán thành lớp B **và** ngược lại) là bằng chứng cho điều này:
     hàng ___ cột ___ = **___**, hàng ___ cột ___ = **___**.
  3. **Cân bằng giữa precision/recall:** lớp có recall cao nhưng precision thấp là lớp bị "nuốt"
     vào lớp khác (dự đoán quá rộng); ngược lại là lớp bị dự đoán quá thườn (over-predict).
- **Một cách cải thiện tôi sẽ thử:** (chọn 1 và giải thích vì sao đúng với quan sát trên)
  - ví dụ: `class_weight` hoặc loss có trọng số nghịch đảo tần suất lớp trong cross-entropy
    (`F.cross_entropy(..., weight=1/n_c)`) → nâng recall lớp hiếm, phù hợp khi vấn đề là **thiếu mẫu**;
  - hoặc oversampling lớp hiếm trong tập train;
  - hoặc thêm đặc trưng phụ (ví dụ tỉ lệ `Vertical/Horizontal_Distance_To_Hydrology`) nếu vấn đề là
    **đặc trưng không phân biệt được**.

---

## 5. Trả lời các câu hỏi dẫn dắt

**1. Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh `lr` công bằng? Nếu `lr` không được chỉnh thì kết luận đổi ra sao?**
Sau khi mỗi bộ được thử 3 `lr` và so ở `lr` tốt nhất của nó, thắng là **___** (val macro-F1 = ___).
Nếu **không** chỉnh `lr` mà dùng cùng một giá trị cho mọi bộ, kết luận sẽ là **___**, vì thang
"bước đi" của SGD (tuyến tính theo `η`) và Adam (chuẩn hoá theo `√v̂`) hoàn toàn khác nhau:
cùng `lr` cho SGD là hợp lý thì với Adam là quá nhỏ và ngược lại. Bằng chứng trong bảng: cùng
`lr` = ___ cho SGD và Adam cho val macro-F1 lần lượt ___ và ___. Đây là lý do rubric yêu cầu
**≥ 2 `lr` mỗi bộ rồi so ở `lr` tốt nhất** — so ở một `lr` chung là so vũ khí chứ không phải so sức mạnh.

**2. Dropout có giúp khi mô hình chưa quá khớp? Khi nào nên dùng?**
Xem mục 3.4. Ở cấu hình baseline, `val_loss − train_loss` chỉ = **___** ⇒ chưa quá khớp, và
dropout `q` = 0,1/0,2/0,5 cho Δ lần lượt **___** ⇒ phần lớn **không vượt 2σ**. Cơ chế: dropout
đánh đổi nhiễu huấn luyện lấy hiệu quả tổng quát hoá, nên khi vốn đã không quá khớp thì cái giá
(học chậm hơn, nhiễu thêm) không được đền bù. Nên dùng khi `val_loss` **tăng** trong khi `train_loss`
vẫn giảm, hoặc khi tăng sức mạnh mô hình (rộng/sâu/epoch nhiều). Trong bài này `q` được chọn cho
cấu hình cuối là **___** dựa trên val.

**3. Gradient clipping giải quyết vấn đề gì? Quan sát nào chứng minh?**
Xem mục 3.5. Nó giải quyết **bước cập nhật bị gradient khổng lồ làm vỡ huấn luyện**: giới hạn
`‖g‖` toàn cục ở `c`. Bằng chứng: ở `lr` cao không clip thì val macro-F1 = **___** (thấp hơn baseline
___; `diverged` = ___), còn **cùng `lr` đó** có clip thì = **___**; và đường `grad_norm` (thang log)
cho thấy các gai ở cao không clip nhưng bị bóp phẳng khi có clip.

**4. Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
Xem mục 3.6. Câu trả lời đo được: **___**. Lý do: mạng chỉ 48k tham số, mỗi phép toán rất nhỏ nên
thời gian bị chi phối bởi **số lần gọi kernel + chuyển đổi dtype** chứ không phải FLOPs; lợi ích của
FP16/BF16 đến từ việc giảm **băng thông bộ nhớ** và tăng **mật độ phép toán** chỉ khi mô hình đủ lớn.
Bộ nhớ giảm **___%**, chất lượng thay đổi **___**.

**5. Vì sao khởi tạo toàn số 0 hỏng? He khác Xavier ở đâu, khi nào điều đó quan trọng?**
Xem mục 3.7. Tóm tắt: `zeros` ⇒ mất đối xứng + `ReLU(0)=0` chặn gradient ⇒ mô hình tương đương
một nơ-ron, macro-F1 = **___** ≈ đoán đa số. He (`Var = 2/n_in`) bù đúng việc ReLU tắt một nửa nơ-ron
nên giữ phương sai kích hoạt không đổi qua tầng; Xavier (`Var = 2/(n_in+n_out)`) tính cả chiều ra
nên làm phương sai giảm dần. He là lựa chọn đúng khi **kích hoạt không đối xứng và có thể tắt**
(ReLU, GELU); Xavier hợp với kích hoạt đối xứng giữ zero-mean (tanh, sigmoid).

**6. Quay lại câu hỏi của bài học.** Một mạng có loss không giảm sau 2 000 bước. Ba phép kiểm tra
đầu tiên tôi sẽ làm và vì sao:

1. **Đo loss bước 0 và kiểm tra nó có hợp lý không.** Nếu loss bằng phẳng ngay từ đầu ở khoảng
   `ln C` thì mô hình chưa học được: nghĩ ngay tới nhãn sai (quên đổi 1..7 → 0..6), softmax bị
   đặt hai lần (trong model và trong loss), hoặc **gradient không chảy**. Rẻ, mất vài giây, và loại
   được ngay nhóm lỗi phổ biến nhất. Trong bài này: `init-zeros` có loss bước 0 = 1,9459 — *đẹp
   đúng `ln 7`* — nhưng sau 2 000 bước loss vẫn 1,2052. **Loss bước 0 đúng không đảm bảo học được**,
   nên chỉ làm phép kiểm tra này thì chưa đủ.
2. **Quá khớp một lô nhỏ (20 mẫu, mọi thành phần tắt) trong vài trăm bước.** Nếu loss về gần 0 thì
   model, loss, optimizer và vòng lặp đều đúng ⇒ vấn đề nằm ở **dữ liệu / tỉ lệ học / kiến trúc
   quá nhỏ**. Nếu không về gần 0 thì gần như chắc chắn là **lỗi code**: quên `zero_grad` khiến
   gradient cộng dồn, tham số không được đưa vào optimizer, nhãn lệch, hoặc quên chuyển mô hình về
   `eval()` khi đo. Ở đây: 20 mẫu → loss **___** sau 400 bước.
3. **In chuẩn gradient của từng tham số sau một lần `backward()`.** Nếu có tham số `grad is None`
   thì nó không nằm trong optimizer (hoặc bị `requires_grad=False`); nếu chuẩn bằng 0 thì đường đi của
   tín hiệu bị chặn (ReLU chết, `zeros` khởi tạo, gradient bị nhân bởi `σ'(z) ≈ 0`). Kèm theo đó là
   nhìn `grad_norm` theo thời gian: **phẳng** ⇒ gradient không truyền hoặc quá nhỏ (lr quá thấp);
   **bùng nổ / có gai** ⇒ lr quá cao, cần giảm lr hoặc bật gradient clipping.

Nếu cả ba đều bình thường mà loss vẫn phẳng, bước tiếp theo là dữ liệu: kiểm tra phân bố nhãn,
kiểm tra chuẩn hoá (thiếu chuẩn hoá làm lớp cuối nhận đầu vào phân bố lệch ⇒ mất tín hiệu), và
tăng sức mạnh mô hình / số epoch.

---

## 6. Hạn chế và điều bất ngờ

- **Kết quả khác dự đoán:**
  - `<Ví dụ: mixed precision không nhanh hơn vì ...>` — đây là kết quả **hợp lệ**, không phải lỗi
    đo; chỉ "nhanh hơn" mới cần số đo chứng minh.
  - `<Ví dụ: clipping ở lr thường không tạo khác biệt nào>` — đúng như dự đoán, và cho thấy cần
    thí nghiệm phản chứng ở `lr` cao mới chứng minh được clipping có ích.
  - `<Ví dụ: dropout q=0.1 ...>`
- **Điều gì trong thiết kế thí nghiệm có thể làm kết luận sai:**
  - 3 seed cho σ → σ ước lượng thô (tương đối chính xác cỡ 30% khi có 3 mẫu); ngưỡng 2σ vì thế
    **có xu hướng rộng**, khiến ta dễ kết luận "không khác" — đây là hướng bảo thủ có chủ ý.
  - Cùng số epoch nhưng khác số bước cập nhật khi đổi batch → không so sánh thuần túy về hiệu quả học.
  - `lr` chỉ quét 3 giá trị lệch nhau ~3–10×; giá trị tối ưu có thể nằm giữa hai mũi.
  - Chọn cấu hình cuối bằng `val macro-F1` ở `best_val_loss`, còn điểm chấm là `eval macro-F1`;
    hai tiêu chí gần nhau nhưng không giống hệt.
  - Ở chủ đề clipping, `lr` cao được chọn bằng `10× lr` baseline — con số này **không** được quét,
    nên chỉ chứng minh clipping cứu được ở *đúng* `lr` đó.
- **Nếu có thêm thời gian:** (chọn 2)
  - quét `lr` mịn hơn quanh giá trị tốt nhất cho từng bộ, và chạy 5 seed thay vì 3 để σ chắc hơn;
  - thử `class_weight` / oversampling cho lớp hiếm rồi đo lại macro-F1 theo lớp (mục 4.1);
  - thử kiến trúc sâu hơn nhiều để khởi tạo `he` vs `xavier` cho khác biệt rõ (slide Chương 4);
  - thử scheduler cosine dài hơn + `weight_decay` cho AdamW trên cấu hình cuối.

## 7. Phụ lục

- **Danh sách file đã nộp:** `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`,
  `eval_result.json`, `figures/` (__ ảnh `<exp_id>.png` + __ ảnh `compare_*.png`),
  `results/` (__ file `.json`), `code/` (`lab.ipynb` + 6 module `.py` + `requirements.txt`).
- **Môi trường:** PyTorch __, GPU __, Python __.
- **Thời gian chạy ước tính:** tổng ___ phút cho __ lần chạy
  (trung bình ___ giây/epoch trên 371 847 mẫu, batch 512).
- **Tính tái lập:** seed cố định (`set_seed` cho `random`/`numpy`/`torch`/`cuda`); `run_experiment`
  có kiểm tra cấu hình trùng với `results/<exp_id>.json` nên chạy lại cho kết quả nhất quán.