# Hướng dẫn chạy lab Day 1 trên Kaggle (GPU)

Bài lab gồm **39 lần chạy mô hình** (~40–60 phút GPU). Notebook tự ghi `results/<exp_id>.json`
nên nếu bị ngắt kết nối thì chạy lại sẽ **bỏ qua** thí nghiệm đã có kết quả.

---

## 1. Chuẩn bị (một lần)

### 1a. Đưa repo lên Kaggle

**Cách A — Kaggle Dataset (không cần Internet, khuyến nghị)**

1. Trên máy: `git clone` / copy thư mục repo, nén thành `repo.zip`.
   Lưu ý: `data/processed/` và `__pycache__/` không cần đưa lên (đã có trong `.gitignore`).
2. Kaggle → **Datasets** → **New Dataset** → upload `repo.zip` → tạo.
3. Notebook dùng lệnh:

```python
!cp -r "/kaggle/input/<tên-dataset>/repo" /kaggle/working/repo
REPO_ROOT = "/kaggle/working/repo"
```

**Cách B — git clone (cần bật Internet)**

Settings → Internet → On, rồi:

```python
!git clone https://github.com/pvksssss/K4-Track4-Day1-Neuralnetwork.git /kaggle/working/repo
REPO_ROOT = "/kaggle/working/repo"
```

### 1b. Đưa `code/lab.ipynb` lên Kaggle

Tạo notebook mới trong Kaggle, xoá các ô mặc định, rồi **kéo thả `code/lab.ipynb`** vào
(hoặc dán nội dung các ô vào). Notebook phải nằm ở `/kaggle/working/` — **không** đặt trong
`/kaggle/working/repo/code/`, nếu không `code/` sẽ bị ghi đè lúc `!cp`/`!git clone`.

Nếu đặt nhầm trong `repo/code/` thì sửa dòng này ở ô setup cho khớp:

```python
CWD = "/kaggle/working"          # nơi chứa lab.ipynb
```

---

## 2. Cấu hình nên dùng trên Kaggle

| Thiết lập | Giá trị |
|---|---|
| Accelerator | **GPU T4 x2** (chỉ dùng 1 GPU là đủ) |
| Internet | Off nếu dùng Cách A |
| Session | ≥ 12 h (không cần, ~1 h là xong) |

Kiểm tra `openpyxl` đã có sẵn trên Kaggle. Nếu không:

```python
!pip install -q openpyxl
```

### 2b. Kiểm tra nhanh sau khi mở notebook

Ô đầu tiên phải in ra đúng 3 dòng:

```
REPO_ROOT = /kaggle/working/repo
CODE_DIR  = /kaggle/working/repo/code
OUT_DIR   = /kaggle/working
```

Nếu `REPO_ROOT` hoặc `CODE_DIR` sai, sửa lại hai dòng định nghĩa ở ô đó trước khi chạy tiếp.

---

## 3. Chạy

1. **Restart & Run All** — hoặc chạy lần lần từ ô đầu.
2. **Kiểm tra nhanh trước** (khuyến nghị lần đầu): đặt `FAST = True` ở ô setup
   (hoặc `os.environ["LAB_FAST"]="1"`), chạy hết để xác nhận không có lỗi — mất ~5 phút.
3. Xoá thư mục kết quả của lần chạy nhanh rồi chạy thật (`FAST = False`):

```python
!rm -rf /kaggle/working/submission /kaggle/working/_nbtest
```

> `SKIP_DONE = True` có trong ô setup: nếu `results/<exp_id>.json` đã tồn tại **và cấu hình
> không đổi**, thí nghiệm đó được bỏ qua. Muốn chạy lại từ đầu thì `SKIP_DONE = False`.

---

## 4. Lấy kết quả về máy

Ô cuối của notebook (**Part 5 — Đóng gói thư mục nộp**) tự dựng thư mục
`/kaggle/working/repo/<LAB_SUBMISSION>/` đúng cấu trúc README mục 6.1
(`figures/`, `results/`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `code/`).

1. Đặt tên thư mục: sửa `SUBMISSION_NAME` trong ô đó thành `submission_<MSSV>`, rồi chạy lại ô.
2. Nén lại:

```python
!cd /kaggle/working/repo && zip -r submission.zip submission_<MSSV> \
    -x "*/__pycache__/*" "*__pycache__*" "*.ipynb_checkpoints*"
```

3. Bấm **Save Version** (Commit & Run) hoặc tải `submission.zip` từ mục Output.

### 4b. Viết `REPORT.md` (20/100 điểm — phần này tốn thời gian nhất)

Repo có sẵn **`REPORT_DRAFT.md`**: phần cơ chế cho cả 7 chủ đề và 6 câu hỏi dẫn dắt đã viết sẵn,
còn số đo để trống (`___`). Việc cần làm:

1. `cp REPORT_DRAFT.md /kaggle/working/repo/submission_<MSSV>/REPORT.md`
   (rồi chạy lại ô Part 5 để nó được đóng gói, hoặc chép tay vào thư mục nộp)
2. Mở ô **"BẢNG SỐ LIỆU CHO REPORT.md"** — in ra toàn bộ số cần dùng cùng bảng 39 dòng thí nghiệm.
3. Điền số và `exp_id` vào báo cáo; chèn ảnh bằng đường dẫn tương đối: `![](figures/compare_optimizer.png)`.
4. Xoá các dòng hướng dẫn `>` và các mục không dùng.

> Sai lầm bị trừ điểm nhiều nhất: kết luận "A tốt hơn B" mà không kèm số, không so với 2σ, không
> giải thích cơ chế. Mỗi kết luận cần **3 thứ**: con số + `exp_id` + cơ chế.

---

## 5. Nộp (nếu giảng viên yêu cầu bộ file nộp ở local)

Trên máy:

```bash
cd <repo>
mkdir -p submission_<MSSV>
cp -r code submission_<MSSV>/code          # chỉ chép .py + lab.ipynb + requirements.txt
# rồi chép từ zip Kaggle: REPORT.md, experiments.xlsx, predictions_eval.csv,
#                          eval_result.json, figures/, results/
# đặt tên thư mục trong .ipynb là submission_<MSSV> (đổi SUBMISSION_NAME nếu cần)
python scripts/split_data.py               # nếu máy chưa có data/processed
python scripts/evaluate.py --pred submission_<MSSV>/predictions_eval.csv \
                           --out submission_<MSSV>/eval_result.json
```

---

## 6. Checklist trước khi nộp

- [ ] `experiments.xlsx` mở được, **mở bằng Excel/LibreOffice một lần** để các cột công thức
      (`step0_gap_vs_lnC`, `gap_val_minus_train`, `delta_val_f1_vs_base`, `beyond_noise`) tính lại.
- [ ] Số file `figures/<exp_id>.png` (không tính `compare_*`) **bằng** số dòng trong sheet `Experiments`.
- [ ] `predictions_eval.csv` đủ 116 203 dòng, nhãn 0..6, qua được `scripts/evaluate.py`.
- [ ] Số trong `REPORT.md` khớp `eval_result.json` và bảng (chép từ ô
      "BẢNG SỐ LIỆU CHO REPORT.md" ở ô cuối của notebook).
- [ ] Không có `__pycache__`, `.ipynb_checkpoints`, `*.pt`, `*.npz`, `data/` trong bộ nộp.
- [ ] **Không có `NotImplementedError`** trong `code/`:

```bash
grep -rn "NotImplementedError" submission_<MSSV>/code || echo "OK: sạch"
```

---

## 7. Chạy trên máy local (không có GPU Kaggle)

```bash
pip install -r code/requirements.txt
python scripts/split_data.py
# mở code/lab.ipynb và Run All
```

Mặc định khi chạy trong `code/` của repo, kết quả ghi vào `<repo>/submission/`
(đổi bằng biến môi trường `LAB_SUBMISSION`). Nếu chạy trong `submission_<MSSV>/code/`
thì kết quả ghi đúng vào `submission_<MSSV>/`.

Trên Windows, đặt `OPENBLAS_NUM_THREADS=1` nếu gặp
`OpenBLAS error: Memory allocation still failed` khi máy thiếu RAM.

---

## 8. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `assert REPO_ROOT` fail ở ô setup | Chưa clone/copy repo. Chạy lệnh ở mục 1 rồi **chạy lại ô setup**. |
| Kết quả ghi nhầm chỗ | Ô setup in ra `REPO_ROOT` / `OUT_DIR` / `notebook` ngay đầu tiên — kiểm tra 3 dòng đó. Ép lại bằng biến môi trường `LAB_OUT_DIR=/kaggle/working` hoặc sửa trực tiếp dòng `OUT_DIR = ...`. |
| `NotFoundError: openpyxl` | `!pip install -q openpyxl` |
| `CUDA out of memory` | Đây là lỗi **GPU**, không phải RAM. Dữ liệu cả tập chỉ ~130 MB nên rất khó xảy ra; nếu gặp thì thường do bật cả 2 GPU — chọn 1 GPU. |
| Notebook bị ngắt giữa chừng | Commit & Run lại: `SKIP_DONE = True` sẽ bỏ qua các thí nghiệm đã có `results/<exp_id>.json`. |
| Muốn chạy lại tất cả từ đầu | `SKIP_DONE = False`, hoặc xoá `results/*.json`. |
| `UnicodeEncodeError` khi in tiếng Việt (Windows) | Đặt `PYTHONIOENCODING=utf-8` (ô setup đã tự làm cho subprocess). |