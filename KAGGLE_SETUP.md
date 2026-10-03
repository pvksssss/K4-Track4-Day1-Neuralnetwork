# Hướng dẫn chạy lab Day 1 trên Kaggle (GPU)

Bài lab gồm **39 lần chạy mô hình** (~40–60 phút GPU). Notebook tự ghi `results/<exp_id>.json`
nên nếu bị ngắt kết nối thì chạy lại sẽ **bỏ qua** thí nghiệm đã có kết quả.

---

## 1. Chuẩn bị (một lần)

> **Không cần clone thủ công.** Ô setup đầu tiên của `lab.ipynb` tự chạy
> `git clone --depth 1` vào `/kaggle/working/repo` nếu chưa thấy repo.
> Bắt buộc bật **Internet: On**, và **phải** bật vì lý do ở mục 1c.

### 1a. Nếu không bật được Internet

1. Nén repo thành `repo.zip` (chỉ ~18 MB vì không kèm `data/processed/`).
2. Kaggle → **Datasets** → **New Dataset** → upload `repo.zip` → tạo.
3. Sửa hai dòng đầu ô setup của `lab.ipynb`:

```python
REPO_URL = None                      # không cần clone
REPO_LOCAL = "/kaggle/working/repo"
```

rồi thêm ngay dưới đó, trước khi `REPO_ROOT = find_repo_root()`:

```python
import shutil; shutil.rmtree(REPO_LOCAL, ignore_errors=True)
shutil.copytree("/kaggle/input/<tên-dataset>", REPO_LOCAL)
```

### 1b. ⚠️ Vì sao clone phải nằm trong notebook

**"Save Version → Save & Run All" của Kaggle chạy ở session MỚI.** Mọi file bạn clone trong
session tương tác (bấm "Run All" lúc nhập liệu) **sẽ biến mất** khi bấm Save Version, và ô setup
sẽ báo:

```
AssertionError: không tìm thấy thư mục gốc repo
```

Đây không phải lỗi cấu hình — đó là hành vi bình thường của Kaggle. Notebook đã xử lý sẵn bằng
cách tự clone trong chính nó.

### 1c. Đưa `code/lab.ipynb` lên Kaggle

Tạo notebook mới trong Kaggle rồi **kéo thả `code/lab.ipynb`** vào (hoặc dán nội dung các ô vào).

Đặt notebook ở `/kaggle/working/`. **Không** đặt trong `/kaggle/working/repo/code/`, vì ô setup sẽ
clone repo và ghi đè thư mục đó.

Các module `.py` không cần chép ra `/kaggle/working`: ô setup tự thêm
`/kaggle/working/repo/code` vào `sys.path`. Nếu bạn muốn chép ra thì chạy sau khi clone:

```python
!cp /kaggle/working/repo/code/*.py /kaggle/working/
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
| `AssertionError: không tìm thấy thư mục gốc repo` | **Internet: Off**, hoặc `REPO_URL` sai. Bật Internet rồi chạy lại — xem mục 1c. |
| `FileNotFoundError: git` | Kaggle không có `git` trong PATH → dùng Cách Dataset (mục 1a). |
| GitHub chặn IP của Kaggle | Bật Internet rồi thử lại; nếu vẫn lỗi thì dùng Cách Dataset. |
| `NotFoundError: openpyxl` | `!pip install -q openpyxl` |
| Kết quả ghi nhầm chỗ | Ô setup in ra `REPO_ROOT` / `CODE_DIR` / `OUT_DIR` ngay đầu tiên — kiểm tra 3 dòng đó. Ép lại bằng `LAB_OUT_DIR`. |
| `NotFoundError: openpyxl` | `!pip install -q openpyxl` |
| `CUDA out of memory` | Đây là lỗi **GPU**, không phải RAM. Dữ liệu cả tập chỉ ~130 MB nên rất khó xảy ra; nếu gặp thì thường do bật cả 2 GPU — chọn 1 GPU. |
| Notebook bị ngắt giữa chừng | Commit & Run lại: `SKIP_DONE = True` sẽ bỏ qua các thí nghiệm đã có `results/<exp_id>.json`. |
| Muốn chạy lại tất cả từ đầu | `SKIP_DONE = False`, hoặc xoá `results/*.json`. |
| `UnicodeEncodeError` khi in tiếng Việt (Windows) | Đặt `PYTHONIOENCODING=utf-8` (ô setup đã tự làm cho subprocess). |