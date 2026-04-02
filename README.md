# HARNN — Phân Loại Văn Bản Đa Nhãn Phân Cấp Tiếng Việt

Huấn luyện và triển khai mô hình **HARNN** (Hierarchical Attention Recurrent Neural Network) để phân loại bài báo tiếng Việt theo 3 cấp độ nhãn:

```
L1 (lĩnh vực) → L2 (lĩnh vực con) → L3 (chi tiết)
```

**Kết quả trên tập test:**

| Cấp độ | Precision | Recall | F1 |
|--------|-----------|--------|----|
| L1     | 0.931     | 0.931  | 0.931 |
| L2     | 0.779     | 0.784  | 0.781 |
| L3     | 0.795     | 0.795  | 0.795 |
| **Global** | — | — | **0.836** |

---

## Cài Đặt

### 1. Tạo và kích hoạt virtual environment

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 2. Cài đặt PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Nếu dùng CPU, bỏ `--index-url`:
> ```bash
> pip install torch torchvision torchaudio
> ```

### 3. Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

## Dataset

### Nguồn dữ liệu hiện tại

- **Nguồn**: Bài báo từ VnExpress
- **Tổng số**: 23,660 bài báo
- **Chia tập**: Train=18,865 / Val=2,353 / Test=2,362
- **Nhãn**: L1=12, L2=47, L3=20
- **Vocabulary**: 69,481 từ (min count=3, min tokens=20)

### Dữ liệu của bạn

> **Dán link raw data của bạn vào đây:**
>
> Link: [________________________________________]
>
> Mô tả dữ liệu:
> - Nguồn:
> - Số lượng mẫu:
> - Số cấp độ nhãn:
> - Định dạng: (JSON / CSV / ...)

---

## Cách Chạy

### Tùy chọn A: Web Demo

1. Khởi chạy backend:
   ```bash
   python app.py
   ```
2. Mở file `demo_ui/code.html` bằng trình duyệt
3. Dán văn bản tiếng Việt và nhấn "Run Classification"

### Tùy chọn B: Jupyter Notebook

Chạy file `notebooks/main_workflow.ipynb` từ đầu đến cuối. Notebook gồm 4 phần:

| Phần | Cells | Mô tả |
|------|-------|-------|
| 1. Tiền xử lý | 1.1–1.4 | Làm sạch, tokenize, xây dựng vocab & label maps |
| 2. Huấn luyện | 2.1–2.7 | Word2Vec + huấn luyện HARNN với đánh giá |
| 3. Dự đoán | 3.1–3.3 | Load checkpoint và dự đoán trên văn bản mới |
| 4. Đánh giá | 4.1–4.3 | Confusion matrices và macro/micro metrics |

### Tùy chọn C: Gọi API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Google chi 1 ty USD de dao tao AI tai cac truong dai hoc My\"}"
```

Kết quả trả về:

```json
{
  "l1": [{"label": "Khoa hoc", "prob": 0.92}],
  "l2": [{"label": "AI", "prob": 0.87}],
  "l3": [{"label": "Machine Learning", "prob": 0.65}],
  "tokens": ["google", "chi", "ty", "usd", "dao_tao", "ai"]
}
```

---

## Cấu Trúc Dự Án

```
Multi-Label-Text-Classification/
├── app.py                          # FastAPI backend (POST /api/predict)
├── requirements.txt                # Các thư viện Python
├── README.md                       # File này
│
├── notebooks/
│   └── main_workflow.ipynb         # Pipeline đầy đủ: preprocess → train → predict → evaluate
│
├── data/
│   ├── dictionary/
│   │   ├── vietnamese-stopwords.txt
│   │   └── vietnamese-stopwords-dash.txt
│   ├── process_data/
│   │   ├── dataset.json            # Dữ liệu đã tiền xử lý
│   │   ├── vocab.json              # Ánh xạ word → index
│   │   └── label_map.json          # Ánh xạ label → index theo từng cấp
│   ├── raw_data.json               # 23,660 bài báo VnExpress gốc
│   ├── train_data.json             # Tập train (tự sinh)
│   └── test_data.json              # Tập test (tự sinh)
│
├── demo_ui/
│   ├── code.html                   # Giao diện web (TailwindCSS frontend)
│   ├── DESIGN.md                   # Tài liệu design system
│   └── screen.png                  # Ảnh chụp giao diện
│
└── output/
    ├── models/
    │   ├── checkpoints/
    │   │   └── best_model.pt       # Model checkpoint đã huấn luyện
    │   └── word2vec.model          # Word2Vec pre-trained
    ├── results/
    │   └── train_history.json      # Metrics huấn luyện theo epoch
    ├── figures/
    │   ├── learning_curve.png
    │   ├── confusion_matrix_l1_full.png
    │   ├── confusion_matrix_l2_top20_norm.png
    │   ├── confusion_matrix_l3_top25_norm.png
    │   └── basic_metrics_multiclass.png
    └── log/
```

---

## Kiến Trúc Mô Hình

**HARNN** (Hierarchical Attention Recurrent Neural Network):

```
Input tokens → Embedding (69,481 × 100) → BiGRU (256, bidirectional)
  → Level-specific Attention → LSTMCell (hierarchical memory)
  → Linear classifiers × 3 (mỗi classifier cho một cấp nhãn)
```

Thiết kế chính:
- **Word2Vec** (skip-gram, 100d) cho word embeddings
- **BiGRU** cho ngữ cảnh mức văn bản
- **Per-level attention** để nắm bắt đặc trưng riêng cho từng nhãn
- **LSTMCell** để lan truyền thông tin phân cấp giữa các cấp
- **Sigmoid** outputs cho bài toán multi-label classification

Hyperparameters:

| Tham số | Giá trị |
|---------|---------|
| Embedding dim | 100 |
| Hidden size | 256 |
| Max sequence length | 512 |
| Optimizer | Adam (lr=3e-4) |
| Epochs | 10 |
| Batch size | 64 |
| Dropout | 0.5 |

---

## Tài Liệu Tham Khảo

Van Lam et al. *"Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language"*. NLPIR 2023.

DOI: https://dl.acm.org/doi/10.1145/3639233.3639244
