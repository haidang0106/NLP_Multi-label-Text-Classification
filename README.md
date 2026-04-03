# HARNN — Phân Loại Văn Bản Đa Nhãn Phân Cấp Tiếng Việt

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Huấn luyện và triển khai mô hình **HARNN** (Hierarchical Attention Recurrent Neural Network) để phân loại bài báo tiếng Việt theo **3 cấp độ nhãn**:

```
L1 (Lĩnh vực) → L2 (Lĩnh vực con) → L3 (Chi tiết)
```

## Kết Quả Đánh Giá

| Cấp độ | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| **L1** | 0.931     | 0.931  | 0.931    |
| **L2** | 0.779     | 0.784  | 0.781    |
| **L3** | 0.795     | 0.795  | 0.795    |
| **Global** | —     | —      | **0.836** |

---

## Hướng Dẫn Cài Đặt

### 1. Tạo và kích hoạt virtual environment

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 2. Cài đặt PyTorch

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU:**
```bash
pip install torch torchvision torchaudio
```

### 3. Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

## Chuẩn Bị Dữ Liệu

```bash
python data/load_hf_dataset.py
```

Script này tải dataset từ [HuggingFace](https://huggingface.co/datasets/dat7505/hierarchical_multi_label_dataset) và lưu vào `data/raw_data.json`.

---

## Cách Chạy

### Tùy chọn A: Web Demo

1. Khởi chạy backend:
   ```bash
   python app.py
   ```
2. Mở file `demo_ui/code.html` bằng trình duyệt
3. Dán văn bản tiếng Việt và nhấn **"Run Classification"**

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

---

## Cấu Trúc Dự Án

```
NLP_Multi-label-Text-Classification/
├── app.py                          # FastAPI REST API server
├── requirements.txt                # Python dependencies
├── README.md                       # File này
├── .gitignore                      # Git ignore rules
├── link_data.txt                   # Link HuggingFace dataset
│
├── data/
│   ├── load_hf_dataset.py          # Script tải dataset
│   ├── raw_data.json               # Dữ liệu gốc (gitignored, tự sinh)
│   └── dictionary/
│       ├── vietnamese-stopwords.txt
│       └── vietnamese-stopwords-dash.txt
│
├── notebooks/
│   └── main_workflow.ipynb         # Pipeline đầy đủ: preprocess → train → predict → evaluate
│
├── demo_ui/
│   ├── code.html                   # Frontend (TailwindCSS)
│   ├── DESIGN.md                   # Tài liệu design system
│   └── screen.png                  # Ảnh chụp giao diện
│
└── output/
    ├── models/checkpoints/         # Model weights (gitignored)
    ├── results/                    # Training history & evaluation metrics
    └── figures/                    # Confusion matrices & learning curves
```

---

## Kiến Trúc Mô Hình

**HARNN** (Hierarchical Attention Recurrent Neural Network):

```
Input tokens → Embedding (69,481 × 100) → BiGRU (256, bidirectional)
  → Per-level Attention → LSTMCell (hierarchical memory)
  → Linear classifiers × 3 (mỗi classifier cho một cấp nhãn)
  → Sigmoid outputs (multi-label)
```

### Thiết Kế Chính

- **Word2Vec** (skip-gram, 100d) cho word embeddings
- **BiGRU** cho ngữ cảnh mức văn bản
- **Per-level attention** để nắm bắt đặc trưng riêng cho từng cấp nhãn
- **LSTMCell** để lan truyền thông tin phân cấp giữa các cấp
- **Sigmoid** outputs cho bài toán multi-label classification

### Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Embedding dim | 100 |
| Hidden size | 256 |
| Max sequence length | 512 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Epochs | 10 |
| Batch size | 64 |
| Dropout | 0.5 |

---

## Dataset

| Thống kê | Giá trị |
|----------|---------|
| Nguồn | Bài báo VnExpress |
| Tổng số mẫu | 23,660 |
| Train / Val / Test | 18,865 / 2,353 / 2,362 |
| Lớp L1 | 12 |
| Lớp L2 | 47 |
| Lớp L3 | 20 |
| Vocabulary size | 69,481 (min count=3, min tokens=20) |

---

## API Reference

### `POST /api/predict`

**Request:**
```json
{
  "text": "Google chi 1 ty USD de dao tao AI tai cac truong dai hoc My"
}
```

**Response:**
```json
{
  "l1": [{"label": "Khoa hoc", "prob": 0.92}],
  "l2": [{"label": "AI", "prob": 0.87}],
  "l3": [{"label": "Machine Learning", "prob": 0.65}],
  "tokens": ["google", "chi", "ty", "usd", "dao_tao", "ai"]
}
```

---

## License

MIT

---

## Tài Liệu Tham Khảo

Van Lam et al. *"Exploring Hierarchical Multi-Label Text Classification Models using Attention-Based Approaches for Vietnamese language"*. NLPIR 2023.

DOI: [10.1145/3639233.3639244](https://dl.acm.org/doi/10.1145/3639233.3639244)
