# 📰 Vietnamese News Classification — BERT + Bi-GRU + LSTM + CNN 1D

Hệ thống phân loại văn bản tin tức tiếng Việt sử dụng kiến trúc kết hợp **BERT Embedding → Bi-GRU → LSTM → CNN 1D → Linear + Sigmoid**, được huấn luyện trên dataset [binhvq-news-corpus](https://huggingface.co/datasets/ademax/binhvq-news-corpus) từ HuggingFace.

## 🏗️ Kiến trúc Mô hình

```
Input Text
    │
    ▼
┌──────────────────────┐
│  BERT Embedding      │  bert-base-multilingual-cased
│  (768-dim)           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Bi-GRU Layer        │  hidden_size=128, bidirectional
│  (256-dim output)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LSTM Layer          │  hidden_size=128
│  (128-dim output)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  CNN 1D + MaxPool    │  out_channels=64, kernel_size=3
│  (64-dim output)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Linear + Sigmoid    │  → Multi-label probabilities
└──────────────────────┘
```

## 📊 Dataset

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | [ademax/binhvq-news-corpus](https://huggingface.co/datasets/ademax/binhvq-news-corpus) |
| **Kích thước gốc** | ~14 triệu bài báo |
| **Mẫu sử dụng** | 20,000 (streaming mode) |
| **Text đầu vào** | `title` + `summary` |
| **Nhãn** | `category` (Công nghệ, Thể thao, Kinh tế, Giáo dục, ...) |
| **Ngôn ngữ** | Tiếng Việt |

## 🚀 Cách Chạy

### Trên Google Colab (Khuyến nghị)

1. Upload file `multi_label_classification.ipynb` lên [Google Colab](https://colab.research.google.com/)
2. Chọn **Runtime → Change runtime type → GPU**
3. Chạy lần lượt từng cell

### Trên máy local

```bash
# Cài đặt thư viện
pip install torch transformers datasets scikit-learn pandas numpy tqdm

# Tạo lại notebook (nếu cần)
python generate_notebook.py
```

## 📁 Cấu trúc Project

```
.
├── README.md                           # File này
├── Agents.md                           # Đặc tả & quy chuẩn dự án
├── generate_notebook.py                # Script sinh notebook tự động
└── multi_label_classification.ipynb    # Notebook chính (8 bước)
```

## 📋 Các Bước trong Notebook

| Bước | Nội dung | Mô tả |
|------|----------|-------|
| 1 | Cài đặt & Import | Thư viện, device setup, seed |
| 2 | Chuẩn bị Dữ liệu | Load HuggingFace dataset, cleaning, binarization |
| 3 | Dataset & DataLoader | Custom `MultiLabelDataset`, tokenization |
| 4 | Định nghĩa Model | BERT + Bi-GRU + LSTM + CNN 1D |
| 5 | Loss & Optimizer | BCEWithLogitsLoss, AdamW (2 learning rates) |
| 6 | Training Loop | Train + Validation với progress bar |
| 7 | Đánh giá | Classification report, F1/Precision/Recall |
| 8 | Inference | Dự đoán trên văn bản mới |

## ⚙️ Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| `MAX_LEN` | 128 |
| `BATCH_SIZE` | 16 |
| `EPOCHS` | 5 |
| `BERT_LR` | 2e-5 |
| `CLASSIFIER_LR` | 1e-3 |
| `DROPOUT` | 0.3 |
| `GRU_HIDDEN` | 128 |
| `LSTM_HIDDEN` | 128 |
| `CNN_OUT_CHANNELS` | 64 |
| `THRESHOLD` | 0.5 |

## 🔧 Tech Stack

- **PyTorch** — Deep learning framework
- **HuggingFace Transformers** — BERT tokenizer & model
- **HuggingFace Datasets** — Streaming data loading
- **scikit-learn** — Metrics & preprocessing
- **pandas / numpy** — Data manipulation

## 📝 Ghi chú

- Dataset sử dụng **streaming mode** để tránh hết bộ nhớ trên Colab free
- BERT model: `bert-base-multilingual-cased` (hỗ trợ tiếng Việt)
- Loss function: `BCEWithLogitsLoss` — tương thích cả single-label và multi-label
- Có thể điều chỉnh `NUM_SAMPLES` để tăng/giảm lượng dữ liệu huấn luyện

## 📜 License

MIT
