# Guidelines for AI Agents (AGENTS.md)

Tài liệu này cung cấp các nguyên tắc, quy chuẩn và luồng công việc (workflow) cho các AI Agent khi tham gia phát triển dự án **Hệ thống Phân loại Văn bản Đa nhãn (Multi-label Text Classification)**. 

Dự án tham khảo kiến trúc mô hình từ bài báo: *Predicting Job Titles from Job Descriptions with Multi-label Text Classification (arXiv:2112.11052)*.

## 1. Định dạng File & Môi trường (Environment & Formatting)

- **Định dạng đầu ra:** Toàn bộ code phải được sinh ra dưới dạng các cell của file Jupyter Notebook (`.ipynb`). Trình bày tuần tự từng bước một (Step-by-step) để chạy mượt mà trên Google Colab hoặc Jupyter cục bộ.
- **Ngôn ngữ:** Python 3.8+
- **Thư viện cốt lõi:** `torch` (PyTorch), `transformers` (Hugging Face), `scikit-learn`, `pandas`, `numpy`, `tqdm`.
- **Formatting:** Thụt lề 4 spaces. Loại bỏ các khoảng trắng thừa ở cuối dòng. 

## 2. Quy chuẩn Đặt tên (Naming Conventions)

- **Biến & Hàm:** Sử dụng `snake_case` (ví dụ: `train_dataloader`, `process_text`).
- **Classes:** Sử dụng `PascalCase` (ví dụ: `MultiLabelDataset`, `JobTitleClassifier`).
- **Hyperparameters & Hằng số:** Sử dụng `UPPER_SNAKE_CASE` (ví dụ: `MAX_LEN`, `BATCH_SIZE`, `LEARNING_RATE`).

## 3. Kiến trúc Mô hình (Model Patterns)

Mô hình học sâu (Deep Learning) cần bám sát cấu trúc được đề xuất trong bài báo nghiên cứu:
- **Pre-trained Language Model:** Sử dụng BERT (ví dụ: `bert-base-multilingual-cased` hoặc mô hình tương đương) để trích xuất Embedding.
- **Deep Learning Layers (Trích xuất đặc trưng):** Thiết lập tuần tự chuỗi xử lý sau embedding:
  1. **Bi-GRU** (Bidirectional Gated Recurrent Unit) để lấy thông tin ngữ cảnh hai chiều.
  2. **LSTM** (Long Short-Term Memory) để nắm bắt phụ thuộc xa.
  3. **CNN 1D** (Convolutional Neural Network) để trích xuất các đặc trưng n-gram cục bộ.
- **Output Layer:** Linear/Dense Layer đi kèm hàm kích hoạt `Sigmoid` (Bắt buộc dùng Sigmoid thay vì Softmax cho bài toán phân loại đa nhãn).

## 4. Cấu trúc các bước trong Notebook (Step-by-step Execution)

Khi Agent tạo code cho hệ thống, phải phân chia rõ ràng thành các block (tương ứng với các cell) theo thứ tự sau:

- **Bước 1: Thiết lập & Import**
  - Cài đặt các thư viện cần thiết (`!pip install transformers evaluate`).
  - Import các module và thiết lập device (`cuda` nếu có GPU, ngược lại dùng `cpu`).
- **Bước 2: Chuẩn bị & Tiền xử lý Dữ liệu**
  - Load dataset bằng `pandas`.
  - Làm sạch văn bản (Text cleaning).
  - Chuyển đổi nhãn (Labels) sang định dạng nhị phân đa nhãn bằng `MultiLabelBinarizer`.
- **Bước 3: Tokenization & Dataset Pipeline**
  - Khởi tạo Tokenizer từ Hugging Face.
  - Viết custom Dataset class kế thừa từ `torch.utils.data.Dataset`.
  - Khởi tạo `DataLoader` cho tập Train và Valid.
- **Bước 4: Xây dựng Kiến trúc Mô hình**
  - Xây dựng class `Bert_BiGRU_LSTM_CNN(nn.Module)`.
  - Chú thích rõ chiều của Tensor (Tensor shape) ở mỗi bước biến đổi (ví dụ: `[batch_size, seq_len, hidden_size]`) để dễ kiểm soát.
- **Bước 5: Định nghĩa Loss Function & Optimizer**
  - Loss Function: `BCEWithLogitsLoss` (Binary Cross Entropy cho multi-label).
  - Optimizer: `AdamW` với learning rate riêng biệt cho nhóm layer của BERT (nhỏ) và nhóm layer classifier (lớn hơn).
- **Bước 6: Vòng lặp Huấn luyện (Training & Validation Loop)**
  - Viết vòng lặp epoch rõ ràng có thanh tiến trình (`tqdm`).
  - Đảm bảo có `model.train()` và `model.eval()` ở đúng pha.
- **Bước 7: Đánh giá Mô hình (Evaluation)**
  - Tính toán và in ra báo cáo các chỉ số: **Micro F1-score**, **Macro F1-score**, Precision và Recall (F1-score là chỉ số quan trọng nhất của bài báo).
- **Bước 8: Suy luận thử nghiệm (Inference)**
  - Viết hàm `predict()` nhận một đoạn text mới và trả về danh sách các nhãn dự đoán vượt qua ngưỡng `threshold = 0.5`.

## 5. Xử lý Lỗi & Tối ưu hóa (Error Handling & Debugging)

- **Tensor Mismatch:** Kiểm tra kỹ tham số `hidden_size` và `out_channels` khi chuyển đổi từ LSTM sang CNN 1D. Yêu cầu dùng `.permute()` chính xác cho CNN.
- **Tránh tràn RAM/VRAM (OOM):** Luôn gọi `torch.cuda.empty_cache()` sau mỗi epoch. Thiết lập mặc định `MAX_LEN = 128` hoặc `256` thay vì 512 để tiết kiệm bộ nhớ khi bắt đầu chạy thử nghiệm.
- **Gradient Exploding:** Áp dụng `torch.nn.utils.clip_grad_norm_` trước bước `optimizer.step()` để ổn định việc huấn luyện các lớp RNN.

## 6. Tiêu chuẩn Ghi chú & Giải thích (Documentation & Comments)

- Mỗi block code (cell) phải được mở đầu bằng một đoạn text Markdown ngắn gọn, giải thích mục đích của cell đó.
- Viết docstring chi tiết (mô tả input, output tensor shape) cho các class mô hình và các hàm xử lý dữ liệu cốt lõi.