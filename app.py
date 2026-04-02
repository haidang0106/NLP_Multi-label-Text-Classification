import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Text Classification API")

# Enable CORS for local testing from static HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(BASE_DIR, 'output', 'models', 'checkpoints', 'best_model.pt')
VOCAB_FILE  = os.path.join(BASE_DIR, 'data', 'process_data', 'vocab.json')
LABEL_FILE  = os.path.join(BASE_DIR, 'data', 'process_data', 'label_map.json')
STOPWORDS_FILE = os.path.join(BASE_DIR, 'data', 'dictionary', 'vietnamese-stopwords.txt')

# Global state
model = None
vocab = {}
label_map = {}
STOPWORDS = set()
IDX_TO_LABEL = {}
NUM_CLASSES = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HARNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_classes_per_level, dropout=0.5):
        super().__init__()
        self.num_levels  = len(num_classes_per_level)
        self.hidden_size = hidden_size
        self.emb         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru       = nn.GRU(embed_dim, hidden_size,
                                  bidirectional=True, batch_first=True)
        self.dropout     = nn.Dropout(dropout)
        self.attention   = nn.ModuleList([
            nn.Linear(hidden_size * 2, 1) for _ in range(self.num_levels)
        ])
        self.ham = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size * 3, n) for n in num_classes_per_level
        ])

    def forward(self, x):
        B    = x.size(0)
        emb  = self.dropout(self.emb(x))
        doc, _ = self.bigru(emb)
        doc  = self.dropout(doc)
        h    = torch.zeros(B, self.hidden_size, device=x.device)
        c    = torch.zeros(B, self.hidden_size, device=x.device)
        preds = []
        for lv in range(self.num_levels):
            score   = self.attention[lv](doc)
            weight  = torch.softmax(score, dim=1)
            context = (weight * doc).sum(dim=1)
            h, c    = self.ham(context, (h, c))
            feat    = self.dropout(torch.cat([context, h], dim=-1))
            preds.append(torch.sigmoid(self.classifiers[lv](feat)))
        return preds

@app.on_event("startup")
def load_resources():
    global model, vocab, label_map, STOPWORDS, IDX_TO_LABEL, NUM_CLASSES

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

    with open(VOCAB_FILE, encoding='utf-8') as f: 
        vocab = json.load(f)
    
    with open(LABEL_FILE, encoding='utf-8') as f: 
        label_map = json.load(f)
        
    with open(STOPWORDS_FILE, encoding='utf-8') as f:
        STOPWORDS = {line.strip() for line in f if line.strip()}

    IDX_TO_LABEL = {
        level: {int(v): k for k, v in label_map[level].items()}
        for level in ['l1', 'l2', 'l3']
    }
    NUM_CLASSES = [len(label_map['l1']), len(label_map['l2']), len(label_map['l3'])]

    # Load Model
    EMBED_DIM   = 100
    HIDDEN_SIZE = 256
    global model
    model = HARNN(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes_per_level=NUM_CLASSES,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print("Startup complete. Model and resources loaded successfully.")

MAX_LEN   = 512
THRESHOLD = 0.5

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def tokenize(text: str) -> list[str]:
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(clean_text(text), format='text').split()
    except ImportError:
        tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def text_to_tensor(text: str) -> torch.Tensor:
    tokens = tokenize(text)
    ids    = [vocab.get(t, 1) for t in tokens][:MAX_LEN]
    ids   += [0] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(device)

@torch.no_grad()
def predict_logic(text: str, threshold: float = THRESHOLD, top_k: int = 3) -> dict:
    x      = text_to_tensor(text)
    preds  = model(x)
    tokens = tokenize(text)

    result = {'tokens': tokens[:10]}

    for i, level in enumerate(['l1', 'l2', 'l3']):
        probs     = preds[i][0].cpu().numpy()
        idx2label = IDX_TO_LABEL[level]
        
        # Lower threshold for L3 to increase recall
        current_threshold = 0.1 if level == 'l3' else threshold

        selected = [
            {'label': idx2label[j], 'prob': float(probs[j])}
            for j in range(len(probs))
            if probs[j] >= current_threshold
        ]

        if not selected:
            top_indices = np.argsort(probs)[::-1][:top_k]
            selected    = [
                {'label': idx2label[j], 'prob': float(probs[j])}
                for j in top_indices
            ]

        selected.sort(key=lambda x: x['prob'], reverse=True)
        result[level] = selected

    return result

class PredictRequest(BaseModel):
    text: str

@app.post("/api/predict")
async def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    try:
        result = predict_logic(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
