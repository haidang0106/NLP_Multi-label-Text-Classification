import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Text Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(BASE_DIR, 'output', 'models', 'checkpoints', 'best_model.pt')
VOCAB_FILE  = os.path.join(BASE_DIR, 'data', 'process_data', 'vocab.json')
LABEL_FILE  = os.path.join(BASE_DIR, 'data', 'process_data', 'label_map.json')
STOPWORDS_FILE = os.path.join(BASE_DIR, 'data', 'dictionary', 'vietnamese-stopwords.txt')

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

HTML_CONTENT = """
<!DOCTYPE html>
<html class="light" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Text Classifier</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
      tailwind.config = {
        darkMode: "class",
        theme: {
          extend: {
            colors: {
              "on-secondary-fixed": "#12183d",
              "surface-bright": "#f8f9ff",
              "on-tertiary-container": "#ffc7a2",
              "secondary-fixed-dim": "#bec4f2",
              "tertiary-fixed": "#ffdcc6",
              "on-primary-container": "#cacfff",
              "secondary-fixed": "#dee0ff",
              "inverse-on-surface": "#ebf1ff",
              "on-background": "#0d1c2f",
              "primary": "#24389c",
              "tertiary-fixed-dim": "#ffb784",
              "primary-fixed-dim": "#bac3ff",
              "outline": "#757684",
              "on-surface-variant": "#454652",
              "surface-tint": "#4355b9",
              "surface-container": "#e6eeff",
              "tertiary": "#6c3400",
              "surface-container-low": "#eff4ff",
              "on-secondary-container": "#51577f",
              "on-tertiary-fixed": "#301400",
              "surface-dim": "#ccdbf4",
              "on-primary-fixed-variant": "#293ca0",
              "inverse-surface": "#233144",
              "secondary": "#565c84",
              "on-primary": "#ffffff",
              "primary-fixed": "#dee0ff",
              "on-primary-fixed": "#00105c",
              "primary-container": "#3f51b5",
              "surface-variant": "#d5e3fd",
              "surface-container-lowest": "#ffffff",
              "error": "#ba1a1a",
              "on-secondary-fixed-variant": "#3e446b",
              "on-secondary": "#ffffff",
              "outline-variant": "#c5c5d4",
              "surface-container-highest": "#d5e3fd",
              "tertiary-container": "#8f4700",
              "background": "#f8f9ff",
              "on-error-container": "#93000a",
              "on-tertiary-fixed-variant": "#713700",
              "secondary-container": "#c9cffd",
              "inverse-primary": "#bac3ff",
              "surface": "#f8f9ff",
              "on-error": "#ffffff",
              "surface-container-high": "#dde9ff",
              "on-surface": "#0d1c2f",
              "error-container": "#ffdad6",
              "on-tertiary": "#ffffff"
            },
            fontFamily: {
              "headline": ["Manrope", "sans-serif"],
              "body": ["Manrope", "sans-serif"],
              "label": ["Manrope", "sans-serif"]
            },
            borderRadius: {"DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px"},
          },
        },
      }
    </script>
<style>
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
        }
        body { font-family: 'Manrope', sans-serif; }
    </style>
</head>
<body class="bg-background text-on-background min-h-screen selection:bg-primary-fixed selection:text-on-primary-fixed">
<header class="fixed top-0 w-full z-50 bg-[#f8f9ff]/80 backdrop-blur-xl flex justify-between items-center px-6 h-16 w-full">
<div class="bg-[#eff4ff] h-[1px] w-full absolute bottom-0 left-0"></div>
<aside class="h-full w-64 fixed left-0 top-16 bg-[#f8f9ff] flex flex-col p-4 gap-2 border-r border-transparent">
<div class="flex items-center gap-3 px-3 py-4 mb-4">
<div class="w-10 h-10 bg-primary rounded-xl flex items-center justify-center text-white">
<span class="material-symbols-outlined">psychology</span>
</div>
<div>
<h3 class="font-bold text-[#0d1c2f] leading-none">Text Engine</h3>
<p class="text-[10px] text-secondary font-medium tracking-wider uppercase mt-1">V2.4 Active</p>
</div>
</div>
<nav class="flex flex-col gap-1 flex-1">
<a class="flex items-center gap-3 px-4 py-3 bg-[#ffffff] text-[#24389c] shadow-[0px_12px_32px_rgba(13,28,47,0.06)] rounded-xl font-manrope text-sm font-medium transition-all" href="#">
<span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">segment</span>
                Classifier
            </a>
</nav>
<div class="mt-auto flex flex-col gap-1 border-t border-surface-container pt-4">
<a class="flex items-center gap-3 px-4 py-3 text-error hover:bg-error-container/20 transition-all rounded-xl font-manrope text-sm font-medium" href="#">
<span class="material-symbols-outlined">logout</span>
                Sign Out
            </a>
</div>
</aside>
<main class="ml-64 pt-16 min-h-screen">
<div class="max-w-7xl mx-auto p-8">
<header class="mb-12">
<h1 class="text-4xl font-extrabold text-on-surface tracking-tighter mb-2">Text Classification Engine</h1>
<p class="text-secondary body-lg">Analyze sentiment, intent, and entities with multi-label deep learning models.</p>
</header>
<div class="grid grid-cols-12 gap-8">
<section class="col-span-12 lg:col-span-7 flex flex-col gap-8">
<div class="bg-surface-container-lowest rounded-xl p-8 shadow-[0px_12px_32px_rgba(13,28,47,0.04)]">
<label class="block text-xs font-bold uppercase tracking-widest text-secondary mb-4">Input Text</label>
<textarea id="inputText" class="w-full h-48 bg-surface-container-low border-none rounded-xl p-6 text-on-surface placeholder:text-outline/50 focus:ring-2 focus:ring-primary-container transition-all resize-none" placeholder="Paste your content here to begin analysis..."></textarea>
<div class="flex items-center justify-between mt-6">
<button id="runBtn" onclick="runClassification()" class="bg-gradient-to-r from-primary to-primary-container text-white px-8 py-3 rounded-full font-bold shadow-xl shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all ml-auto">
                            Run Classification
                        </button>
</div>
</div>
</section>
<aside class="col-span-12 lg:col-span-5 flex flex-col gap-8">
<div class="bg-surface-container-lowest rounded-xl p-8 shadow-[0px_12px_32px_rgba(13,28,47,0.04)]">
<div class="flex items-center justify-between mb-8">
<h3 class="text-xl font-bold text-on-surface tracking-tight">Classification Results</h3>
<span id="topConfidence" class="px-3 py-1 bg-surface-container-highest text-primary text-xs font-bold rounded-full">--</span>
</div>
<div class="space-y-6" id="resultsContainer">
    <div class="text-secondary text-sm italic">Waiting for input...</div>
</div>
<div class="mt-8 pt-6 border-t border-surface-container">
<div class="flex items-center gap-4">
<div class="flex -space-x-2">
<div class="w-8 h-8 rounded-full border-2 border-white bg-surface-container-high flex items-center justify-center text-[10px] font-bold">ML</div>
<div class="w-8 h-8 rounded-full border-2 border-white bg-surface-container-high flex items-center justify-center text-[10px] font-bold">DL</div>
<div class="w-8 h-8 rounded-full border-2 border-white bg-surface-container-high flex items-center justify-center text-[10px] font-bold">NLP</div>
</div>
<p class="text-xs text-secondary font-medium">Analyzed via 3 Ensemble Models</p>
</div>
</div>
</div>
</aside>
</div>
<footer class="mt-20 flex justify-between items-center py-8 border-t border-surface-container">
</footer>
</div>
</main>
<button class="md:hidden fixed bottom-6 right-6 w-14 h-14 bg-primary text-white rounded-full shadow-2xl flex items-center justify-center active:scale-95 transition-transform z-50">
<span class="material-symbols-outlined">add</span>
</button>
<script>
async function runClassification() {
    const text = document.getElementById('inputText').value;
    if (!text.trim()) {
        alert("Please enter some text to analyze.");
        return;
    }
    const runBtn = document.getElementById('runBtn');
    runBtn.innerText = "Analyzing...";
    runBtn.disabled = true;
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';
        let highestProb = 0;
        let highestLabel = "";
        const topResults = [];
        if (data.l1 && data.l1.length > 0) {
            topResults.push({...data.l1[0], level: 'L1'});
            highestProb = Math.max(highestProb, data.l1[0].prob);
            highestLabel = data.l1[0].label;
        }
        if (data.l2 && data.l2.length > 0) {
            topResults.push({...data.l2[0], level: 'L2'});
        }
        if (data.l3 && data.l3.length > 0) {
            topResults.push({...data.l3[0], level: 'L3'});
        }
        topResults.forEach(item => {
            const perc = Math.round(item.prob * 100);
            const isPrimary = (item.level === 'L1') ? 'bg-primary' : (item.level === 'L2' ? 'bg-primary-container' : 'bg-outline-variant/30');
            container.innerHTML += `
                <div>
                    <div class="flex justify-between items-center mb-2">
                        <span class="font-bold text-sm text-on-surface">${item.label} (${item.level})</span>
                        <span class="text-xs font-bold text-secondary">${perc}%</span>
                    </div>
                    <div class="h-2 w-full bg-surface-container-low rounded-full overflow-hidden">
                        <div class="h-full ${isPrimary} rounded-full" style="width: ${perc}%"></div>
                    </div>
                </div>
            `;
        });
        document.getElementById('topConfidence').innerText = `${Math.round(highestProb * 100)}% Confidence`;
    } catch (error) {
        console.error("Fetch error:", error);
        alert("Failed to analyze text. Is the backend server running?");
    } finally {
        runBtn.innerText = "Run Classification";
        runBtn.disabled = false;
    }
}
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
