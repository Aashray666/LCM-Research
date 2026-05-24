# Biomedical Domain-Adapted Embedding Research Pipeline

This repository contains a complete end-to-end research pipeline for biomedical domain-adapted embedding training and evaluation. It processes 300 PMC Open Access papers, generates synthetic query-passage triplets via LLM, fine-tunes a sentence embedding model, and evaluates retrieval quality against biomedical-specific baselines (BioBERT, PubMedBERT).

The primary notebook is **[biomedical_embedding_pipeline_ayan_v1.ipynb](biomedical_embedding_pipeline_ayan_v1.ipynb)** — the production-ready version with batched LLM generation and `MultipleNegativesRankingLoss`.

---

## 📊 Key Results

Fine-tuning `all-MiniLM-L6-v2` on LLM-generated biomedical triplets **outperforms both BioBERT and PubMedBERT** across all query types:

| Model | Avg MRR | Avg NDCG@10 | Avg Recall@10 |
|---|---|---|---|
| **fine_tuned (ours)** | **0.709** | **0.757** | **0.918** |
| general_baseline (MiniLM) | 0.636 | 0.686 | 0.860 |
| PubMedBERT | 0.631 | 0.689 | 0.883 |
| BioBERT | 0.614 | 0.667 | 0.851 |

---

## 🚀 How to Run

### 1. Environment Setup

The repository includes a `.venv` Python virtual environment with all dependencies pre-installed (PyTorch, SentenceTransformers, FAISS, LangChain, UMAP, etc.).

```powershell
# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

Open the notebook in VS Code and select the `.venv (Python 3.11)` kernel.

---

### 2. Online Mode — Full Pipeline Run

To generate synthetic pairs from scratch and train a new model:

1. Set your API keys in `.env`:
   ```ini
   NVIDIA_API_KEY="your-nvapi-key"
   GROQ_API_KEY="your-groq-key"   # optional alternative
   LCM_LLM_PROVIDER="nvidia"
   ```
2. In the notebook CONFIG (Step 2), set:
   ```python
   "offline_mode": False,
   "mode": "full",          # uses all 300 papers
   ```
3. Run all cells top to bottom.

The pipeline will:
- Extract and chunk 300 papers (~24,000 chunks)
- Generate synthetic `(query, positive, hard_negative)` triplets via NVIDIA NIM (batched, 5 chunks per LLM call)
- Fine-tune `all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss`
- Evaluate against BioBERT and PubMedBERT baselines

> **Rate limits:** NVIDIA NIM free tier is ~40 RPM. With 5-chunk batching, ~2,000 training pairs ≈ 2–3 hours of generation. The pipeline checkpoints every 25 pairs and **resumes automatically** if interrupted.

---

### 3. Offline Mode — Use Pre-Cached Data

If you already have synthetic pairs and cached models, you can run the full pipeline without internet:

1. Set `"offline_mode": True` in CONFIG
2. Ensure the following exist:
   - `data/synthetic_pairs/train_pairs.csv`
   - `data/synthetic_pairs/test_pairs.csv`
   - `models/all-MiniLM-L6-v2/`
   - `models/BioBERT-sentence-baseline/`
   - `models/PubMedBERT-embeddings/`
3. Run all cells — the pipeline skips all API calls

To pre-cache models while online:
```bash
.venv\Scripts\python scratch/prepare_offline.py prototype
```

---

## 📂 Project Structure

```
LCM-Research/
├── biomedical_embedding_pipeline_ayan_v1.ipynb  # ← Main notebook (use this)
├── biomedical_embedding_pipeline_v2.ipynb       # Original v2 notebook
├── walkthrough.md                               # Full change log and results
├── research_idea_current_status_detailed.md     # Research background
├── implementation_plan.md                       # Technical design notes
├── requirements.txt                             # Python dependencies
├── .env                                         # API keys (not committed)
├── data/
│   ├── raw_papers/          # 297 source PMC XML papers
│   ├── processed_chunks/    # chunks.csv — 24,166 extracted passages
│   ├── splits/              # paper_split.json — train/test paper IDs
│   └── synthetic_pairs/     # train_pairs.csv (1,998) + test_pairs.csv (498)
├── models/
│   ├── all-MiniLM-L6-v2/           # Base model (cached locally)
│   ├── BioBERT-sentence-baseline/  # BioBERT baseline (cached)
│   ├── PubMedBERT-embeddings/      # PubMedBERT baseline (cached)
│   └── biomedical_embedding_ft/    # Fine-tuned model output
├── artifacts/
│   └── biomedical_pipeline/
│       ├── evaluation_results.csv            # Model comparison metrics
│       ├── evaluation_by_query_type.csv      # Per-persona breakdown
│       ├── biomedical_term_similarity.csv    # Cosine similarity analysis
│       ├── umap_baseline_vs_ft.png           # UMAP visualization
│       ├── baseline_test.index               # FAISS index (baseline)
│       ├── finetuned_test.index              # FAISS index (fine-tuned)
│       └── training_config.json             # Training hyperparameters
└── scratch/
    ├── prepare_offline.py   # Downloads and caches models + prototype pairs
    └── verify_offline.py    # Validates full offline pipeline execution
```

---

## 🛠️ Key Technical Decisions

| Decision | Rationale |
|---|---|
| **`MultipleNegativesRankingLoss`** | Better than `TripletLoss` — uses all other batch items as additional negatives, stronger gradient signal |
| **5-chunk batching** | 5× fewer LLM API calls; avoids NVIDIA NIM rate limit while generating same quality pairs |
| **Paper-level 80/20 split** | Prevents data leakage — no chunks from the same paper appear in both train and test |
| **TF-IDF hard negatives** | Provides semantically similar but non-matching negatives as fallback when LLM hard negative is invalid |
| **Incremental checkpointing** | Saves every 25 pairs; safe to interrupt and resume at any time |
