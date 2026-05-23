# Biomedical Domain-Adapted Embedding Research Pipeline

This repository contains a complete end-to-end research pipeline for biomedical domain-adapted embedding training and evaluation.

The pipeline is packaged into a single Jupyter Notebook [biomedical_embedding_pipeline_v2.ipynb](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/biomedical_embedding_pipeline_v2.ipynb) and supports running **fully offline** (loading cached models and dataset pairs) or **online** (for downloading new models and generating synthetic data via LLM APIs).

---

## 🚀 How to Run the Pipeline

### 1. Environment Setup

The repository is pre-configured with a Python virtual environment (`.venv`) containing all required libraries (PyTorch, SentenceTransformers, FAISS, LangChain text splitters, UMAP, beautifulsoup4, etc.).

To activate the virtual environment and launch Jupyter:

**On Windows (PowerShell):**
```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Launch Jupyter Notebook
jupyter notebook
```

---

### 2. Running in Offline Mode (Default)

The notebook is configured by default to run in **Offline Mode** (`"offline_mode": True` in `CONFIG`).

In this mode, you do **not** need an internet connection. The notebook will automatically:
1. Load the pre-downloaded Hugging Face models from the local `models/` directory:
   - `sentence-transformers/all-MiniLM-L6-v2` $\rightarrow$ `models/all-MiniLM-L6-v2`
   - `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` $\rightarrow$ `models/BioBERT-sentence-baseline`
   - `NeuML/pubmedbert-base-embeddings` $\rightarrow$ `models/PubMedBERT-embeddings`
2. Skip the API query generation phase and load the pre-cached synthetic dataset pairs directly from:
   - `data/synthetic_pairs/train_pairs.csv`
   - `data/synthetic_pairs/test_pairs.csv`
3. Skip any remote external benchmark downloads (BEIR/NFCorpus).

To run the pipeline offline:
1. Open the [biomedical_embedding_pipeline_v2.ipynb](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/biomedical_embedding_pipeline_v2.ipynb) notebook.
2. Select the `.venv` Python kernel.
3. Run all cells from top to bottom.

---

### 3. Running in Online Mode (Scale Up & Re-generate)

If you have internet access and want to scale up the training data, re-generate synthetic query pairs, or download fresh Hugging Face models, you can run in Online Mode:

1. Open `.env` and set your API keys:
   ```ini
   NVIDIA_API_KEY="your-nvapi-key"
   GROQ_API_KEY="your-groq-key" # optional alternative
   LCM_LLM_PROVIDER="nvidia"
   ```
2. Open the notebook and modify `CONFIG` in Step 2:
   - Set `"offline_mode": False`
   - Set `"mode": "full"` (to select all 300 papers) or keep `"mode": "prototype"` (for 20 papers).
   - Set `"train_pairs_target"` and `"test_pairs_target"` to your desired query limits (e.g. 2000 train / 500 test).
3. Run the notebook. It will call the NVIDIA NIM API (incorporating rate-limiting and backoff delays) to generate query-passage pairs and write them to `data/synthetic_pairs/`, updating the local cache.

---

## 🛠️ Offline Preparation & Verification Scripts

We provide utility scripts under the `scratch/` directory:

1. **[prepare_offline.py](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/scratch/prepare_offline.py)**:
   Downloads all target SentenceTransformer models and generates a small synthetic query pair cache via LLM. Run this script while online to ensure everything is prepared:
   ```bash
   .venv\Scripts\python scratch/prepare_offline.py prototype
   ```
2. **[verify_offline.py](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/scratch/verify_offline.py)**:
   Simulates offline mode programmatically. It runs the entire notebook top-to-bottom using cached resources and validates that all 12 expected output artifacts (fine-tuned models, FAISS indices, evaluation CSVs, term similarity shift CSVs, and UMAP plots) are correctly created without errors.
   ```bash
   .venv\Scripts\python scratch/verify_offline.py
   ```

---

## 📂 Project Directory Structure

```
LCM-Research/
├── biomedical_embedding_pipeline_v2.ipynb  # Main Research Pipeline
├── walkthrough.md                          # Walkthrough documentation
├── requirements.txt                        # Project dependencies
├── .env                                    # LLM API keys
├── data/
│   ├── raw_papers/                         # 300 source PMC XML/PDF papers
│   ├── processed_chunks/                   # Extracted and chunked paragraphs
│   ├── splits/                             # Train/test paper split mapping
│   └── synthetic_pairs/                    # Cache of synthetic query-passage triplets
├── models/
│   ├── all-MiniLM-L6-v2/                   # Local base model weights
│   ├── BioBERT-sentence-baseline/          # Local BioBERT weights
│   ├── PubMedBERT-embeddings/              # Local PubMedBERT weights
│   └── biomedical_embedding_ft/            # Fine-tuned model outputs
├── artifacts/
│   └── biomedical_pipeline/                # Retrieval indices, evaluation results, and plots
└── scratch/
    ├── prepare_offline.py                  # Preparation and caching script
    └── verify_offline.py                   # Offline verification script
```
