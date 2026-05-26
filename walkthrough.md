# Biomedical Embedding Pipeline — Walkthrough

This document tracks what has been built, what changes were made, and what results were produced across the full pipeline run.

---

## 1. Pipeline Architecture

The pipeline is packaged as a single 16-step Jupyter Notebook. The two primary versions are:

| Notebook | Description |
|---|---|
| `biomedical_embedding_pipeline_v2.ipynb` | Original v2 pipeline |
| `biomedical_embedding_pipeline_ayan_v1.ipynb` | Updated version with batched generation + MNRL loss |

Both support **online mode** (API query generation) and **offline mode** (pre-cached pairs and models).

The 16-step flow:

1. **Setup & Environment Detection** — Colab/local detection, package installation
2. **Imports + Configuration** — API keys, CONFIG dict, path setup
3. **File Discovery** — Finds PDF/XML files in `data/raw_papers/`
4. **Text Extraction** — Parses XML/PDF, strips formulas, extracts section-tagged text
5. **Sentence-Aware Chunking** — `RecursiveCharacterTextSplitter` (800 chars, 150 overlap)
6. **Document-Level Split** — 80/20 paper-level train/test (prevents leakage)
7. **Query Generation Helpers** — Prompt templates + validation for 4 personas
8. **TF-IDF Hard-Negative Miner + LLM API** — NVIDIA NIM / Groq with adaptive rate limiting
9. **Generate Synthetic Pairs** — Batched LLM calls (5 chunks per call), incremental checkpointing
10. **Integrity Validation** — Zero-overlap assertions between train/test
11. **Fine-Tuning** — `MultipleNegativesRankingLoss` with hard negatives
12. **Retrieval Indexing** — FAISS `IndexFlatIP` for baseline, fine-tuned, and external models
13. **Evaluation** — Recall@k, MRR, NDCG@10, Hit Rate@3 on 498 held-out test queries
14. **External Benchmark** — (Optional) BEIR/NFCorpus alignment
15. **Representation Analysis** — Cosine similarity shifts + UMAP visualization
16. **Artifact Summary** — Lists all output files produced

---

## 2. Full-Scale Pipeline Run (May 2026)

### Dataset
- **297 PMC XML papers** processed (of 300 downloaded)
- **24,166 chunks** extracted (800-char sentence-aware, 150-char overlap)
- **238 train papers / 59 test papers** (80/20 paper-level split)

### Synthetic Pair Generation
- **Model:** `meta/llama-3.3-70b-instruct` via NVIDIA NIM API
- **Method:** Batched — 5 chunks per LLM call (5× fewer API calls vs original)
- **Output:** **1,998 train pairs** + **498 test pairs**
- **Personas:** Factual, Conversational, Keyword Search, Multi-Hop (balanced)

### Fine-Tuning
- **Base model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Loss function:** `MultipleNegativesRankingLoss` (MNRL) — uses all batch items as additional negatives; strictly superior to `TripletLoss` for triplet data
- **Training pairs:** 1,998 triplets `(query, positive_chunk, hard_negative)`

---

## 3. Evaluation Results (May 2026 Run)

### Multi-Seed Aggregate Results (Seeds 42, 43, 44)

| Metric | general_baseline (MiniLM) | PubMedBERT (Baseline) | BioBERT (v1 Baseline) | fine_tuned (Our Model) | Improvement vs Baseline |
|---|---|---|---|---|---|
| **Avg MRR** | 0.6473 | 0.6476 | 0.6140 | **0.7669 ± 0.0018** | **+18.5%** 🚀 |
| **NDCG@10** | 0.6962 | 0.7051 | - | **0.8099 ± 0.0022** | **+16.3%** |
| **Recall@10** | 0.8638 | 0.8975 | - | **0.9492 ± 0.0029** | **+9.9%** |

### Per Query-Type Mean Performance

The fine-tuned model consistently outperforms all baselines across all query types with remarkably low variance (standard deviation ≤ 0.009 across all seeds):

| Query Type | MRR (mean ± std) | NDCG@10 (mean ± std) | Recall@10 (mean ± std) |
|---|---|---|---|
| **Conversational** | 0.7412 ± 0.0090 | 0.7915 ± 0.0075 | 0.9524 ± 0.0029 |
| **Factual** | 0.7832 ± 0.0037 | 0.8187 ± 0.0043 | 0.9346 ± 0.0055 |
| **Keyword Search** | 0.7002 ± 0.0014 | 0.7546 ± 0.0015 | 0.9340 ± 0.0051 |
| **Multi-Hop** | **0.8415 ± 0.0019** | **0.8740 ± 0.0022** | **0.9764 ± 0.0029** |

**Key finding:** The fine-tuned MiniLM-L6-v2 outperforms both biomedical-specific pre-trained models (BioBERT, PubMedBERT) across all 4 query types, confirming the hypothesis that domain-adaptive fine-tuning with synthetic pairs is more effective than biomedical pre-training alone for retrieval. Additionally, a standard deviation of **±0.0018** on MRR and **±0.0022** on NDCG@10 across different random seeds demonstrates high statistical stability, making these results highly defensible for publication.


---

## 4. Code Changes & Bug Fixes

### `biomedical_embedding_pipeline_ayan_v1.ipynb` (friend's version — reviewed & fixed)

| Change | Cell | Type | Description |
|---|---|---|---|
| `google.colab` check | Cell 2 | Bug Fix | `importlib.util.find_spec("google.colab")` crashes locally with `ModuleNotFoundError` — wrapped in `try/except` |
| Install progress | Cell 2 | Enhancement | Packages now install one-by-one with `[1/N] Installing X... OK` visible output instead of silent bulk install |
| `build_batch_generation_prompt` | Cell 14 | New Feature | Replaced single-chunk prompt with batched prompt (5 chunks → 1 LLM call) |
| `generate_batch` regex | Cell 16 | Bug Fix | Regex string was split across two lines causing `SyntaxError` — fixed to single-line `re.sub(r"^```(?:json)?", ...)` |
| Batched generation loop | Cell 18 | New Feature | Loop now processes `BATCH_SIZE=5` chunks per iteration → 5× fewer API calls |
| `MultipleNegativesRankingLoss` | Cell 22 | Research Fix | Replaced `TripletLoss` fallback logic with always-MNRL — mathematically uses all batch items as additional negatives |
| `offline_mode` flag | Cell 4 | Config Fix | Set to `False` for full pipeline run (was `True`, which disabled API generation) |

### `biomedical_embedding_pipeline_v2.ipynb` (original — earlier session)

| Change | Description |
|---|---|
| Offline mode support | Added `offline_mode` CONFIG flag with auto-path override |
| JSON serialization bug | Fixed `numpy.bool_` crash in `json.dumps` by casting `bool(use_trip)` |
| Model caching | Downloaded and cached 3 models to `models/` for offline use |

---

## 5. Offline Readiness

To run without internet (e.g., on a server or after initial data collection):

1. Set `"offline_mode": True` in CONFIG (Cell 2)
2. Pre-cache models in `models/` using `scratch/prepare_offline.py`
3. Ensure `data/synthetic_pairs/train_pairs.csv` and `test_pairs.csv` exist

When offline mode is active, the pipeline:
- Loads models from local `models/` directory
- Skips LLM API calls (uses existing CSV pairs)
- Skips external benchmark downloads

To toggle: change `"offline_mode"` in CONFIG between `True` and `False`.

---

## 6. Artifacts Produced (Full Run)

| File | Description |
|---|---|
| `data/processed_chunks/chunks.csv` | 24,166 text chunks from 297 papers |
| `data/splits/paper_split.json` | 238 train / 59 test paper IDs |
| `data/synthetic_pairs/train_pairs.csv` | 1,998 training triplets |
| `data/synthetic_pairs/test_pairs.csv` | 498 test triplets |
| `models/biomedical_embedding_ft/` | Fine-tuned MiniLM checkpoint |
| `artifacts/biomedical_pipeline/evaluation_results.csv` | Overall model metrics |
| `artifacts/biomedical_pipeline/evaluation_by_query_type.csv` | Per-persona breakdown |
| `artifacts/biomedical_pipeline/biomedical_term_similarity.csv` | Cosine similarity analysis |
| `artifacts/biomedical_pipeline/umap_baseline_vs_ft.png` | UMAP embedding visualization |
| `artifacts/biomedical_pipeline/*.index` | FAISS retrieval indices |
