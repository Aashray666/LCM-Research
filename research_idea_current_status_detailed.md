# Research Idea and Current Status

## 1. Project Overview

This project investigates whether **domain-adapted biomedical embeddings** can improve retrieval performance over general-purpose embeddings when applied to PubMed Central literature.

The central idea is to compare retrieval systems that are identical in every respect except for the embedding model used. One system uses a frozen general-purpose embedding model, and the other uses the same base model after fine-tuning on biomedical text. Two dedicated biomedical pre-trained models (BioBERT, PubMedBERT) are also included as external baselines.

---

## 2. Research Question

> **Do domain-adaptive fine-tuning techniques on LLM-generated synthetic triplets produce embeddings that outperform both general-purpose and biomedical-specific pre-trained models for biomedical passage retrieval?**

A related sub-question is whether the fine-tuned model produces a more semantically coherent embedding space for biomedical terminology — measurable via cosine similarity shifts and UMAP cluster separation.

---

## 3. Core Hypothesis

General-purpose embeddings are less effective in biomedical retrieval because important scientific terms may be semantically diluted by their broader-language usage. Dedicated biomedical pre-trained models (BioBERT, PubMedBERT) help but were not trained with a retrieval objective.

By fine-tuning with a contrastive ranking loss on biomedical-specific `(query, positive, hard_negative)` triplets, the model should:
- Place query and matching passage closer together in embedding space
- Improve retrieval ranking quality over the frozen baseline
- Outperform biomedical pre-trained models that lack retrieval-specific supervision

---

## 4. Experimental Design

### Fixed Components (same across all models)
- Corpus source (297 PMC Open Access XML papers)
- Chunking strategy (800-char sentence-aware, 150-char overlap, section-tagged)
- Retrieval index type (FAISS `IndexFlatIP` with L2-normalized embeddings)
- Query generation method (NVIDIA NIM `meta/llama-3.3-70b-instruct`, batched 5 chunks/call)
- Evaluation metrics (MRR, NDCG@10, Recall@k, Hit Rate@3)

### Variable Component (only the embedding model changes)
- **general_baseline:** `all-MiniLM-L6-v2`, frozen (no fine-tuning)
- **fine_tuned:** `all-MiniLM-L6-v2`, fine-tuned on 1,998 biomedical triplets with MNRL
- **BioBERT_sentence_baseline:** `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- **PubMedBERT_embedding_baseline:** `NeuML/pubmedbert-base-embeddings`

This design isolates the effect of domain adaptation from architecture differences.

---

## 5. Dataset

**Corpus:** PubMed Central Open Access Subset — 297 XML papers processed

| Split | Papers | Chunks |
|---|---|---|
| Train | 238 | ~19,300 |
| Test | 59 | ~4,800 |

**Synthetic Pairs Generated:**

| Split | Pairs | Method |
|---|---|---|
| Train | 1,998 | LLM (5 chunks/call), 4 personas, TF-IDF hard negatives |
| Test | 498 | Same method, held-out papers only |

Query personas: **Factual, Conversational, Keyword Search, Multi-Hop** (balanced).

---

## 6. Pipeline

The experiment runs through a 16-step Jupyter Notebook (`biomedical_embedding_pipeline_ayan_v1.ipynb`):

1. Setup & environment detection
2. Imports + configuration
3. File discovery (PDF/XML)
4. Text extraction (XML parser, formula removal, section tagging)
5. Sentence-aware chunking (`RecursiveCharacterTextSplitter`)
6. Document-level 80/20 paper split (prevents leakage)
7. Query generation helpers (prompt templates + validation)
8. TF-IDF hard-negative miner + LLM API setup
9. Synthetic pair generation (batched, incremental, resumable)
10. Integrity validation (zero-overlap assertion)
11. Fine-tuning (`MultipleNegativesRankingLoss`)
12. Retrieval indexing (FAISS)
13. Evaluation on held-out test queries
14. Optional: External BEIR benchmark
15. Representation analysis (cosine similarity + UMAP)
16. Artifact summary

---

## 7. Results ✅

### Average Retrieval Metrics (498 test queries, 4 query types)

| Model | Avg MRR | Avg NDCG@10 | Avg Recall@10 |
|---|---|---|---|
| **fine_tuned (ours)** | **0.709** | **0.757** | **0.918** |
| general_baseline | 0.636 | 0.686 | 0.860 |
| PubMedBERT | 0.631 | 0.689 | 0.883 |
| BioBERT | 0.614 | 0.667 | 0.851 |

### Per-Query-Type MRR

| Model | Factual | Conversational | Keyword | Multi-Hop |
|---|---|---|---|---|
| **fine_tuned** | **0.704** | **0.712** | **0.627** | **0.792** |
| general_baseline | 0.617 | 0.627 | 0.603 | 0.695 |
| PubMedBERT | 0.593 | 0.631 | 0.604 | 0.697 |
| BioBERT | 0.566 | 0.612 | 0.574 | 0.704 |

### Key Findings

- The fine-tuned model outperforms the general baseline by **+11.5% MRR** on average
- The fine-tuned model outperforms PubMedBERT by **+12.4%** and BioBERT by **+15.5%**
- The largest gains appear on **Factual (+24% over BioBERT)** and **Multi-Hop** queries
- **Recall@10 exceeds 86%** across all query types for the fine-tuned model

---

## 8. Evaluation Metrics

### Retrieval Metrics
- **MRR (Mean Reciprocal Rank):** How early the correct passage appears in ranked results
- **NDCG@10:** Ranking quality across the full top-10 list
- **Recall@k:** Whether the correct passage appears in top-k (k = 3, 5, 10, 20)
- **Hit Rate@3:** Binary: does the correct passage appear in top-3?

### Embedding Space Metrics
- Cosine similarity comparisons between biomedical term pairs (pre vs post fine-tuning)
- UMAP visualization of embedding space separation

---

## 9. Risks and Mitigations

| Risk | Status |
|---|---|
| **Scale** — 300 papers may be too small | ✅ 297 papers processed, 24,166 chunks; adequate for a research paper |
| **Leakage** — chunk-level split could contaminate test | ✅ Paper-level split strictly enforced with zero-overlap assertions |
| **Weak queries** — generic queries dilute training signal | ✅ 4 personas + strict validation (must end in `?`, min/max length, token overlap filter) |
| **Novelty gap** — BioBERT/PubMedBERT already exist | ✅ Confirmed both are outperformed — this IS the research contribution |
| **Synthetic test bias** — train and test pairs from same LLM | ⚠️ Acknowledged limitation; external BEIR benchmark can address this |

---

## 10. What Success Looks Like

The project can now make the following defensible claim:

> *"LLM-generated synthetic triplets with MultipleNegativesRankingLoss enable a general-purpose embedding model (`all-MiniLM-L6-v2`) to achieve higher biomedical retrieval accuracy than models specifically pre-trained on biomedical corpora (BioBERT, PubMedBERT), demonstrating that retrieval-objective fine-tuning is more important than domain-specific pre-training for biomedical passage retrieval."*

---

## 11. Next Steps for Publication

- [ ] Run BEIR/NFCorpus external evaluation (`run_external_benchmark_evaluation: True`) for distribution-shifted validation
- [ ] Repeat with multiple random seeds (3 runs) to report mean ± std
- [ ] Add ablation: MNRL vs TripletLoss, with vs without hard negatives
- [ ] Write up results section with the evaluation table above
- [ ] Include UMAP figure and cosine similarity shift table as supporting evidence
