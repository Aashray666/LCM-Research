# Biomedical Embedding Pipeline v2 Walkthrough

I have completely refactored your pipeline into a single, comprehensive Jupyter Notebook (`biomedical_embedding_pipeline_v2.ipynb`) that is fully compatible with both local execution and Google Colab. 

Here is an overview of what was done, how it works, and how to use it.

## 1. Flow of the Notebook

The notebook follows a logical 16-step flow to process your papers, generate synthetic training data, fine-tune an embedding model, and evaluate it:

1.  **Setup & Environment Detection:** Automatically detects if you are in Google Colab (mounts Google Drive) or running locally.
2.  **Imports + Configuration:** Loads your API keys and defines all pipeline parameters (like corpus size, chunk sizes, and LLM settings) in one central `CONFIG` dictionary.
3.  **File Discovery:** Finds the 300 biomedical papers (PDFs or XMLs) in your data directory.
4.  **Text Extraction:** Parses the papers, removes uninformative noise (like formulas), and extracts the body text along with section headers.
5.  **Sentence-Aware Chunking:** Uses `RecursiveCharacterTextSplitter` to break passages into chunks while respecting sentence boundaries, and prepends the section context (e.g., "[results]").
6.  **Document-Level Split:** Safely splits papers into 80% train and 20% test *at the paper level* to prevent data leakage.
7.  **Query Generation Helpers:** Defines functions to validate queries and format prompts for different personas (Factual, Conversational, Keyword, Multi-Hop).
8.  **Hard-Negative Miner + NVIDIA NIM API:** Sets up a TF-IDF miner to find tricky negative examples and defines the adaptive rate-limited LLM caller.
9.  **Generate Synthetic Pairs:** Calls the LLM to generate `(query, positive_chunk, hard_negative)` triplets for training and testing.
10. **Integrity Validation:** Runs automated checks to ensure no data leaked between train and test sets.
11. **Fine-Tuning:** Uses `SentenceTransformers` with a `TripletLoss` objective to fine-tune your base embedding model.
12. **Retrieval Indexing:** Builds FAISS vector indexes for the baseline model, your new fine-tuned model, and any external baselines.
13. **Evaluation:** Tests all models on the held-out synthetic test set and calculates Recall, MRR, NDCG, and Hit Rate.
14. **External Benchmark Alignment:** (Optional) Evaluates models against standard datasets like BEIR/NFCorpus.
15. **Representation Analysis:** Calculates cosine similarity shifts for biomedical terms and generates a UMAP visualization.
16. **Artifact Summary:** Lists all the datasets, models, and evaluation files produced by the run.

## 2. What Inputs Does It Take?

To run this pipeline, you need the following inputs:

*   **Biomedical Papers:** The raw `.pdf` or `.xml` files. 
    *   *Where:* Placed in the `data/raw_papers/` directory within your project root. (You already have 300 XML files there).
*   **LLM API Key:** For generating the synthetic training queries.
    *   *Where:* Handled via your `.env` file or Google Colab Secrets.

## 3. LLM API Key Usage (NVIDIA NIM)

You requested to use **NVIDIA NIM** due to its better rate limits, and you provided your API key.

*   **Provider Set:** I have configured the notebook to use `NVIDIA` as the primary LLM provider (`LCM_LLM_PROVIDER="nvidia"` in your `.env` file).
*   **API Key:** I have stored your provided API key (`NVIDIA_API_KEY="nvapi-..."`) in your project's `.env` file. 
*   **Model Used:** It defaults to using `meta/llama-3.3-70b-instruct` via the NVIDIA integration endpoint (`https://integrate.api.nvidia.com/v1`).
*   **Rate Limit Handling:** The NVIDIA free tier allows **40 Requests Per Minute (RPM)**. I have built an **adaptive rate limiter** directly into the notebook. It ensures requests are spaced by at least 1.8 seconds and automatically applies exponential backoff if it hits a `429 Too Many Requests` error, ensuring the query generation phase runs smoothly overnight without crashing.

## 4. Risks Addressed

Based on your `cm_research_pipeline.svg`, the following risks were mitigated:
*   **Scale Risk:** The notebook is parameterized to handle the full 300-paper set seamlessly.
*   **Query Quality Risk:** The LLM now generates diverse queries using personas and applies strict validation filters.
*   **Baseline Coverage:** Added `BioSentBERT` and `PubMedBERT` as explicit comparative baselines during evaluation.
*   **Leakage Risk:** The document-level split (Step 6) is strictly enforced with zero-overlap assertions.

You can now open `biomedical_embedding_pipeline_v2.ipynb` and run it top-to-bottom!

## 5. Offline Readiness Configuration

To enable running the pipeline without an active internet connection (e.g., on a local server or offline workstation), we prepared and cached the necessary dependencies and resources:

*   **Pre-cached Models**: Downloaded and cached three models under the `models/` directory:
    *   `sentence-transformers/all-MiniLM-L6-v2` $\rightarrow$ [all-MiniLM-L6-v2](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/models/all-MiniLM-L6-v2)
    *   `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` $\rightarrow$ [BioBERT-sentence-baseline](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/models/BioBERT-sentence-baseline)
    *   `NeuML/pubmedbert-base-embeddings` $\rightarrow$ [PubMedBERT-embeddings](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/models/PubMedBERT-embeddings)
*   **Pre-cached Synthetic Dataset**: Generated the training and test dataset pairs while online and saved them under [synthetic_pairs/](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/data/synthetic_pairs/). This allows the pipeline to skip the LLM API request phase entirely:
    *   [train_pairs.csv](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/data/synthetic_pairs/train_pairs.csv)
    *   [test_pairs.csv](file:///c:/Users/aashr/Desktop/Projects/LCM-Research/data/synthetic_pairs/test_pairs.csv)
*   **Auto-configured `offline_mode` Flag**: Modified the `CONFIG` dictionary in Step 2 of the notebook to add `"offline_mode": True`. When active, it automatically:
    *   Redirects model loading paths to the local `models/` folder.
    *   Sets `"run_query_generation"` to `False` (bypasses online API requests and loads the cached CSVs).
    *   Disables remote external benchmark downloads (`"run_external_benchmark_evaluation": False`).
*   **Fixed a JSON serialization bug**: Resolved a bug in standard Step 11 where fine-tuned training config serialization crashed due to passing `numpy.bool_` into `json.dumps()`. This is now correctly handled via `bool(use_trip)`.

To toggle between online and offline runs, simply change the `"offline_mode"` flag in `CONFIG` to `True` or `False`.

