import json
import os
import sys
import subprocess
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def download_models():
    print("=== Downloading Hugging Face SentenceTransformer Models ===")
    from sentence_transformers import SentenceTransformer
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_download = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "BioBERT-sentence-baseline": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "PubMedBERT-embeddings": "NeuML/pubmedbert-base-embeddings",
    }
    
    for name, model_id in models_to_download.items():
        local_path = models_dir / name
        if local_path.exists() and any(local_path.iterdir()):
            print(f"Model {name} already exists locally at: {local_path}")
            continue
        print(f"Downloading {model_id} and saving locally to: {local_path}...")
        model = SentenceTransformer(model_id)
        model.save(str(local_path))
        print(f"Saved model {name} successfully.")
    print("All models downloaded successfully.\n")

def inject_offline_mode_to_notebook():
    print("=== Injecting 'offline_mode' config into the notebook ===")
    nb_path = PROJECT_ROOT / "biomedical_embedding_pipeline_v2.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found at: {nb_path}")
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    modified = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_text = "".join(cell.get("source", []))
            if "CONFIG = {" in source_text and "offline_mode" not in source_text:
                print("Found CONFIG cell. Injecting offline mode logic...")
                
                # We need to find "CONFIG = {" and insert '"offline_mode": True,' inside the dictionary,
                # and then append the resolution logic at the end of the cell.
                lines = cell["source"]
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if "CONFIG = {" in line:
                        new_lines.append('    # Mode: True to run fully offline (uses pre-cached models & pairs)\n')
                        new_lines.append('    "offline_mode": True,\n')
                
                # Append resolution logic to the end of the cell
                resolution_logic = (
                    "\n# ── Auto-configuration for Offline Mode ──\n"
                    "if CONFIG.get(\"offline_mode\", False):\n"
                    "    print(\"\\n>>> OFFLINE MODE ACTIVE: Overriding model paths and disabling network steps <<<\")\n"
                    "    CONFIG[\"base_embedding_model\"] = \"models/all-MiniLM-L6-v2\"\n"
                    "    CONFIG[\"run_query_generation\"] = False\n"
                    "    CONFIG[\"run_external_benchmark_evaluation\"] = False\n"
                    "    CONFIG[\"external_embedding_models\"] = {\n"
                    "        \"BioBERT_sentence_baseline\": \"models/BioBERT-sentence-baseline\",\n"
                    "        \"PubMedBERT_embedding_baseline\": \"models/PubMedBERT-embeddings\",\n"
                    "    }\n"
                    "    # Ensure absolute paths resolved from PROJECT_ROOT\n"
                    "    for k, v in CONFIG[\"external_embedding_models\"].items():\n"
                    "        if not Path(v).is_absolute():\n"
                    "            CONFIG[\"external_embedding_models\"][k] = str(PROJECT_ROOT / v)\n"
                    "    if not Path(CONFIG[\"base_embedding_model\"]).is_absolute():\n"
                    "        CONFIG[\"base_embedding_model\"] = str(PROJECT_ROOT / CONFIG[\"base_embedding_model\"])\n"
                )
                new_lines.append(resolution_logic)
                cell["source"] = new_lines
                modified = True
                break
                
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully with 'offline_mode' support.\n")
    else:
        print("Notebook already has 'offline_mode' configured or CONFIG cell not found.\n")

def run_query_generation_script(mode="prototype"):
    print(f"=== Generating Synthetic Pairs in '{mode}' mode ===")
    nb_path = PROJECT_ROOT / "biomedical_embedding_pipeline_v2.ipynb"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # We will extract python cells up to step 10 (Integrity Validation)
    code_cells = []
    # Identify the cells we want to execute.
    # We will ignore cell 1 (package check) and collect subsequent ones up to Step 10
    started = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_text = "".join(cell.get("source", []))
            if "import importlib.util" in source_text and "detect environment" in source_text.lower():
                # This is setup cell, skip it
                continue
            if "CONFIG = {" in source_text:
                started = True
            if started:
                code_cells.append(source_text)
                if "validate_pairs(" in source_text and "integrity checks passed" in source_text.lower():
                    # This is Step 10 validation, stop here
                    break
                    
    # Combine code cells
    script_content = (
        "from __future__ import annotations\n"
        "import sys, os\n"
        "from pathlib import Path\n"
        f"os.environ['LCM_PROJECT_ROOT'] = r'{str(PROJECT_ROOT)}'\n"
        "sys.path.insert(0, r'" + str(PROJECT_ROOT) + "')\n\n"
    )
    
    # We concatenate the cells, but we want to force the CONFIG mode to the chosen mode (prototype or full)
    # and ensure run_query_generation is True to generate pairs now while online.
    combined_code = "\n# --- CELL ---\n".join(code_cells)
    
    # Remove __future__ imports from combined code to avoid SyntaxError in middle of file
    combined_code = combined_code.replace("from __future__ import annotations", "")
    
    # Replace CONFIG mode dynamically
    combined_code = combined_code.replace('"mode": "full"', f'"mode": "{mode}"')
    combined_code = combined_code.replace('"mode": "prototype"', f'"mode": "{mode}"')
    # Make sure run_query_generation is True
    combined_code = combined_code.replace('"run_query_generation": False', '"run_query_generation": True')
    
    # Adjust targets for prototype mode to run quickly
    if mode == "prototype":
        combined_code = combined_code.replace('"train_pairs_target": 2000', '"train_pairs_target": 20')
        combined_code = combined_code.replace('"test_pairs_target": 500', '"test_pairs_target": 10')
    
    script_content += combined_code
    
    # Write to a temp run script
    temp_script = PROJECT_ROOT / "scratch" / "run_generation.py"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    temp_script.write_text(script_content, encoding="utf-8")
    
    print(f"Running data generation script using the virtual environment...")
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable # fallback
        
    try:
        # Run the generation script as a subprocess
        result = subprocess.run(
            [str(python_exe), str(temp_script)],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        print("Data generation run completed successfully.")
    finally:
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()

if __name__ == "__main__":
    # 1. Download models
    download_models()
    
    # 2. Inject offline flag into the notebook
    inject_offline_mode_to_notebook()
    
    # 3. Generate synthetic pairs (default to prototype for speed, change to "full" if full is needed)
    mode = "prototype"
    if len(sys.argv) > 1 and sys.argv[1] in ["prototype", "full"]:
        mode = sys.argv[1]
    run_query_generation_script(mode)
