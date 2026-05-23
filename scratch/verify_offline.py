import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_verification():
    print("=== Extracting and Preparing Verification Script ===")
    nb_path = PROJECT_ROOT / "biomedical_embedding_pipeline_v2.ipynb"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_text = "".join(cell.get("source", []))
            # Skip the environment check / installation cell
            if "import importlib.util" in source_text and "detect environment" in source_text.lower():
                continue
            code_cells.append(source_text)
            
    # Combine cells with top imports
    script_content = (
        "from __future__ import annotations\n"
        "import sys, os\n"
        "from pathlib import Path\n"
        f"os.environ['LCM_PROJECT_ROOT'] = r'{str(PROJECT_ROOT)}'\n"
        "sys.path.insert(0, r'" + str(PROJECT_ROOT) + "')\n\n"
    )
    
    combined_code = "\n# --- CELL ---\n".join(code_cells)
    
    # Remove future imports and clean magic
    combined_code = combined_code.replace("from __future__ import annotations", "")
    
    # Force offline_mode = True and mode = prototype
    combined_code = combined_code.replace('"offline_mode": False', '"offline_mode": True')
    combined_code = combined_code.replace('"mode": "full"', '"mode": "prototype"')
    
    script_content += combined_code
    
    # Write to a verification script
    verify_script = PROJECT_ROOT / "scratch" / "run_verification.py"
    verify_script.parent.mkdir(parents=True, exist_ok=True)
    verify_script.write_text(script_content, encoding="utf-8")
    
    print("Running verification script (offline simulation)...")
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable
        
    try:
        # Run the full pipeline in offline mode!
        result = subprocess.run(
            [str(python_exe), str(verify_script)],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        print("\n=== Offline Pipeline verification completed successfully! ===")
        check_artifacts()
    except subprocess.CalledProcessError as exc:
        print(f"\nVerification run failed with exit code {exc.returncode}")
        sys.exit(1)
    finally:
        # Clean up
        if verify_script.exists():
            verify_script.unlink()

def check_artifacts():
    print("\n=== Checking generated artifacts ===")
    expected_files = [
        "data/processed_chunks/chunks.csv",
        "data/splits/paper_split.json",
        "data/synthetic_pairs/train_pairs.csv",
        "data/synthetic_pairs/test_pairs.csv",
        "data/synthetic_pairs/all_pairs.csv",
        "models/biomedical_embedding_ft/pytorch_model.bin",  # check if fine-tuned weights exist
        "artifacts/biomedical_pipeline/evaluation_results.csv",
        "artifacts/biomedical_pipeline/evaluation_by_query_type.csv",
        "artifacts/biomedical_pipeline/biomedical_term_similarity.csv",
        "artifacts/biomedical_pipeline/umap_baseline_vs_ft.png",
        "artifacts/biomedical_pipeline/baseline_test.index",
        "artifacts/biomedical_pipeline/finetuned_test.index",
    ]
    
    all_exist = True
    for rel_path in expected_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            size = p.stat().st_size if p.is_file() else "directory"
            print(f"[OK]  {rel_path} (exists, size={size})")
        else:
            # Check model directory generally if it's there
            if rel_path == "models/biomedical_embedding_ft/pytorch_model.bin":
                m_dir = PROJECT_ROOT / "models/biomedical_embedding_ft"
                if m_dir.exists() and any(m_dir.iterdir()):
                    print(f"[OK]  models/biomedical_embedding_ft/ (exists and contains weights)")
                    continue
            print(f"[ERR] {rel_path} is MISSING!")
            all_exist = False
            
    if all_exist:
        print("\nAll expected output artifacts exist! Offline readiness fully validated.")
    else:
        print("\nSome expected artifacts are missing.")

if __name__ == "__main__":
    run_verification()
