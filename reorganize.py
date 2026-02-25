import os
import shutil
from pathlib import Path

def reorganize():
    # 1. Create directories
    dirs = [
        "configs",
        "data/db",
        "data/raw",
        "data/taxonomy_db",
        "logs",
        "src/edge",
        "src/interface",
        "src/utils",
        "results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 2. Move files
    moves = {
        "src/edge/config.py": "configs/config.py",
        "src/interface/app.py": "app.py",
        "src/edge/seed_usb.py": "src/utils/seed_usb.py",
        "src/edge/seed_worms.py": "src/utils/seed_worms.py",
        "src/edge/fetch_real_data.py": "src/utils/fetch_real_data.py",
        "src/edge/verify_taxonomy_engine.py": "src/utils/verify_taxonomy_engine.py",
        "init_lancedb_100k.py": "src/utils/init_lancedb_100k.py",
        "fetch_real_deepsea_data.py": "src/utils/fetch_real_deepsea_data.py",
        "diagnostic_log.py": "src/utils/diagnostic_log.py",
        "manifold_ui.py": "src/utils/manifold_ui.py",
        "Expedition_DeepSea_Batch.fasta": "data/raw/Expedition_DeepSea_Batch.fasta",
        "genomic_manifold.html": "results/genomic_manifold.html"
    }

    for src, dst in moves.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        else:
            print(f"Warning: {src} not found")

    # 3. Update imports in all python files
    for root, _, files in os.walk("."):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content.replace("from configs.config import", "from configs.config import")
                new_content = new_content.replace("import configs.config", "import configs.config")
                
                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated imports in {filepath}")

    # 4. Update Launch_BioScan.bat
    bat_file = "Launch_BioScan.bat"
    if os.path.exists(bat_file):
        with open(bat_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = content.replace("streamlit run src/interface/app.py", "streamlit run app.py")
        
        if new_content != content:
            with open(bat_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {bat_file}")

if __name__ == "__main__":
    reorganize()
