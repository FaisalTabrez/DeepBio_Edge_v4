import os
from pathlib import Path
import logging

# ==========================================
# @BioArch & @Data-Ops: Master Configuration
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepBio-Config')

def get_base_path() -> Path:
    """
    Detects the runtime environment.
    Prioritizes the Pendrive location E:/DeepBio_Scan.
    Falls back to current working directory if not found.
    """
    pendrive_path = Path("E:/DeepBio_Scan")
    if pendrive_path.exists():
        logger.info(f"Pendrive detected at {pendrive_path}")
        return pendrive_path
    
    cwd_path = Path(os.getcwd())
    logger.warning(f"Pendrive not found at E:/DeepBio_Scan. Using local CWD: {cwd_path}")
    return cwd_path

# 1. Global Path Detection
BASE_DIR = get_base_path()

# Define core paths
DB_PATH = BASE_DIR / 'data' / 'db'
ATLAS_TABLE = 'reference_atlas_v100k'
WORMS_ORACLE = BASE_DIR / 'data' / 'taxonomy_db' / 'worms_deepsea_ref.csv'
TAXON_DIR = BASE_DIR / 'data' / 'taxonomy_db'
LOG_FILE = BASE_DIR / 'logs' / 'session.log'

# Legacy paths for compatibility
RAW_PATH = BASE_DIR / 'data' / 'raw'
RESULTS_PATH = BASE_DIR / 'results'
TAXONKIT_EXE_PATH = BASE_DIR / 'taxonkit.exe'

# 2. System States
# Toggle between real 100k data and mock fallbacks if paths are missing
IS_DEMO_MODE = not (DB_PATH.exists() and (DB_PATH / f"{ATLAS_TABLE}.lance").exists())

if IS_DEMO_MODE:
    logger.warning("100k Atlas not found. System running in DEMO/MOCK mode.")
else:
    logger.info("100k Atlas detected. System running in FULL mode.")

def initialize_folders():
    """
    Ensures the directory structure exists on the target drive.
    Does not overwrite existing data.
    """
    dirs_to_create = [DB_PATH, RAW_PATH, LOG_FILE.parent, RESULTS_PATH, TAXON_DIR]
    
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")

if __name__ == "__main__":
    initialize_folders()
    print(f"Configuration Initialized on {BASE_DIR}")
    print(f"Demo Mode: {IS_DEMO_MODE}")
