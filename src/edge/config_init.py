import os
from pathlib import Path
import logging

# Configure basic logging
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

# 1. Auto-Drive Detection
BASE_PATH = get_base_path()

# Define core paths
DB_PATH = BASE_PATH / 'data' / 'db'
RAW_PATH = BASE_PATH / 'data' / 'raw'
LOGS_PATH = BASE_PATH / 'logs'
RESULTS_PATH = BASE_PATH / 'results'

# Taxonomy specific paths (External dependencies on the drive)
# The user specified E:/DeepBio_Scan/data/taxonomy_db
TAXONOMY_DB_PATH = BASE_PATH / 'data' / 'taxonomy_db'
# The user specified E:/DeepBio_Scan/taxonkit.exe
TAXONKIT_EXE_PATH = BASE_PATH / 'taxonkit.exe'

def initialize_folders():
    """
    Ensures the directory structure exists on the target drive.
    Does not overwrite existing data.
    """
    dirs_to_create = [DB_PATH, RAW_PATH, LOGS_PATH, RESULTS_PATH]
    
    # We do not verify TAXONOMY_DB_PATH creation here as it's a pre-requisite provided by user
    
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")

# Export paths
PATHS = {
    "BASE": BASE_PATH,
    "DB": DB_PATH,
    "RAW": RAW_PATH,
    "LOGS": LOGS_PATH,
    "TAXONOMY_DB": TAXONOMY_DB_PATH,
    "TAXONKIT": TAXONKIT_EXE_PATH
}

if __name__ == "__main__":
    initialize_folders()
    print(f"Configuration Initialized on {BASE_PATH}")
