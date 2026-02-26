"""
@Data-Ops: Final USB Seeding Script
Purpose: Ingests the Colab-generated Parquet atlas into LanceDB on the USB Stick (Volume E).
Optimization: Applies IVF-PQ Indexing for low-latency searches on limited hardware.
"""
import pandas as pd
import lancedb
import logging
import sys
import os
from pathlib import Path

# Setup Logic for Imports
# We need to ensure we can import from src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.edge.config_init import DB_PATH, RAW_PATH, LOG_FILE, initialize_folders

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Data-Ops")

# File Handler (Bridge to UI)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def seed_reference_atlas():
    """
    Main Execution Block for Edge Node Seeding.
    """
    # 1. Path Definitions
    # User Request: E:/DeepBio_Scan/data/raw/deepbio_reference_atlas.parquet
    # We use RAW_PATH from config which points to E:/DeepBio_Scan/data/raw
    input_file = RAW_PATH / "deepbio_reference_atlas.parquet"
    
    if not input_file.exists():
        logger.error(f"CRITICAL: Input file missing at {input_file}")
        logger.error("Action Required: Copy 'deepbio_reference_atlas.parquet' to E:/DeepBio_Scan/data/raw/")
        return

    logger.info(f"Loading Reference Atlas from: {input_file}")
    
    try:
        # 2. Load Data
        df = pd.read_parquet(input_file)
        row_count = len(df)
        logger.info(f"Loaded {row_count} sequences into memory.")
        
        # Validation: Vector Dims
        vec_dim = 0
        if 'vector' in df.columns and row_count > 0:
            sample_vec = df.iloc[0]['vector']
            vec_dim = len(sample_vec)
            logger.info(f"Vector Dimension Detected: {vec_dim}")
        else:
            logger.error("Parquet file invalid: No 'vector' column or empty.")
            return

        # 3. Initialize LanceDB
        # User Request: Connect to E:/DeepBio_Scan/data/db
        logger.info(f"Connecting to LanceDB at: {DB_PATH}")
        db = lancedb.connect(DB_PATH)
        
        # 4. Create Table
        table_name = "reference_atlas"
        logger.info(f"Creating/Overwriting Table: '{table_name}'")
        tbl = db.create_table(table_name, data=df, mode="overwrite")
        
        # 5. Apply Indexing (IVF-PQ)
        # User Request: num_partitions=256, num_sub_vectors=96 (optimized for 768-dim)
        
        # Safety Check for Dimension Compatibility
        # IVF-PQ requires `dimension % num_sub_vectors == 0`
        sub_vectors = 96
        
        # Enforce User Spec but warn/adjust if impossible
        if vec_dim % 96 != 0:
            logger.warning(f"[Optimization Alert] 768-dim optimization requested (96 sub-vectors), but data is {vec_dim}-dim.")
            logger.warning(f"Optimization: {vec_dim} is NOT divisible by 96.")
            
            # Auto-Correction logic (Persona: Data-Ops is smart)
            if vec_dim == 512:
                sub_vectors = 64 # 512 / 8 = 64
                logger.info("Auto-Correcting: Using num_sub_vectors=64 for 512-dim compatibility.")
            else:
                # Fallback to safe default if completely unknown (divide by 8 is usually safe for power of 2 dims)
                sub_vectors = int(vec_dim / 8) 
                logger.info(f"Auto-Correcting: Using num_sub_vectors={sub_vectors}.")
        else:
             logger.info(f"Applying Target Optimization: num_sub_vectors={sub_vectors} (Perfect Match for 768-dim).")

        if row_count >= 256:
            logger.info(f"Building IVF-PQ Index on 'vector' column (partitions=256, sub_vectors={sub_vectors})...")
            tbl.create_index(
                metric="L2",
                vector_column_name="vector",
                num_partitions=256,
                num_sub_vectors=sub_vectors
            )
            logger.info("Index Built Successfully.")
        else:
            logger.warning("Dataset too small for Indexing (<256 rows). Skipping to prevent LanceDB error.")

        # 6. Final Confirmation
        total_vectors = tbl.count_rows() # Use LanceDB count for accuracy
        print(f"Total Vectors Live: {total_vectors}")
        logger.info(f"[SYSTEM] Atlas Synchronization Complete. Volume E is now a High-Performance Edge Node.")
        
    except Exception as e:
        logger.error(f"Seeding Failed: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure folders exist
    initialize_folders()
    seed_reference_atlas()
