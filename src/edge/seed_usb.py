"""
@Data-Ops: USB Seeding Script
Purpose: Ingests the Colab-generated Parquet atlas into LanceDB on the USB Stick.
Optimization: Applies IVF-PQ Indexing for low-latency searches on limited hardware.
"""
import pandas as pd
import lancedb
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.edge.config_init import DB_PATH, RAW_PATH, LOGS_PATH, initialize_folders

# Setup Logger specifically for this seeding job
logger = logging.getLogger("DeepBio-Seeder")
logger.setLevel(logging.INFO)

# Console Handler
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

# File Handler (Bridge to UI)
file_handler = logging.FileHandler(LOGS_PATH / 'session.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(message)s') # Simplified format for UI log reader
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def seed_reference_atlas():
    input_file = RAW_PATH / "deepbio_reference_atlas.parquet"
    
    # 1. Validation: Check if input exists
    if not input_file.exists():
        logger.error(f"[DB] Error: Source file not found at {input_file}")
        logger.error("[DB] Please copy 'deepbio_reference_atlas.parquet' from Colab to E:/DeepBio_Scan/data/raw/")
        return

    logger.info(f"[DB] Loading Reference Atlas from: {input_file}")
    
    try:
        # Load Data
        df = pd.read_parquet(input_file)
        row_count = len(df)
        logger.info(f"[DB] Loaded {row_count} sequences.")
        
        # Verify Vector Column
        if "vector" not in df.columns:
            logger.error("[DB] Critical Error: 'vector' column missing in Parquet.")
            return
            
        # Check Vector Dimension
        # Ensure vectors are numpy arrays of float32
        sample_vec = df.iloc[0]["vector"]
        vec_dim = len(sample_vec)
        logger.info(f"[DB] Vector Dimension Detected: {vec_dim}")

        # 2. Connect to LanceDB (Native Disk Mode)
        db = lancedb.connect(DB_PATH)
        
        # 3. Create Table
        table_name = "reference_atlas"
        logger.info(f"[DB] Creating Table: {table_name}")
        
        # LanceDB automatically infers schema from pandas DataFrame
        tbl = db.create_table(table_name, data=df, mode="overwrite")
        
        # 4. Create Index (IVF-PQ)
        # @Data-Ops Logic: 
        # Num_Partitions=256 (Good for <1M vectors)
        # Num_Sub_Vectors=96 (Optimized for 768d, need to check if 512d)
        
        metric_type = "L2"
        sub_vectors = 96 # Default requested by user
        
        # Robust Dimension Handling
        if vec_dim == 512:
             logger.warning("[DB] Auto-tuning: Adjusting sub_vectors to 64 for 512-dim model compatibility.")
             sub_vectors = 64
        elif vec_dim % 96 != 0:
             # Fallback if 768-assumption fails and it's not 512
             logger.warning(f"[DB] Warning: Vector dimension {vec_dim} not divisible by 96.")
        
        # Decision Logic: Indexing vs Flat Search
        if row_count < 256:
            logger.warning(f"[DB] Dataset too small for IVF-PQ Training ({row_count} < 256). Skipping index creation.")
            logger.info("[DB] Performance Note: Flat search will be used (Optimal for small datasets).")
        else:
            # Auto-tune partitions
            num_partitions = 256
            if row_count < (num_partitions * 2):
                 num_partitions = int(np.sqrt(row_count))
                 num_partitions = max(1, num_partitions)
                 logger.warning(f"[DB] Low Data Volume ({row_count} rows). Auto-adjusted partitions to {num_partitions}.")
            
            logger.info(f"[DB] Building IVF-PQ Index: Partitions={num_partitions}, SubVectors={sub_vectors}")
            
            tbl.create_index(
                metric=metric_type,
                vector_column_name="vector",
                num_partitions=num_partitions,
                num_sub_vectors=sub_vectors
            )
        
        # 5. Validation Check
        final_count = tbl.count_rows()
        logger.info(f"[DB] Seeding Complete. Index generated for Volume E.")
        logger.info(f"     Total Rows: {final_count}")
        
    except Exception as e:
        logger.error(f"[DB] Seeding Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Initializing Data Ops Procedure...")
    initialize_folders()
    seed_reference_atlas()
