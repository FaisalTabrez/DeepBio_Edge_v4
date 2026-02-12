import os
import sys
import logging
from pathlib import Path

# Fix module path for src imports
sys.path.append(os.getcwd())

from src.edge.config_init import RAW_PATH, DB_PATH
from src.edge.database import AtlasManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [@Data-Ops] - %(levelname)s - %(message)s")
logger = logging.getLogger("reinit_db")

def reinit_database():
    """
    Utility script to wipe and re-initialize the local LanceDB instance 
    from a source parquet file.
    """
    logger.info("Starting Database Re-initialization Sequence...")

    # 1. Path Configuration
    parquet_filename = "deepbio_reference_atlas.parquet"
    parquet_path = RAW_PATH / parquet_filename
    
    logger.info(f"Target Parquet: {parquet_path}")
    logger.info(f"Target Database: {DB_PATH}")

    if not parquet_path.exists():
        logger.error(f"CRITICAL: Reference Atlas Parquet not found at {parquet_path}")
        logger.error("Please ensure the file is placed in 'data/raw/' before running this script.")
        return

    # 2. Database Connection & Setup
    try:
        # AtlasManager handles connection logic to DB_PATH
        atlas = AtlasManager(db_path=str(DB_PATH))
        
        # 3. Drop & Recreate (Ingestion)
        # The ingest_atlas method in database.py already implements:
        # - overwrite mode (Drop if exists)
        # - IVF-PQ Indexing with num_partitions=256, num_sub_vectors=96
        logger.info("Dropping existing tables and ingesting new data...")
        atlas.ingest_atlas(str(parquet_path))
        
        # 4. Explicit Verification
        logger.info("Verifying Database Integrity...")
        
        if atlas.table:
            row_count = atlas.table.count_rows()
            logger.info(f"✅ Database Live. Total Rows: {row_count}")
            
            # Fetch first 5
            logger.info("--- Sample Data Preview (First 5) ---")
            df_head = atlas.table.head(5).to_pandas()
            
            # Check for standard columns or fallbacks
            name_col = 'Scientific_Name' if 'Scientific_Name' in df_head.columns else 'species'
            
            if name_col in df_head.columns:
                print(df_head[[name_col]].to_string(index=False))
            else:
                print(df_head.columns)
                
            logger.info("--- End Preview ---")
            logger.info("Database Re-initialization Complete. Ready for Inference.")
            
        else:
            logger.error("Table creation failed. Table object is None.")

    except Exception as e:
        logger.error(f"Database Initialization Failed: {e}", exc_info=True)

if __name__ == "__main__":
    reinit_database()
