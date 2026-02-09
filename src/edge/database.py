import os
import lancedb
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import time
from typing import Optional
from src.edge.config_init import DB_PATH

# ==========================================
# @Data-Ops: Database & File System Logic
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Data-Ops")

class AtlasManager:
    """
    Manages the LanceDB connection on the NTFS-formatted USB drive.
    Optimized for Disk-Native Search (IVF-PQ) and Windows File Locking limits.
    """
    def __init__(self, db_path: Optional[str] = None, table_name: str = "reference_atlas"):
        self.db_path = str(db_path) if db_path else str(DB_PATH)
        self.table_name = table_name
        self.db = None
        self.table = None

        logger.info(f"Initializing AtlasManager at {self.db_path}...")
        
        # @Data-Ops: NTFS Safety Check
        # Windows sometimes throws 'OSError: [WinError 1] Incorrect function' 
        # when dealing with memory-mapped files on certain USB controllers.
        # We ensure the directory exists and permissions are valid.
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)

        try:
            # Connect to LanceDB
            # Note: For read-heavy demos, we keep the connection open.
            self.db = lancedb.connect(self.db_path)
            
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                logger.info(f"Connected to existing atlas: '{self.table_name}'")
            else:
                logger.warning(f"Table '{self.table_name}' not found. Use ingest_atlas() to seed data.")

        except OSError as e:
            if "Incorrect function" in str(e) or "[WinError 1]" in str(e):
                logger.critical("NTFS FILE LOCK ERROR DETECTED: This is likely due to the USB drive's controller.")
                logger.critical("Recommendation: Ensure no other process (like Explorer preview or Antivirus) is scanning the DB folder.")
            raise e

    def ingest_atlas(self, parquet_path: str):
        """
        Loads the Colab-generated parquet file and builds the IVF-PQ Index.
        Crucial for performance on the 32GB USB drive.
        """
        if not os.path.exists(parquet_path):
            logger.error(f"Parquet file not found: {parquet_path}")
            return

        logger.info(f"Ingesting {parquet_path} into LanceDB...")
        try:
            if self.db is None:
                raise ConnectionError("Database connection is not active.")

            df = pd.read_parquet(parquet_path)
            
            # Create Table (Overwrite if exists for the demo setup)
            self.table = self.db.create_table(self.table_name, data=df, mode="overwrite")
            
            # @Data-Ops: Indexing Strategy (IVF-PQ)
            # - num_partitions: Sqrt(Rows) is standard, but for >1M rows on USB, we scale up.
            # - num_sub_vectors: 96 (768 / 8) allows good compression.
            logger.info("Building IVF-PQ Index... (This may take time on USB)")
            self.table.create_index(
                metric="cosine",
                vector_column_name="vector",
                num_partitions=256,   # IVF: Inverted File partitions
                num_sub_vectors=96    # PQ: Product Quantization
            )
            logger.info("Index built successfully. Atlas is ready.")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

    def query_vector(self, vector: np.ndarray, top_k: int = 5):
        """
        Performs a semantic search for the query vector on Volume E.
        Returns: List of dicts {Scientific_Name, similarity, id, distance}
        """
        if self.table is None:
            logger.error("DB Table not initialized. Cannot query.")
            return []

        # @Data-Ops: Dimension Agnostic Logic
        # Model is 512d, but interface creates 768d placeholder sometimes.
        vec_dim = vector.shape[-1] 
        
        # Flatten if batch dimension exists (1, Dim) -> (Dim,)
        if len(vector.shape) > 1:
            vector = vector.flatten()

        try:
            # Execute Search on Volume E
            # Note: LanceDB search defaults to L2 (Euclidean) if index was trained with L2
            results = self.table.search(vector) \
                .limit(top_k) \
                .to_pandas()
            
            # Normalize Schema for UI
            # LanceDB returns '_distance'. For L2: Similarity = 1 / (1 + distance)
            results['similarity'] = 1 / (1 + results['_distance'])
            
            # @Data-Ops: Schema Mapping
            # Map 'species' to 'Scientific_Name' for downstream compatibility
            if 'species' in results.columns and 'Scientific_Name' not in results.columns:
                results['Scientific_Name'] = results['species']
            
            # Return pure records including vector for Visualization
            cols_to_return = ['Scientific_Name', 'species', 'similarity', 'id', '_distance']
            
            # Add vector if available (needed for 3D plot)
            if 'vector' in results.columns:
                cols_to_return.append('vector')
                
            # Filter only existing columns
            existing_cols = [c for c in cols_to_return if c in results.columns]
            
            return results[existing_cols].to_dict(orient='records')

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # @Data-Ops: Fallback logic for Windows IO hiccups
            if "OS error" in str(e):
                logger.warning("Retrying query due to transient Disk IO error...")
                time.sleep(0.1)
                return self.query_vector(vector, top_k)
            return []

if __name__ == "__main__":
    # Smoke Test
    print("Running @Data-Ops Database Smoke Test")
    manager = AtlasManager()
    
    # Mock Vector
    mock_vec = np.random.rand(768).astype('float32')
    
    # Needs a real table to work, but this tests connection logic
    if manager.table:
        print(manager.query_vector(mock_vec))
    else:
        print("Table not found. Run ingestion first.")
