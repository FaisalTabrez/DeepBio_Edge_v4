import os
import time
import pyarrow as pa
import pyarrow.dataset as ds
import lancedb
import numpy as np

# @Data-Ops: Phase 2 - LanceDB Vector Indexing (100k Scale)
print("Initializing DeepBio-Scan Vector Database (100k Scale)...")

# 1. Data Load
DATA_DIR = r"E:\DeepBio_Scan\data\raw"
DB_DIR = r"E:\DeepBio_Scan\data\db"
PARQUET_FILE = os.path.join(DATA_DIR, "reference_atlas_100k.parquet")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

print(f"Loading Reference Atlas from: {PARQUET_FILE}")
if not os.path.exists(PARQUET_FILE):
    raise FileNotFoundError(f"Could not find {PARQUET_FILE}. Please ensure the Colab export is placed here.")

# Use PyArrow Dataset for memory-efficient streaming instead of loading 3.7GB into Pandas
dataset = ds.dataset(PARQUET_FILE, format="parquet")
print(f"Dataset loaded. Schema:\n{dataset.schema}")

# 2. LanceDB Table Creation
print(f"Connecting to LanceDB at: {DB_DIR}")
db = lancedb.connect(DB_DIR)

table_name = "reference_atlas_v100k"
print(f"Creating/Overwriting table: {table_name} (Streaming data to disk...)")

# Define the strict schema required by LanceDB (Fixed Size List for Vectors)
schema = pa.schema([
    pa.field("AccessionID", pa.string()),
    pa.field("ScientificName", pa.string()),
    pa.field("TaxID", pa.string()),
    pa.field("Phylum", pa.string()),
    pa.field("Class", pa.string()),
    pa.field("Order", pa.string()),
    pa.field("Family", pa.string()),
    pa.field("Genus", pa.string()),
    pa.field("Sequence", pa.string()),
    pa.field("Quality_Check", pa.bool_()),
    pa.field("Vector", pa.list_(pa.float32(), 768)) # Fixed size list
])

def batch_generator():
    for batch in dataset.to_batches(batch_size=10000):
        # Cast the variable-length list to a fixed-size list of 768 floats
        arrays = []
        for name in schema.names:
            if name == "Vector":
                # Convert to fixed size list
                fixed_list_array = pa.FixedSizeListArray.from_arrays(
                    batch[name].values, 768
                )
                arrays.append(fixed_list_array)
            else:
                arrays.append(batch[name])
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)

# Create table using the generator
table = db.create_table(table_name, data=batch_generator(), schema=schema, mode="overwrite")
print(f"Table '{table_name}' created successfully. Total records: {table.count_rows()}")

# 3. IVF-PQ Indexing (Crucial for 100k)
print("Building IVF-PQ Index on 'Vector' column...")
print("Parameters: num_partitions=128, num_sub_vectors=96")
start_time = time.time()

# Create the index
# This reduces search complexity from O(N) to O(sqrt(N))
table.create_index(
    metric="cosine", 
    vector_column_name="Vector", 
    num_partitions=128, 
    num_sub_vectors=96
)

index_time = time.time() - start_time
print(f"Indexing complete. Total time taken: {index_time:.2f} seconds.")

# 4. Diagnostic
print("Running diagnostic test search...")
# Generate a random query vector of the correct shape
query_vector = np.random.rand(768).astype(np.float32)

search_start = time.time()
results = table.search(query_vector).limit(5).to_pandas()
search_time = (time.time() - search_start) * 1000 # Convert to ms

print(f"\n[SUCCESS] 100k Atlas Live. Mean Latency: {search_time:.2f}ms.")
print("\nTop 5 Test Results (Random Query):")
print(results[["AccessionID", "ScientificName", "_distance"]])
