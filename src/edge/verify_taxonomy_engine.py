"""
@Bio-Taxon: Triple-Tier Verification Script
Purpose: Validates the Consensus -> Oracle -> Fallback logic using a real/simulated sequence.
Target: Volume E (LanceDB) + Local Embedder
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.edge.embedder import NucleotideEmbedder
from src.edge.database import AtlasManager
from src.edge.taxonomy import TaxonomyEngine
from src.edge.parser import stream_sequences

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Bio-Taxon-Test")

def verify_triple_tier():
    logger.info("Initializing Edge Biological Engines...")
    
    # 1. Initialize Components
    try:
        embedder = NucleotideEmbedder() # Uses CPU/Quantized model
        atlas = AtlasManager()         # Connects to E:/DeepBio_Scan/data/db
        taxonomy = TaxonomyEngine()    # Connects to WoRMS/TaxonKit
    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return

    # 2. Load Sequence
    # We look for the file or create a dummy one for the test
    fasta_path = Path("Expedition_DeepSea_Batch.fasta")
    
    # If file missing, create a mock 'Bathymodiolus-like' sequence for testing
    if not fasta_path.exists():
        logger.warning(f"{fasta_path} not found. Creating mock Bio-Probe...")
        with open(fasta_path, "w") as f:
            # A short synthetic sequence (in reality, would be 500bp+)
            # Using a placeholder strong enough to generate a vector
            f.write(">BioProbe_001 | Deep Sea Vent | Bathymodiolus sp.\n")
            f.write("AGCT" * 100 + "\n") # 400bp generic
    
    sequences = list(stream_sequences(str(fasta_path)))
    if not sequences:
        logger.error("No sequences found in batch.")
        return

    target_seq = sequences[0]
    logger.info(f"Loaded Probe: {target_seq['id']}")
    
    # 3. Generate Embedding
    logger.info("Generating Vector Embedding (768-dim)...")
    # Note: embed_sequences returns a list or numpy array
    vectors = embedder.embed_sequences([target_seq['sequence']])
    
    # Numpy array truth ambiguity fix
    if vectors is None or len(vectors) == 0:
         logger.error("Embedding generation failed.")
         return
    
    query_vector = vectors[0]

    # 4. Search Atlas (LanceDB)
    logger.info("Querying Global Atlas (Volume E)...")
    search_results = atlas.query_vector(query_vector, top_k=5)
    
    if not search_results:
        logger.warning("No hits found in Atlas. Is the DB seeded?")
        # Creating mock results to verify Taxonomy Logic if DB is empty
        logger.info("Injecting Synthetic Hits for Logic Verification...")
        search_results = [
            {"Scientific_Name": "Bathymodiolus thermophilus", "similarity": 0.98, "phylum": "Mollusca"},
            {"Scientific_Name": "Bathymodiolus sp.", "similarity": 0.95, "phylum": "Mollusca"},
            {"Scientific_Name": "Bathymodiolus septemdierum", "similarity": 0.92, "phylum": "Mollusca"},
            {"Scientific_Name": "Gigantopelta chessoia", "similarity": 0.85, "phylum": "Mollusca"},
            {"Scientific_Name": "Unknown Abyssal Mussel", "similarity": 0.80, "phylum": "Mollusca"}
        ]
    
    logger.info(f"Top Hit: {search_results[0].get('Scientific_Name')} (Sim: {search_results[0].get('similarity'):.2f})")

    # 5. Taxonomy Resolution (Triple-Tier)
    logger.info("Executing Triple-Tier Resolution...")
    # Passing the matches to the engine
    # Note: 'format_search_results' calls 'resolve_identity' internally for the top hit usually,
    # but here we call 'resolve_identity' directly as requested to see the raw Identity Card.
    
    # We might need to mock metadata if the search results are raw vectors
    # (The Atlas query usually returns metadata dicts).
    
    resolution = taxonomy.resolve_identity(search_results)
    
    # 6. Verification Output
    print("\n" + "="*40)
    print("   🧬 DEEPBIO TAXONOMY VERIFICATION")
    print("="*40)
    print(f"Query ID      : {target_seq['id']}")
    print(f"Resolved Name : {resolution['display_name']}")
    print(f"Status        : {resolution['status']}")
    print(f"Confidence    : {resolution['confidence']:.2%}")
    print(f"Lineage       : {resolution.get('lineage')}")
    print(f"Novelty       : {'DETECTED' if resolution.get('is_novel') else 'Negative'}")
    
    if 'worms_id' in resolution:
        print(f"WoRMS AphiaID : {resolution['worms_id']} (Oracle Verified)")
    else:
        print("WoRMS AphiaID : Not Found (Check Internet/DB)")

    print("="*40 + "\n")
    
    if resolution['confidence'] > 0.8:
        logger.info("✅ SUCCESS: High-confidence identification achieved.")
    else:
        logger.warning("⚠️ ALERT: Low confidence result. Potential Dark Taxon or DB Mismatch.")

if __name__ == "__main__":
    verify_triple_tier()
