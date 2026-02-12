import os
import sys
import logging
import random
from time import sleep

# Ensure project root is in path
sys.path.append(os.getcwd())

from Bio import Entrez, SeqIO
from pathlib import Path
from src.edge.config_init import RAW_PATH
from src.edge.logger import get_logger

# ==========================================
# @Data-Ops: NCBI Fetch Routine
# ==========================================
# Purpose: acquire real-world environmental eDNA for testing.
# Targets both specific species (Ground Truth) and mystery sequences (Discovery).

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = get_logger()

# Configure Entrez
Entrez.email = "deepbio_bot@example.com"  # Set a dummy email for Identification
Entrez.api_key = os.getenv("NCBI_API_KEY", None) # Optional

def fetch_deep_sea_samples():
    logger.info("Initiating Bio.Entrez Connection to NCBI Nucleotide Database...")
    
    # 1. Search Query
    # EXTREMELY BROAD Search to ensure we get > 100 hits
    search_term = '("marine"[All Fields] OR "ocean"[All Fields]) AND ("18S"[All Fields] OR "small subunit"[All Fields]) AND "eukaryota"[Organism] AND 500:2000[Sequence Length]'
    
    try:
        # Search for IDs first
        logger.info(f"Querying: {search_term}")
        # RetMax increased significantly to allow for aggressive filtering later
        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=10000, idtype="acc")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        if not id_list:
             logger.error("Zero results even with broad query.")
             return

        logger.info(f"Found {len(id_list)} candidate sequences. Filtering for quality (300-900bp)...")

        # 2. Fetch Details (Batch)
        # Fetch in chunks if extensive list, but 500 is ok for one call usually, limit to 400 for safety
        ids_to_fetch = id_list[:800]
        
        # efetch documentation recommends rettype="fasta"
        try:
             # fetch full records to get length and desc
             handle = Entrez.efetch(db="nucleotide", id=ids_to_fetch, rettype="fasta", retmode="text")
             logger.info("Fetched batch from NCBI. Parsing...")
        except Exception as e:
             logger.error(f"Efetch Error: {e}")
             return
        
        # 3. Stream & Filter
        known_taxa = []
        environmental_taxa = []
        
        # Suppress Biopython warning by just iterating through handle directly first? 
        # Actually just use fasta-pearson if needed, but standard fasta usually works.
        # The warning is about comments. Let's filter lines starting with # or ! manually if needed, 
        # but SeqIO handles it mostly.
        
        for record in SeqIO.parse(handle, "fasta"):
            seq_len = len(record.seq)
            desc = record.description.lower()
            
            # Length Filter (Broadened slightly)
            if 300 <= seq_len <= 3000:
                
                # Check for "uncultured", "environmental sample", "clone"
                is_env = any(k in desc for k in ["uncultured", "environmental sample", "clone", "metagenome"])
                
                # Clean description
                clean_desc = record.description.replace(record.id, "").strip()
                if not clean_desc: clean_desc = "Unknown Eukaryote"

                entry = f">{record.id} | {clean_desc} | {seq_len}bp\n{str(record.seq).upper()}"
                
                if is_env:
                    environmental_taxa.append(entry)
                else:
                    known_taxa.append(entry)

        handle.close()
        
        # 4. Construct the Final Dataset (60/40 Split)
        # Targets
        total_target = 150
        target_known = int(total_target * 0.6) # 90
        target_env = int(total_target * 0.4)   # 60
        
        # Slice
        final_known = known_taxa[:target_known]
        final_env = environmental_taxa[:target_env]
        
        logger.info(f"Selection Stats: Known={len(final_known)}, Env={len(final_env)}")
        
        combined = final_known + final_env
        random.shuffle(combined)
        
        # 5. Write to File
        output_path = RAW_PATH / "Expedition_DeepSea_Batch.fasta"
        
        with open(output_path, "w") as f:
            f.write("\n".join(combined))
                
        logger.info(f"✅ SUCCESSFULLY WROTE {len(combined)} SEQUENCES TO {output_path}")
        logger.info("Dataset ready for Colab Embedding phase.")

    except Exception as e:
        logger.error(f"NCBI Fetch Failed: {e}", exc_info=True)

if __name__ == "__main__":
    fetch_deep_sea_samples()
