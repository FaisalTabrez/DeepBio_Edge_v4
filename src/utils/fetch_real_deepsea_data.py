import os
import time
import logging
from pathlib import Path
from Bio import Entrez, SeqIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepSea-Fetcher')

# Set Entrez email
Entrez.email = "faisaltabrez01@gmail.com"

# Define output directory
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_sequences(query: str, max_results: int, category: str) -> list:
    """
    Fetches sequences from NCBI Nucleotide database based on a query.
    Filters for sequence length between 300bp and 800bp.
    """
    logger.info(f"Searching NCBI for: {category}...")
    try:
        # Search for the query
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_results * 3) # Fetch more to account for filtering
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        if not id_list:
            logger.warning(f"No sequences found for {category}.")
            return []
            
        logger.info(f"Found {len(id_list)} potential matches. Fetching and filtering...")
        
        # Fetch the actual sequences
        fetch_handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
        records = list(SeqIO.parse(fetch_handle, "fasta"))
        fetch_handle.close()
        
        # Filter by length
        filtered_records = []
        for rec in records:
            seq_len = len(rec.seq)
            if 300 <= seq_len <= 800:
                # Clean up description to ensure Scientific Name is prominent if possible
                # For known taxa, the query usually ensures the name is in the description
                filtered_records.append(rec)
                if len(filtered_records) >= max_results:
                    break
                    
        logger.info(f"Successfully fetched {len(filtered_records)} sequences for {category}")
        return filtered_records
        
    except Exception as e:
        logger.error(f"Error fetching {category}: {e}")
        return []

def generate_known_taxa():
    """
    Fetches 10 high-quality 18S sequences for specific deep-sea taxa.
    """
    taxa_list = [
        "Bathymodiolus septemdierum", # Vent Mussel
        "Bathynomus giganteus",       # Giant Isopod
        "Riftia pachyptila",          # Giant Tube Worm
        "Grimpoteuthis",              # Dumbo Octopus
        "Macrouridae"                 # Grenadier Fish
    ]
    
    all_records = []
    for taxon in taxa_list:
        # Query specifically for 18S rRNA
        query = f"{taxon}[Organism] AND (18S[All Fields] OR small subunit ribosomal RNA[All Fields])"
        # We want 2 per taxon to get 10 total (or just try to get 10 total across all)
        # Let's fetch 2 per taxon to ensure diversity
        records = fetch_sequences(query, max_results=2, category=taxon)
        all_records.extend(records)
        time.sleep(1) # Respect NCBI rate limits
        
    output_file = RAW_DATA_DIR / "known_taxa.fasta"
    SeqIO.write(all_records, output_file, "fasta")
    logger.info(f"Successfully fetched {len(all_records)} sequences for Known Taxa")

def generate_dark_taxa():
    """
    Fetches 15 recent environmental/uncultured 18S sequences.
    """
    query = "(deep sea[All Fields]) AND (uncultured eukaryote[All Fields]) AND (18S[All Fields])"
    
    records = fetch_sequences(query, max_results=15, category="Dark Taxa (Uncultured Eukaryotes)")
    
    output_file = RAW_DATA_DIR / "discovery_dark_taxa.fasta"
    SeqIO.write(records, output_file, "fasta")
    logger.info(f"Successfully fetched {len(records)} sequences for Dark Taxa")

if __name__ == "__main__":
    logger.info("Starting Deep-Sea Data Fetcher...")
    generate_known_taxa()
    time.sleep(1)
    generate_dark_taxa()
    logger.info("Data fetching complete.")
