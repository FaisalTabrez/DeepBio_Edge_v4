import os
import time
import logging
import sys
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add root to path so we can import config
sys.path.append(os.getcwd())

from configs.config import RAW_PATH

# ==========================================
# @Data-Ops & @Bio-Taxon: Real Data Acquisition
# ==========================================

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NCBI-Fetcher')

class NCBIExpeditionFetcher:
    """
    Automated agent to fetch, curate, and format deep-sea genomic data 
    for the Global-BioScan expedition simulation.
    """
    def __init__(self, email="biologist@deepbio.org"):
        # NCBI requires an email address
        Entrez.email = email
        self.output_file = RAW_PATH / "Expedition_DeepSea_Batch.fasta"
        
        # Ensure raw directory exists
        if not RAW_PATH.exists():
            RAW_PATH.mkdir(parents=True, exist_ok=True)

    def search_sequences(self, term: str, retmax: int = 50) -> list:
        """
        Searches NCBI Nucleotide database using specific terms.
        """
        logger.info(f"Searching @Bio-Taxon targets for term: '{term}'...")
        try:
            handle = Entrez.esearch(db="nucleotide", term=term, retmax=retmax, idtype="acc")
            # Entrez.read returns a Dict[str, Any] but usage implies checking the structure
            record = Entrez.read(handle)
            handle.close()
            
            # Type guard for mypy / strict checking
            if isinstance(record, dict) and "IdList" in record:
                ids = record["IdList"] # Returns List[str]
            else:
                 ids = []
                 
            logger.info(f"Found {len(ids)} candidates.")
            return ids
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def fetch_and_curate(self, id_list: list) -> list:
        """
        Fetches full GenBank records to extract metadata (Depth, Environment).
        Converts to FASTA with enriched headers.
        """
        if not id_list:
            return []
            
        curated_records = []
        batch_size = 10 # Respect API limits
        
        logger.info(f"Fetching metadata for {len(id_list)} sequences...")
        
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i+batch_size]
            try:
                # Fetch GenBank format to get features like 'depth' or 'isolation_source'
                handle = Entrez.efetch(db="nucleotide", id=batch_ids, rettype="gb", retmode="text")
                records = SeqIO.parse(handle, "genbank")
                
                for record in records:
                    # @Bio-Taxon: Metadata Extraction Logic
                    depth = "Unknown Depth"
                    desc = record.description
                    
                    # Scan features for ecological data
                    for feature in record.features:
                        if feature.type == "source":
                            quals = feature.qualifiers
                            if "depth" in quals:
                                depth = f"{quals['depth'][0]}m"
                            elif "isolation_source" in quals:
                                # Sometimes depth is hidden here
                                src = quals['isolation_source'][0]
                                if "m" in src and any(c.isdigit() for c in src):
                                    depth = src
                    
                    # @Data-Ops: Header Formatting for Demo
                    # Format: >Accession | Species Name | Depth | Description
                    clean_species = record.annotations.get("organism", "Unknown Organism").replace(" ", "_")
                    new_header = f"{record.id} | {clean_species} | Depth:{depth}"
                    
                    # Create clean SeqRecord
                    # Filter length 100-500bp strictly here if search query didn't catch it
                    if 100 <= len(record.seq) <= 2000: # Expanded slightly for safety
                        new_rec = SeqRecord(
                            record.seq,
                            id=new_header,
                            description="" # Clear description to avoid duplication in header
                        )
                        curated_records.append(new_rec)
                
                handle.close()
                time.sleep(0.5) # Be polite to NCBI servers
                
            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                
        return curated_records

    def run_expedition(self):
        """
        Orchestrates the data gathering for the 'Real Data' demo.
        """
        all_records = []
        
        # 1. @Bio-Taxon: Known Deep-Sea Species (Hydrothermal Vents)
        # Search for Bathymodiolus (Mussels) and Riftia (Tube worms) COI genes
        query_known = (
            "(Bathymodiolus[Organism] OR Riftia[Organism]) AND COI[Gene] "
            "AND 100:600[Sequence Length]"
        )
        ids_known = self.search_sequences(query_known, retmax=50)
        records_known = self.fetch_and_curate(ids_known)
        all_records.extend(records_known)
        logger.info(f"Curated {len(records_known)} 'Known' specimen sequences.")

        # 2. @Bio-Taxon: Environmental Dark Taxa (Abyssal Plain)
        # Search for unclassified eukaryotes in deep sea sediment (18S)
        query_env = (
            "(Deep sea[All Fields] AND 18S rRNA[Gene]) "
            "AND (environmental samples[Organism] OR uncultured eukaryote[Organism]) "
            "AND 100:600[Sequence Length]"
        )
        ids_env = self.search_sequences(query_env, retmax=50)
        records_env = self.fetch_and_curate(ids_env)
        all_records.extend(records_env)
        logger.info(f"Curated {len(records_env)} 'Environmental' sample sequences.")

        # 3. @Data-Ops: Save to Disk
        if all_records:
            count = SeqIO.write(all_records, self.output_file, "fasta")
            logger.info(f"[SUCCESS] Expedition Complete. Generated {count} validated sequences.")
            logger.info(f"Data stored at: {self.output_file}")
            
            # Print a preview for the user
            print("\n--- SAMPLE PREVIEW ---")
            for rec in all_records[:3]:
                print(f">{rec.id}")
                print(f"{rec.seq[:50]}...")
            print("----------------------")
        else:
            logger.warning("Expedition returned empty-handed. Check internet connection or query parameters.")

if __name__ == "__main__":
    fetcher = NCBIExpeditionFetcher()
    fetcher.run_expedition()
