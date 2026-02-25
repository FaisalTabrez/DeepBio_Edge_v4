import requests
import pandas as pd
import logging
import time
from pathlib import Path
from typing import List, Dict, Set, Optional
from configs.config import TAXON_DIR

# ==========================================
# @Data-Ops: Tier 2 Marine Oracle Seeder (Expanded)
# ==========================================
# EXPANDED CAPABILITY:
# - Targets WoRDSS (Deep-Sea) coverage via broad Taxon crawling
# - Includes 18S Targets: Protists, Foraminifera, Fungi
# - Implements Self-Healing Merge (Deduplication)
# - auto-resolves synonyms to 'Accepted' names
# - recurses 7-levels deep for lineage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Data-Ops")

class WormsSeederExpanded:
    def __init__(self):
        self.api_url = "https://www.marinespecies.org/rest"
        self.output_path = TAXONOMY_DB_PATH / "worms_deepsea_ref.csv"
        
        # Expanded Seed List for Deep Sea Coverage (Metazoa + 18S Eukaryotes)
        self.seed_taxa = [
            # --- 18S Eukaryote Targets ---
            "Foraminifera",     # Crucial for benthic 18S
            "Radiozoa",         # Radiolarians (Deep pelagic)
            "Cercozoa",         # Protists
            "Ciliophora",       # Ciliates
            "Fungi",            # Marine Fungi (Check depth?)
            
            # --- Deep Sea Metazoa (Invertebrates) ---
            "Hexactinellida",   # Glass Sponges
            "Holothuroidea",    # Sea Cucumbers (dominant in abyss)
            "Ophiuroidea",      # Brittle stars
            "Crinoidea",        # Sea lilies
            "Octocorallia",     # Soft corals
            "Scleractinia",     # Stony corals (Lophelia etc)
            "Actiniaria",       # Anemones
            "Pycnogonida",      # Sea spiders
            "Isopoda",          # Giant deep sea isopods
            "Amphipoda",        # Eurythenes etc.
            "Siboglinidae",     # Vent tubeworms
            "Alvinocarididae",  # Vent shrimp
            
            # --- Vertebrates ---
            "Macrouridae",      # Grenadiers
            "Ophidiidae",       # Cusk eels
            "Liparidae",        # Snailfish
            "Myxinidae",        # Hagfish
            "Chimaeriformes",   # Ghost sharks
            "Squaliformes"      # Deep sharks
        ]
        
        # State
        self.existing_ids: Set[int] = set()
        self.new_records: List[Dict] = []
        self.session = requests.Session()
        
    def load_existing(self):
        """Loads existing CSV to prevent duplicates (Self-Healing)."""
        if self.output_path.exists():
            try:
                df = pd.read_csv(self.output_path)
                if 'AphiaID' in df.columns:
                    self.existing_ids = set(df['AphiaID'].dropna().astype(int))
                logger.info(f"Loaded {len(self.existing_ids)} existing records from cache.")
            except Exception as e:
                logger.warning(f"Could not load existing DB, starting fresh. Error: {e}")
        else:
            logger.info("No existing DB found. Starting fresh.")

    def get_aphia_id(self, name: str) -> Optional[int]:
        """Resolves name to ID."""
        try:
            resp = self.session.get(f"{self.api_url}/AphiaIDByName/{name}")
            if resp.status_code == 200 and resp.text:
                return int(resp.json())
            elif resp.status_code == 204:
                 logger.warning(f"Taxon not found: {name}")
        except Exception as e:
            logger.error(f"Error resolving {name}: {e}")
        return None

    def fetch_node_recursive(self, aphia_id: int, depth: int = 0):
        """
        Recursively fetches children.
        Logic:
        - If rank is Species -> Process & Store (Base Case).
        - If rank is Family/Genus/Order -> Fetch Children & Recurse.
        - Depth Limit: Preventing infinite loops, though tax tree is finite.
        """
        # Hard stop to prevent runaway recursion in huge groups
        if depth > 5: 
            return

        try:
            url = f"{self.api_url}/AphiaChildrenByAphiaID/{aphia_id}?marine_only=true"
            resp = self.session.get(url)
            
            if resp.status_code == 200:
                children = resp.json()
                for child in children:
                    c_id = child['AphiaID']
                    c_rank = child['rank']
                    c_status = child['status']
                    
                    # 1. Deduplication Check
                    if c_id in self.existing_ids:
                        continue
                    
                    # 2. Process Species
                    if c_rank == 'Species':
                        if c_status == 'accepted':
                             self.process_species(child)
                             self.existing_ids.add(c_id)
                        elif c_status == 'unaccepted':
                             # Resolve synonym
                             valid_id = child.get('valid_AphiaID')
                             if valid_id and valid_id not in self.existing_ids:
                                 self.fetch_valid_record(valid_id)
                    
                    # 3. Recurse (Drill down)
                    # We drill down through Class, Order, Family, Genus
                    elif c_rank in ['Class', 'Order', 'Family', 'Genus', 'Subclass', 'Superfamily']:
                        # Simple rate limit
                        # time.sleep(0.02) 
                        self.fetch_node_recursive(c_id, depth + 1)
                        
        except Exception as e:
            logger.debug(f"Branch traversal issue for {aphia_id}: {e}")

    def fetch_valid_record(self, aphia_id: int):
        """Fetches the accepted record (resolves synonym)."""
        try:
            resp = self.session.get(f"{self.api_url}/AphiaRecordByAphiaID/{aphia_id}")
            if resp.status_code == 200:
                rec = resp.json()
                self.process_species(rec)
                self.existing_ids.add(aphia_id)
        except Exception:
            pass

    def process_species(self, record: Dict):
        """Extracts full 7-level lineage and appends to new_records."""
        # record usually has denormalized fields
        entry = {
            "ScientificName": record.get('scientificname'),
            "AphiaID": record.get('AphiaID'),
            "Kingdom": record.get('kingdom', 'Unknown'),
            "Phylum": record.get('phylum', 'Unknown'),
            "Class": record.get('class', 'Unknown'),
            "Order": record.get('order', 'Unknown'),
            "Family": record.get('family', 'Unknown'),
            "Genus": record.get('genus', 'Unknown'),
            "Rank": record.get('rank'),
            "Source": "WoRMS_DeepSea_Expanded"
        }
        self.new_records.append(entry)
        
        # Incremental Save
        if len(self.new_records) >= 50:
            self.save_results()
            self.new_records = [] # Flush buffer

    def save_results(self):
        if not self.new_records:
            return

        new_df = pd.DataFrame(self.new_records)
        columns = ["ScientificName", "AphiaID", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Rank", "Source"]
        
        # Reorder/Ensure cols
        new_df = new_df[columns]
        
        if self.output_path.exists():
            # Append mode
            new_df.to_csv(self.output_path, mode='a', header=False, index=False)
            logger.info(f"APPENDED {len(new_df)} species (Total: ~{len(self.existing_ids) + len(new_df)})")
        else:
            # Write mode
            if not self.output_path.parent.exists():
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(self.output_path, index=False)
            logger.info(f"CREATED {self.output_path} with {len(new_df)} species")
            
    def run(self):
        logger.info("Initializing Tier 2 Expansion (WoRMS Deep-Sea)...")
        self.load_existing()
        
        try:
            for taxon in self.seed_taxa:
                logger.info(f"Crawling Lineage: {taxon}")
                tax_id = self.get_aphia_id(taxon)
                if tax_id:
                    self.fetch_node_recursive(tax_id)
                time.sleep(0.5) # Politeness
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user. Saving remaining buffer...")
        except Exception as e:
            logger.error(f"Unexpected crash: {e}")
        finally:
            self.save_results() # Save any remainders
            logger.info("Expansion Paused/Complete.")

if __name__ == "__main__":
    seeder = WormsSeederExpanded()
    seeder.run()
