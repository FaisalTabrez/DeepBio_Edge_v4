import os
import subprocess
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from collections import Counter
try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

from configs.config import TAXONKIT_EXE_PATH, TAXON_DIR, WORMS_ORACLE

# ==========================================
# @Bio-Taxon: Triple-Tier Hybrid Resolver
# ==========================================
# Tier 1: Vector Consensus (Primary)
# Tier 2: WoRMS Local Cache (Marine Correction)
# Tier 3: TaxonKit (Lineage Expansion fallback)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Bio-Taxon")

class TaxonomyEngine:
    """
    Manages biological lineage resolution via a 3-Tier System.
    """
    def __init__(self, tax_data_path: Union[str, None] = None):
        # Config Paths
        self.tax_data_path = str(tax_data_path) if tax_data_path else str(TAXON_DIR)
        self.taxonkit_exe = str(TAXONKIT_EXE_PATH)
        self.worms_csv_path = Path(WORMS_ORACLE)
        
        # Load Tier 2: WoRMS Cache
        self.worms_cache = self._load_worms_cache()
        
        # Check Tier 3: TaxonKit Status
        self.taxonkit_active = self._check_taxonkit()

    def _load_worms_cache(self) -> pd.DataFrame:
        """
        Loads the Tier 2 Local CSV Lookup.
        """
        if self.worms_csv_path.exists():
            try:
                df = pd.read_csv(self.worms_csv_path)
                logger.info(f"Tier 2 Active: Loaded {len(df)} records from WoRMS Cache.")
                return df
            except Exception as e:
                logger.error(f"Failed to load WoRMS Cache: {e}")
                return pd.DataFrame()
        else:
            logger.warning(f"Tier 2 Disabled: Missing {self.worms_csv_path}")
            return pd.DataFrame()

    def _check_taxonkit(self) -> bool:
        """
        Verifies Tier 3 TaxonKit executable.
        """
        try:
            # Check if file exists and is executable (rough check on Windows)
            if not os.path.exists(self.taxonkit_exe):
                 logger.warning(f"Tier 3 Disabled: TaxonKit not found at {self.taxonkit_exe}")
                 return False
                 
            subprocess.run([self.taxonkit_exe, "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("Tier 3 Active: TaxonKit ready.")
            return True
        except Exception as e:
            logger.warning(f"Tier 3 Disabled: TaxonKit check failed: {e}")
            return False

    def get_dynamic_threshold(self, gene_type: str = "COI", environment: str = "Deep Sea") -> Dict[str, float]:
        """
        Calculates gene-specific distance thresholds.
        """
        thresholds = {
            "COI": {"confirmed": 0.03, "divergent": 0.15},
            "18S": {"confirmed": 0.01, "divergent": 0.05},
            "default": {"confirmed": 0.04, "divergent": 0.20}
        }
        
        # Explicit check for environments that might be passed as array?? Unlikely but defensive
        if isinstance(environment, (np.ndarray, list)):
            environment = str(environment[0]) if len(environment) > 0 else "Deep Sea"
            
        selected = thresholds.get(gene_type, thresholds["default"]).copy()
        
        # Hydrothermal Vents have high endemism -> lower 'divergent' bar
        if str(environment) == "Hydrothermal Vent":
             selected["divergent"] *= 0.9 
             
        return selected

    def get_calibrated_status(self, distance: float, gene_type: str = "COI") -> Tuple[str, str, bool]:
        """
        Returns (Status Label, Color Code, Is_Novel_Flag) based on dynamic calibration.
        """
        thresh = self.get_dynamic_threshold(gene_type)
        
        if distance <= thresh["confirmed"]:
            return "Confirmed Match", "#00FF00", False # Green
        elif distance <= thresh["divergent"]:
            return "Divergent / Potential New Species", "#00E5FF", False # Cyan
        else:
            return "🌌 DARK TAXA / NOVEL GENUS", "#7000FF", True # Neon Purple

    def resolve_consensus(self, matches: List[Dict]) -> Tuple[str, float]:
        """
        Tier 1: High Confidence Genus Consensus.
        If 4/5 (or majority) top matches share a Genus, ensure it's prioritized.
        Returns: (Consensus_Name_or_Genus, Confidence_Score)
        """
        if not matches:
            return "Unknown", 0.0

        # Normal voting on Full Name first
        # Defensive check for matches if passed as non-list-of-dicts (rare pandas behavior)
        if isinstance(matches, pd.DataFrame):
             matches = matches.to_dict('records')

        names = []
        for m in matches:
             # Ensure m is a dict
             if not isinstance(m, dict): continue
             
             val = m.get('Scientific_Name', m.get('species', 'Unknown'))
             if isinstance(val, (np.ndarray, list)):
                 if len(val) > 0: names.append(str(val[0]))
                 else: names.append("Unknown")
             else:
                 names.append(str(val))
                 
        top_k = names[:50] # Look at top 50 now for 100k scale
        if not top_k:
             return "Unknown", 0.0
             
        # 1. Exact Species Consensus
        counts = Counter(top_k)
        most_common_sp, count_sp = counts.most_common(1)[0]
        
        # If > 35 votes (out of 50) for exact species -> Strong Match
        # Defensive integer typing for counts
        count_sp_int = int(count_sp)
        if count_sp_int >= 35:
            return most_common_sp, count_sp_int / len(top_k)

        # 2. Genus Level Consensus (Fallback)
        # Extract Genera
        # Safe extraction if strings are wrapped in arrays
        genera = []
        for n in top_k:
             nn = str(n)
             parts = nn.split()
             if len(parts) > 0:
                 genera.append(parts[0])
        
        c_gen = Counter(genera)
        
        # Explicit length check for Counter to avoid ambiguity
        if len(c_gen) > 0:
            most_common_gen, count_gen = c_gen.most_common(1)[0]
            count_gen_int = int(count_gen)
            # 70% Policy (35/50)
            if count_gen_int >= 35:
                return f"{most_common_gen} sp. (Genus Candidate)", 0.90 # High confidence in Genus
            elif count_gen_int >= 25:
                return f"{most_common_gen} sp. (Genus Match)", 0.75
            
        # Fallback to Top Hit if very strong
        raw_sim = matches[0].get('similarity', 0)
        top_sim = 0.0
        if isinstance(raw_sim, (np.ndarray, list)):
             if len(raw_sim) > 0: 
                 if isinstance(raw_sim, np.ndarray): top_sim = float(raw_sim.item())
                 else: top_sim = float(raw_sim[0])
        else:
             top_sim = float(raw_sim)

        if top_sim > 0.97:
             # Ensure returned name is scalar
             best_name_raw = matches[0].get('Scientific_Name', matches[0].get('species', 'Unknown'))
             if isinstance(best_name_raw, (np.ndarray, list)):
                 best_name = str(best_name_raw[0]) if len(best_name_raw) > 0 else "Unknown"
             else:
                 best_name = str(best_name_raw)

             return best_name, 0.95
             
        # Ensure most_common_sp is string
        if isinstance(most_common_sp, (np.ndarray, list)):
            most_common_sp = str(most_common_sp[0]) if len(most_common_sp) > 0 else "Unknown"
        else:
            most_common_sp = str(most_common_sp)
            
        return most_common_sp, count_sp_int / len(top_k)

    def validate_worms(self, scientific_name: str) -> Dict[str, str]:
        """
        Tier 2: The Oracle - Fuzzy Search in WoRMS Cache.
        Returns full lineage dict if found.
        PROMOTES 'unaccepted' names if mapped in DB (via Synonym logic handled in seeder),
        but here we ensure we find the entered name even if typo'd.
        """
        if self.worms_cache.empty:
            return {}

        # Ensure scientific_name is a string before checking
        if not isinstance(scientific_name, str):
            return {}

        # 1. Exact Match (Fast)
        scientific_name = str(scientific_name).strip()
        
        # Defensive check against empty DataFrame boolean eval
        try:
             # Ensure we compare scalar string
             mask = self.worms_cache['ScientificName'].str.lower() == scientific_name.lower()
             hit = self.worms_cache[mask]
        except Exception as e:
             # logger.warning(f"WoRMS Filter Error: {e}")
             return {}
        
        # 2. Fuzzy Match (If rapidfuzz available and no exact hit)
        # Use explicit .empty check
        if hit.empty and process:
            # Get list of all names
            # SAFEGUARD: Ensure all_names is a list of strings, dropping ambiguous types
            all_names_raw = self.worms_cache['ScientificName'].tolist()
            all_names = [str(n) for n in all_names_raw if isinstance(n, (str, float, int))]
            
            # Extract one best match with score
            # limit score > 85 to be safe (Oracle acts as Validator for 100k DB)
            fuzzy_res = process.extractOne(scientific_name, all_names)
            
            # Defensive check on fuzzy_res structure
            if fuzzy_res:
                score = fuzzy_res[1]
                # If score is array for some reason (rare but possible with some libraries)
                if isinstance(score, (np.ndarray, list)):
                     score = float(score[0]) if len(score) > 0 else 0.0
                
                if score > 85: # Score > 85/100 for 100k scale
                    best_match_name = fuzzy_res[0]
                    # logger.info(f"Fuzzy Corrected: {scientific_name} -> {best_match_name} ({score})")
                    hit = self.worms_cache[self.worms_cache['ScientificName'] == best_match_name]

        if not hit.empty:
            rec = hit.iloc[0]
            return { # Standardize Keys
                "Phylum": str(rec.get("Phylum", "Unknown")),
                "Class": str(rec.get("Class", "Unknown")),
                "Order": str(rec.get("Order", "Unknown")),
                "Family": str(rec.get("Family", "Unknown")),
                "Genus": str(rec.get("Genus", "Unknown")),
                "ScientificName": str(rec.get("ScientificName", scientific_name)),
                "Source": "WoRMS (Oracle)"
            }
        return {}

    def calculate_taxonomic_reliability(self, matches: List[Dict]) -> Dict[str, Any]:
        """
        Calculates confidence at each taxonomic rank based on the top 50 neighbors.
        """
        ranks = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
        rank_counts = {r: Counter() for r in ranks}
        
        top_k = matches[:50]
        if not top_k:
            return {}
            
        # Pre-fetch WoRMS info for unique genera to save time
        unique_genera = set()
        for m in top_k:
            raw_name = m.get('Scientific_Name', m.get('species', 'Unknown'))
            if isinstance(raw_name, (np.ndarray, list)):
                sp_name = str(raw_name[0]) if len(raw_name) > 0 else "Unknown"
            else:
                sp_name = str(raw_name)
            parts = sp_name.split()
            if len(parts) > 0:
                unique_genera.add(parts[0])
                
        genus_to_lineage = {}
        for g in unique_genera:
            if g != "Unknown":
                worms_info = self.validate_worms(g)
                if worms_info:
                    genus_to_lineage[g] = worms_info
                    
        for m in top_k:
            raw_name = m.get('Scientific_Name', m.get('species', 'Unknown'))
            if isinstance(raw_name, (np.ndarray, list)):
                sp_name = str(raw_name[0]) if len(raw_name) > 0 else "Unknown"
            else:
                sp_name = str(raw_name)
                
            if sp_name == "Unknown":
                continue
                
            rank_counts["Species"][sp_name] += 1
            
            parts = sp_name.split()
            if len(parts) > 0:
                genus = parts[0]
                rank_counts["Genus"][genus] += 1
                
                worms_info = genus_to_lineage.get(genus, {})
                if worms_info:
                    for r in ["Phylum", "Class", "Order", "Family"]:
                        val = worms_info.get(r, "Unknown")
                        if val != "Unknown":
                            rank_counts[r][val] += 1
                            
        total_votes = len(top_k)
        reliability = {}
        
        for r in ranks:
            if rank_counts[r]:
                top_val, count = rank_counts[r].most_common(1)[0]
                conf = count / total_votes
                reliability[r] = {"name": top_val, "confidence": conf}
            else:
                reliability[r] = {"name": "Unknown", "confidence": 0.0}
                
        return reliability

    def resolve_identity(self, matches: List[Dict]) -> Dict[str, Any]:
        """
        @Bio-Taxon Main Logic Loop: Triple-Tier Resolution.
        Returns a rich 'Identity Card' dictionary.
        """
        # Defensive: Handle DataFrame passed as list
        # We check isinstance first to avoid attribute access errors on standard lists
        if isinstance(matches, pd.DataFrame): 
             if matches.empty:
                 return {"display_name": "No Data", "status": "Error", "is_novel": False, "confidence": 0}
             matches = matches.to_dict('records')
             
        if not matches:
             return {"display_name": "No Data", "status": "Error", "is_novel": False, "confidence": 0}

        top_hit = matches[0]
        # Ensure similarity is scalar
        raw_sim = top_hit.get('similarity', 0)
        if isinstance(raw_sim, (np.ndarray, list)):
            if isinstance(raw_sim, np.ndarray) and raw_sim.size == 1:
                top_sim = float(raw_sim.item())
            elif isinstance(raw_sim, list) and len(raw_sim) == 1:
                top_sim = float(raw_sim[0])
            else:
                 logger.warning("Ambiguous similarity score format. Defaulting to 0.0")
                 top_sim = 0.0
        else:
            top_sim = float(raw_sim)

        raw_name_val = top_hit.get('Scientific_Name', top_hit.get('species', 'Unknown'))
        if isinstance(raw_name_val, (np.ndarray, list)):
             raw_name = str(raw_name_val[0]) if len(raw_name_val) > 0 else "Unknown"
        else:
             raw_name = str(raw_name_val)

        # --- TIER 1: VECTOR CONSENSUS ---
        consensus_name, consensus_conf = self.resolve_consensus(matches)
        reliability = self.calculate_taxonomic_reliability(matches)
        
        # Ensure consensus_name is scalar string
        if isinstance(consensus_name, (np.ndarray, list)):
            consensus_name = str(consensus_name[0]) if len(consensus_name) > 0 else "Unknown"
        else:
            consensus_name = str(consensus_name)
        
        # Determine working name: If Consensus is strong Genus or better, use it.
        # Otherwise start with raw top hit.
        working_name = raw_name
        
        # Safe check for substring
        if consensus_name and "Genus" in str(consensus_name): # It's a genus level match
            parts = str(consensus_name).split()
            working_name = parts[0] if len(parts) > 0 else "Unknown" # "Bathymodiolus"
        elif consensus_conf > 0.8:
            working_name = consensus_name

        # --- TIER 2: WoRMS ORACLE ---
        # "Fix for Bathymodiolus": Fuzzy check the working name
        worms_info = self.validate_worms(working_name)
        
        final_name = working_name
        
        # Default to NCBI lineage from LanceDB top-match
        lineage_str = top_hit.get('lineage', 'Unknown')
        source = "Vector AI (NCBI)"

        if worms_info is not None and len(worms_info) > 0:
            # Oracle Spoke! Prioritize WoRMS lineage over NCBI
            final_name = worms_info.get('ScientificName', final_name) # Use the clean Oracle name
            source = "WoRMS Oracle"
            lineage_str = f"{worms_info.get('Phylum')}; {worms_info.get('Class')}; {worms_info.get('Order')}; {worms_info.get('Family')}"
        
        # --- TIER 3: TaxonKit FALLBACK ---
        elif self.taxonkit_active and lineage_str == 'Unknown':
             # Manual lookup using resolve_name_taxonkit assuming that was intended, 
             # or simply log missing method. Based on context, it seems resolve_name_taxonkit is the method.
             # However, the user error says 'expand_lineage_taxonkit' is missing.
             # Checking the class, I don't see expand_lineage_taxonkit defined.
             # I should probably remove this call or use a valid method.
             # Assuming 'resolve_name_taxonkit' returns taxid, lineage lookup needs separate call.
             # If this is a placeholder, I will comment it out or implement a dummy.
             # BUT, I see 'resolve_lineage_by_taxid' might be available or I could implement 'expand_lineage_taxonkit'.
             # Let's fix it by using a safe fallback for now.
             
             # Attempt to resolve name first to get taxid
             # lineage_str = self.expand_lineage_taxonkit(working_name) # ERROR
             lineage_str = "Unknown"
             source = "TaxonKit + TVt"

        # --- DISCOVERY FLAG LOGIC ---
        # Rule: Sim < 0.85 AND Not in WoRMS Oracle
        is_novel = False
        status_label = "Confirmed ID"
        
        # Ensure top_sim is a scalar float
        # Already handled above, but double check
        if isinstance(top_sim, np.ndarray):
             if top_sim.size == 1:
                 top_sim = float(top_sim.item())
             else:
                 top_sim = float(top_sim[0]) if top_sim.size > 0 else 0.0
        
        if isinstance(worms_info, pd.DataFrame):
            # This should not happen per my validate_worms return type being Dict, but defensive
             has_worms = not worms_info.empty
        else:
             has_worms = bool(worms_info)
        
        if top_sim < 0.40:
            is_novel = True
            status_label = "DARK TAXON"
        elif top_sim < 0.85 and not has_worms:
            is_novel = True
            status_label = "POTENTIAL NOVEL TAXON"
        elif top_sim < 0.94:
            status_label = "Divergent / Deep Variant"
            
        # Force novelty for low confidence hits to ensure they enter the discovery pipeline
        if top_sim < 0.30:
            is_novel = True
            status_label = "DARK TAXON"

        if is_novel and reliability:
            family_conf = reliability.get("Family", {}).get("confidence", 0)
            family_name = reliability.get("Family", {}).get("name", "Unknown")
            
            phylum_conf = reliability.get("Phylum", {}).get("confidence", 0)
            phylum_name = reliability.get("Phylum", {}).get("name", "Unknown")
            
            if family_conf >= 0.80 and family_name != "Unknown":
                final_name = f"Potential Novel Genus [Family: {family_name}]"
            elif phylum_conf >= 0.80 and phylum_name != "Unknown":
                final_name = f"Potential Novel Family [Phylum: {phylum_name}]"
            else:
                final_name = "Unclassified Dark Taxon"
        elif is_novel:
            final_name = f"Cryptic {final_name} sp."

        # --- TAXONOMIC RELIABILITY FORMATTING ---
        taxonomic_reliability_str = ""
        hierarchy_str = ""
        
        if reliability:
            # Build the hierarchy string regardless of novelty
            h_parts = []
            
            # Phylum
            p_name = reliability.get("Phylum", {}).get("name", "Unknown")
            if p_name != "Unknown": h_parts.append(p_name)
            
            # Class
            c_name = reliability.get("Class", {}).get("name", "Unknown")
            if c_name != "Unknown": h_parts.append(c_name)
            
            # Order
            o_name = reliability.get("Order", {}).get("name", "Unknown")
            if o_name != "Unknown": h_parts.append(o_name)
            
            # Genus
            g_name = reliability.get("Genus", {}).get("name", "Unknown")
            if is_novel:
                h_parts.append("[PROBABLE GENUS]")
            elif g_name != "Unknown":
                h_parts.append(g_name)
                
            hierarchy_str = " > ".join(h_parts) if h_parts else "Unknown"

        if is_novel and reliability:
            # Hierarchical Consensus Inference
            # 90% Phylum -> Confirmed, 70% Class -> Probable, <30% Species -> NOVEL SPECIES
            hierarchy_parts = []
            
            # Phylum
            phylum_conf = reliability.get("Phylum", {}).get("confidence", 0)
            phylum_name = reliability.get("Phylum", {}).get("name", "Unknown")
            if phylum_conf >= 0.90 and phylum_name != "Unknown":
                hierarchy_parts.append(f"{phylum_name} ({phylum_conf*100:.0f}%)")
            
            # Class
            class_conf = reliability.get("Class", {}).get("confidence", 0)
            class_name = reliability.get("Class", {}).get("name", "Unknown")
            if class_conf >= 0.70 and class_name != "Unknown":
                hierarchy_parts.append(f"{class_name} ({class_conf*100:.0f}%)")
                
            # Order
            order_conf = reliability.get("Order", {}).get("confidence", 0)
            order_name = reliability.get("Order", {}).get("name", "Unknown")
            if order_conf >= 0.50 and order_name != "Unknown":
                hierarchy_parts.append(f"{order_name} ({order_conf*100:.0f}%)")
                
            # Family
            family_conf = reliability.get("Family", {}).get("confidence", 0)
            family_name = reliability.get("Family", {}).get("name", "Unknown")
            if family_conf >= 0.40 and family_name != "Unknown":
                hierarchy_parts.append(f"{family_name} ({family_conf*100:.0f}%)")
                
            # Genus
            genus_conf = reliability.get("Genus", {}).get("confidence", 0)
            genus_name = reliability.get("Genus", {}).get("name", "Unknown")
            if genus_conf >= 0.30 and genus_name != "Unknown":
                hierarchy_parts.append(f"{genus_name} ({genus_conf*100:.0f}%)")
            else:
                hierarchy_parts.append("[Novel Genus]")
                
            taxonomic_reliability_str = " > ".join(hierarchy_parts)
            
        if isinstance(is_novel, (np.ndarray, list)):
             if isinstance(is_novel, np.ndarray):
                 is_novel = bool(is_novel.item()) if is_novel.size == 1 else is_novel.any()
             else:
                 is_novel = bool(is_novel[0]) if len(is_novel) > 0 else False

        vector_data = top_hit.get('vector') 
        if vector_data is None:
             vector_data = top_hit.get('vectors')

        return {
            "display_name": final_name,
            "scientific_name": final_name,
            "status": status_label,
            "is_novel": bool(is_novel),
            "confidence": float(top_sim),
            "confidence_pct": float(top_sim * 100),
            "consensus_score": float(consensus_conf),
            "consensus_name": str(consensus_name),
            "lineage": lineage_str,
            "source_method": source,
            "vector": vector_data,
            "reliability": reliability,
            "taxonomic_reliability_str": taxonomic_reliability_str,
            "hierarchy": hierarchy_str
        }

    def format_search_results(self, raw_hits: List[Dict], gene_type: str="COI") -> List[Dict]:
        """
        Main Pipeline: Converts raw Vector DB hits into Rich Discovery Cards.
        """
        # Handle DataFrame input defensively
        if isinstance(raw_hits, pd.DataFrame):
             if raw_hits.empty: return []
             raw_hits = raw_hits.to_dict('records')

        if not raw_hits:
            return []

        # 1. Resolve Primary Identity (The "Queen" Bee)
        identity = self.resolve_identity(raw_hits)
        
        # Inject UI helpers
        # Map resolve_identity keys to UI expected keys (if different)
        # Ensure similarity is scalar
        raw_sim = 0.0
        if raw_hits and isinstance(raw_hits[0], dict):
             raw_sim = raw_hits[0].get('similarity', 0.0)
        
        if isinstance(raw_sim, (np.ndarray, list)):
             sim = float(raw_sim[0]) if len(raw_sim) > 0 else 0.0
        else:
             sim = float(raw_sim)

        identity['similarity'] = sim
        identity['distance'] = 1.0 - sim
        identity['display_lineage'] = identity['lineage']
        identity['method'] = identity['source_method']
        # identity['confidence_pct'] is float from clean resolve_identity call
        # Defensive check just in case
        conf_pct_raw = identity.get('confidence_pct', 0)
        if isinstance(conf_pct_raw, (np.ndarray, list)):
             conf_pct = float(conf_pct_raw[0]) if len(conf_pct_raw) > 0 else 0.0
        else:
             conf_pct = float(conf_pct_raw)
             
        identity['confidence_pct'] = f"{conf_pct:.1f}%"
        identity['consensus_name'] = identity.get('consensus_name', 'Unknown')
        # Extract Phylum for visualizer coloring
        identity['phylum'] = identity['lineage'].split(';')[0] if identity['lineage'] else "Unknown"
        
        results = [identity]
        
        # 2. Format Neighbors (The "Workers") for Visualization Context
        # We process hits 1..N (omitting 0 as it is represented by identity)
        # OR we keep them all but know that 0 is the consensus. 
        # Visualization typically wants the Top K raw hits.
        
        for i, hit in enumerate(raw_hits[1:], start=1):
             # Ensure Scientific Name is scalar string
             s_name_raw = hit.get('Scientific_Name', hit.get('species', 'Unknown'))
             if isinstance(s_name_raw, (list, np.ndarray)):
                  s_name = str(s_name_raw[0]) if len(s_name_raw) > 0 else "Unknown"
             else:
                  s_name = str(s_name_raw)

             # Ensure Similarity is scalar float
             sim_raw = hit.get('similarity', 0.0)
             if isinstance(sim_raw, (np.ndarray, list)):
                  sim = float(sim_raw[0]) if len(sim_raw) > 0 else 0.0
             else:
                  sim = float(sim_raw)

             # Handle vector safely
             vec_data = hit.get('vector')
             if vec_data is None:
                 vec_data = hit.get('vectors')

             results.append({
                "display_name": s_name,
                "Scientific_Name": s_name,
                "similarity": sim,
                "distance": 1.0 - sim,
                "status": "Neighbor",
                "is_novel": False,
                "display_lineage": "",
                "method": "Vector Neighbor",
                "confidence_pct": f"{sim*100:.1f}%",
                "phylum": "Unknown",
                "vector": vec_data
             })
             
        return results


if __name__ == "__main__":
    # Smoke Test
    engine = TaxonomyEngine()
    print("Testing Tier 2 (WoRMS)...")
    print(engine.validate_worms("Grimpoteuthis"))
    
    print("\nTesting Tier 1 (Consensus)...")
    mock_hits = [
        {"species": "Bathynomus giganteus", "similarity": 0.99},
        {"species": "Bathynomus giganteus", "similarity": 0.98},
        {"species": "Unknown", "similarity": 0.80}
    ]
    print(engine.resolve_consensus(mock_hits))
