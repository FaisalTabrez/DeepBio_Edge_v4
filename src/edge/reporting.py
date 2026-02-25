import json
import csv
import math
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any
from src.edge.config_init import RESULTS_PATH
from src.edge.logger import get_logger

logger = get_logger()

class ResearchReporter:
    """
    @BioArch & @Data-Ops: Research Reporting Module
    Generates professional CSV and JSON Research Briefs for expedition sessions.
    """
    def __init__(self):
        self.results_dir = RESULTS_PATH
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def calculate_shannon_wiener(self, taxa_counts: Dict[str, int]) -> float:
        """
        Calculates the Shannon-Wiener Diversity Index (H').
        H' = -sum(p_i * ln(p_i))
        """
        total_individuals = sum(taxa_counts.values())
        if total_individuals == 0:
            return 0.0
            
        h_prime = 0.0
        for count in taxa_counts.values():
            p_i = count / total_individuals
            if p_i > 0:
                h_prime -= p_i * math.log(p_i)
                
        return round(h_prime, 4)

    def generate_expedition_summary(self, session_results: List[Dict], novel_clusters: List[Dict] | None = None) -> Dict[str, str | None]:
        """
        Generates the Research Brief containing Taxonomic Summary, Novelty Registry, and Diversity Metrics.
        Saves to E:/DeepBio_Scan/results/ (or configured RESULTS_PATH).
        """
        if not session_results:
            logger.warning("No session results provided for reporting.")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"Expedition_Brief_{timestamp}"
        
        # 1. Process Taxonomic Summary
        taxa_counts = Counter()
        taxonomic_summary = []
        
        for hit in session_results:
            name = hit.get('display_name', 'Unknown')
            taxa_counts[name] += 1
            
            # Extract AphiaID if available from lineage or source (mocked extraction if not explicitly stored)
            # In a real scenario, AphiaID would be part of the hit dictionary from WoRMS
            aphia_id = hit.get('aphia_id', 'N/A') 
            
            taxonomic_summary.append({
                "Query_ID": hit.get('query_id', 'Unknown'),
                "Identified_Taxa": name,
                "AphiaID": aphia_id,
                "Consensus_Confidence": hit.get('confidence_pct', '0%'),
                "Status": hit.get('status', 'Unknown'),
                "Is_Novel": hit.get('is_novel', False)
            })

        # 2. Process Novelty Registry (Dark Taxa)
        novelty_registry = []
        if novel_clusters:
            for cluster in novel_clusters:
                # Extract centroid coordinates if available
                centroid = cluster.get('centroid', [])
                coords = f"[{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]" if len(centroid) >= 3 else "N/A"
                
                novelty_registry.append({
                    "NTU_ID": cluster.get('cluster_id', 'Unknown'),
                    "Size": cluster.get('size', 0),
                    "Holotype_Sequence": cluster.get('holotype_seq', 'Sequence Data Unavailable')[:50] + "...", # Truncated for brief
                    "Latent_Coordinates": coords,
                    "Nearest_Known_Relative": cluster.get('nearest_taxon', 'Unknown')
                })

        # 3. Calculate Diversity Metrics
        species_richness = len(taxa_counts)
        shannon_index = self.calculate_shannon_wiener(taxa_counts)
        
        metrics = {
            "Total_Sequences_Analyzed": len(session_results),
            "Species_Richness": species_richness,
            "Shannon_Wiener_Index": shannon_index,
            "Novel_Taxonomic_Units_Discovered": len(novel_clusters) if novel_clusters else 0
        }

        # Compile Full Report
        report = {
            "Expedition_Metadata": {
                "Timestamp": datetime.now().isoformat(),
                "System_Version": "DeepBio-Scan v4.2",
                "Reference_Scale": "100,000 Marine Genomic Signatures",
                "Indexing": "IVF-128 / PQ-96"
            },
            "Diversity_Metrics": metrics,
            "Taxonomic_Summary": taxonomic_summary,
            "Novelty_Registry": novelty_registry
        }

        # Save JSON
        json_path = self.results_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)

        # Save CSV (Taxonomic Summary)
        csv_path = self.results_dir / f"{base_filename}_Taxa.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if taxonomic_summary:
                writer = csv.DictWriter(f, fieldnames=taxonomic_summary[0].keys())
                writer.writeheader()
                writer.writerows(taxonomic_summary)
                
        # Save CSV (Novelty Registry)
        if novelty_registry:
            novelty_csv_path = self.results_dir / f"{base_filename}_Novelty.csv"
            with open(novelty_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=novelty_registry[0].keys())
                writer.writeheader()
                writer.writerows(novelty_registry)

        logger.info(f"Research Brief generated successfully at {self.results_dir}")
        
        return {
            "json": str(json_path),
            "csv_taxa": str(csv_path),
            "csv_novelty": str(novelty_csv_path) if novelty_registry else None
        }
