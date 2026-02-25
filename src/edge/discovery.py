import numpy as np
import logging
import uuid
from typing import List, Dict, Any, Tuple
from collections import Counter

# Try importing HDBSCAN, fallback to sklearn DBSCAN if necessary
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        DBSCAN = None

from src.edge.database import AtlasManager

# ==========================================
# @BioArch & @Embedder-ML: Unsupervised Discovery Module
# ==========================================
# Goal: Detect latent clusters in 'Dark Taxa' (unknowns).
# Logic: 
# 1. Collect novel vectors.
# 2. Perform density-based clustering (HDBSCAN/DBSCAN).
# 3. Analyze Cluster Centroids against the Knowledge Graph (Atlas).
# 4. Assign Provisional OTU IDs (e.g., DeepBio-NTU-001).

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@BioArch-Discovery")

class DiscoveryEngine:
    """
    Unsupervised learning engine that acts as the 'Scientist in the Loop'.
    Monitors the 'Dark Taxa' buffer and proposes new biological groups.
    """
    def __init__(self, atlas: AtlasManager):
        self.atlas = atlas
        self.min_cluster_size = 3 # Optimized for small batch "Edge" discovery
        self.min_samples = 2
        self.metric = 'euclidean'
        
        if not HAS_HDBSCAN and (DBSCAN is None):
            logger.warning("No clustering libraries found (hdbscan/sklearn). Discovery disabled.")
            self.mode = "DISABLED"
        elif HAS_HDBSCAN:
            self.mode = "HDBSCAN"
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, metric=self.metric)
        elif DBSCAN is not None:
            self.mode = "DBSCAN"
            # eps=0.3 is a heuristic for normalized vector space (cosine-ish)
            self.clusterer = DBSCAN(eps=0.3, min_samples=self.min_samples, metric=self.metric)
        else:
            self.mode = "DISABLED"
            logger.warning("Falback logic failed. Discovery disabled.")
            
        logger.info(f"Discovery Engine Online. Mode: {self.mode}")

    def _calculate_centroid(self, vectors: np.ndarray) -> np.ndarray:
        """@Embedder-ML: Computes geometric center of the cluster in latent space."""
        return np.mean(vectors, axis=0)

    def _find_nearest_neighbor(self, centroid: np.ndarray) -> Tuple[str, float, str, str]:
        """
        Queries the Atlas to find the nearest KNOWN relative to this new cluster.
        Performs a 'Deep Search' (top 50) to find Rank Stability.
        Returns: (Taxon_Name, Distance, Consensus_Rank, Consensus_Name)
        """
        # top_k=50 for Deep Search
        results = self.atlas.query_vector(centroid, top_k=50)
        if not results:
            return "Unknown", 1.0, "Unknown", "Unknown"
            
        # Calculate distance to the absolute nearest known point
        hit = results[0]
        name = hit.get('Scientific_Name', hit.get('species', 'Unknown'))
        similarity = hit.get('similarity', 0.0)
        distance = 1.0 - similarity
        
        # High-Level Consensus: Rank Stability
        families = []
        genera = []
        for r in results:
            lineage = r.get('lineage', '')
            if lineage:
                parts = lineage.split(';')
                if len(parts) >= 4:
                    families.append(parts[3].strip())
                if len(parts) >= 5:
                    genera.append(parts[4].strip())
                    
        family_counts = Counter(families)
        genus_counts = Counter(genera)
        
        consensus_rank = "Unknown"
        consensus_name = "Unknown"
        
        if family_counts:
            top_family, f_count = family_counts.most_common(1)[0]
            if f_count >= 35: # 70% of 50
                consensus_rank = "Family"
                consensus_name = top_family
                
                # Check if genera are mixed
                if genus_counts:
                    top_genus, g_count = genus_counts.most_common(1)[0]
                    if g_count < 35: # Mixed genera
                        consensus_rank = "Novel Genus within Family"
                        
        return name, distance, consensus_rank, consensus_name

    def analyze_novelty(self, session_buffer: List[Dict]) -> List[Dict]:
        """
        Main Routine: Groups 'Potential Novel Taxa' into provisional OTUs.
        """
        if self.mode == "DISABLED" or not session_buffer:
            return []

        # 1. Filter for 'Dark Taxa' (Is Novel == True)
        # Check for valid vector data. Use direct None check to avoid numpy ambiguity if 'vector' is array
        dark_taxa = []
        for item in session_buffer:
             raw_novel = item.get('is_novel', False)
             is_novel_bool = False
             if isinstance(raw_novel, np.ndarray):
                 is_novel_bool = bool(raw_novel.item()) if raw_novel.size == 1 else raw_novel.any()
             else:
                 is_novel_bool = bool(raw_novel)

             vec = item.get('vector')
             if is_novel_bool and vec is not None:
                 dark_taxa.append(item)
        
        if len(dark_taxa) < self.min_cluster_size:
            logger.info("Insufficient Dark Taxa for clustering.")
            return []
            
        # 2. Prepare Tensor
        # vectors = np.array([np.array(item['vector']) for item in dark_taxa])
        # Force conversion to list first to avoid ambiguous truth value check on numpy array inside list comp or constructor
        vec_list = []
        for item in dark_taxa:
            v = item.get('vector')
            if v is not None:
                # Ensure it is a flat list/array for clustering
                if isinstance(v, np.ndarray):
                    v = v.flatten()
                elif isinstance(v, list):
                    v = np.array(v).flatten()
                vec_list.append(v)
        
        if not vec_list:
             return []

        vectors = np.array(vec_list)
        
        # 3. Fit Clusters
        try:
            if self.mode == "HDBSCAN" and self.clusterer:
                labels = self.clusterer.fit_predict(vectors)
            elif self.mode == "DBSCAN" and self.clusterer:
                labels = self.clusterer.fit_predict(vectors)
            else:
                 logger.warning("Clustering attempted but engine disabled or invalid.")
                 return []
        except Exception as e:
            logger.error(f"Clustering Crash: {e}")
            return []

        # 4. Process Clusters
        discovered_entities = []
        unique_labels = set(labels)
        
        # Stable ID assignment (Alpha, Beta, Gamma, etc.)
        greek_alphabet = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa']
        cluster_idx = 0
        
        for label in unique_labels:
            if label == -1:
                # Noise points (orphans) - ignore or treat as singletons?
                # @BioArch: Ignore noise in strict grouping for now.
                continue
                
            # Get members of this cluster
            indices = [i for i, x in enumerate(labels) if x == label]
            cluster_vectors = vectors[indices]
            cluster_members = [dark_taxa[i] for i in indices]
            
            # --- @Embedder-ML: Centroid Analysis ---
            centroid = self._calculate_centroid(cluster_vectors)
            
            # Find Representative (Member closest to centroid)
            dists = [np.linalg.norm(v - centroid) for v in cluster_vectors]
            rep_idx = np.argmin(dists)
            representative_seq = cluster_members[rep_idx]
            
            # --- @BioArch: Biological Divergence Hook ---
            # Deep Search for Rank Stability
            nearest_relative, centroid_dist, consensus_rank, consensus_name = self._find_nearest_neighbor(centroid)
            
            # Calculate the average distance from the cluster members to the nearest 'Known' point
            member_distances = []
            for v in cluster_vectors:
                res = self.atlas.query_vector(v, top_k=1)
                if res:
                    member_distances.append(1.0 - res[0].get('similarity', 0.0))
                else:
                    member_distances.append(1.0)
            avg_member_distance = float(np.mean(member_distances)) if member_distances else 1.0
            
            # Generate Provisional ID
            # e.g. DeepBio-NTU-Modiolidae-Alpha
            if consensus_name != "Unknown":
                otu_id = f"DeepBio-NTU-{consensus_name}-{greek_alphabet[cluster_idx % len(greek_alphabet)]}"
            else:
                otu_id = f"DeepBio-NTU-Unknown-{greek_alphabet[cluster_idx % len(greek_alphabet)]}"
            cluster_idx += 1
            
            # Divergence Metric
            if avg_member_distance > 0.25:
                status = "Candidate New Genus"
            elif avg_member_distance > 0.03:
                status = "New Species Group"
            else:
                status = "Cryptic Variant Group"
                
            if consensus_rank == "Novel Genus within Family":
                classification = f"Novel Genus within Family {consensus_name}"
            elif consensus_rank == "Family":
                classification = f"Novel Species within Family {consensus_name}"
            else:
                classification = f"Novel Taxon near {nearest_relative}"
            
            entity = {
                "otu_id": otu_id,
                "cluster_size": len(indices),
                "representative_id": representative_seq.get('query_id', 'Unknown'),
                "representative_name": representative_seq['display_name'], # e.g. "Cryptic Bathymodiolus sp."
                "nearest_relative": nearest_relative,
                "biological_divergence": avg_member_distance,
                "divergence_pct": f"{avg_member_distance*100:.1f}%",
                "avg_vector": centroid,
                "status": status,
                "classification": classification,
                "consensus_rank": consensus_rank,
                "consensus_name": consensus_name,
                "lineage": representative_seq.get('lineage', 'Unknown'),
                "members": [{"id": m.get('query_id', 'Unknown'), "seq": m.get('raw_sequence', 'NNNN')} for m in cluster_members],
                "member_vectors": cluster_vectors
            }
            
            # Tag the original buffer items with their new cluster ID
            for member in cluster_members:
                member['cluster_id'] = otu_id
                member['cluster_entity'] = entity
                
            discovered_entities.append(entity)
            
        return discovered_entities

if __name__ == "__main__":
    # Mock Test
    print("Initializing Discovery Engine Mock...")
    # need mock atlas
    class MockAtlas(AtlasManager):
        def __init__(self):
            # Bypass super init to avoid DB connection in mock
            pass 
            
        def query_vector(self, v, top_k):
            return [{"Scientific_Name": "Rimicaris exoculata", "similarity": 0.85, "lineage": "Eukaryota;Metazoa;Arthropoda;Alvinocarididae;Rimicaris;Rimicaris exoculata"}] * top_k
            
    engine = DiscoveryEngine(MockAtlas())
    
    # Mock Vectors (3 close, 1 far)
    # Using 512 dim as per system spec (though code is dim-agnostic)
    v1 = np.random.rand(512)
    v2 = v1 + 0.01 # Close
    v3 = v1 + 0.02 # Close
    v4 = np.random.rand(512) # Far
    
    buffer = [
        {"is_novel": True, "vector": v1, "display_name": "Seq A"},
        {"is_novel": True, "vector": v2, "display_name": "Seq B"},
        {"is_novel": True, "vector": v3, "display_name": "Seq C"},
        {"is_novel": True, "vector": v4, "display_name": "Seq D"}, 
        {"is_novel": False, "vector": v1, "display_name": "Known Seq"} # Should be ignored
    ]
    
    results = engine.analyze_novelty(buffer)
    print(f"Discovered: {results}")
