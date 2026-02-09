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
        self.min_cluster_size = 2 # Optimized for small batch "Edge" discovery
        self.metric = 'euclidean'
        
        if not HAS_HDBSCAN and DBSCAN is None:
            logger.warning("No clustering libraries found (hdbscan/sklearn). Discovery disabled.")
            self.mode = "DISABLED"
        elif HAS_HDBSCAN:
            self.mode = "HDBSCAN"
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric=self.metric)
        else:
            self.mode = "DBSCAN"
            # eps=0.3 is a heuristic for normalized vector space (cosine-ish)
            self.clusterer = DBSCAN(eps=0.3, min_samples=self.min_cluster_size, metric=self.metric)
            
        logger.info(f"Discovery Engine Online. Mode: {self.mode}")

    def _calculate_centroid(self, vectors: np.ndarray) -> np.ndarray:
        """@Embedder-ML: Computes geometric center of the cluster in latent space."""
        return np.mean(vectors, axis=0)

    def _find_nearest_neighbor(self, centroid: np.ndarray) -> Tuple[str, float]:
        """
        Queries the Atlas to find the nearest KNOWN relative to this new cluster.
        Returns: (Taxon_Name, Distance)
        """
        # top_k=1 to find closest anchor
        results = self.atlas.query_vector(centroid, top_k=1)
        if results:
            hit = results[0]
            name = hit.get('Scientific_Name', hit.get('species', 'Unknown'))
            similarity = hit.get('similarity', 0.0)
            return name, (1.0 - similarity)
        return "Unknown", 1.0

    def analyze_novelty(self, session_buffer: List[Dict]) -> List[Dict]:
        """
        Main Routine: Groups 'Potential Novel Taxa' into provisional OTUs.
        """
        if self.mode == "DISABLED" or not session_buffer:
            return []

        # 1. Filter for 'Dark Taxa' (Is Novel == True)
        # Check for valid vector data
        dark_taxa = [item for item in session_buffer if item.get('is_novel', False) and item.get('vector') is not None]
        
        if len(dark_taxa) < self.min_cluster_size:
            logger.info("Insufficient Dark Taxa for clustering.")
            return []
            
        # 2. Prepare Tensor
        # session_buffer vectors might be lists; convert to numpy
        vectors = np.array([np.array(item['vector']) for item in dark_taxa])
        
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
            # How far is this group from known biology?
            nearest_relative, divergence = self._find_nearest_neighbor(centroid)
            
            # Generate Provisional ID
            # e.g. DeepBio-NTU-A1B2
            # NTU = Novel Taxon Unit
            otu_id = f"DeepBio-NTU-{str(uuid.uuid4())[:6].upper()}"
            
            entity = {
                "otu_id": otu_id,
                "cluster_size": len(indices),
                "representative_name": representative_seq['display_name'], # e.g. "Cryptic Bathymodiolus sp."
                "nearest_relative": nearest_relative,
                "biological_divergence": divergence,
                "divergence_pct": f"{divergence*100:.1f}%",
                "avg_vector": centroid,
                "status": "New Species Group" if divergence > 0.03 else "Cryptic Variant Group"
            }
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
            return [{"Scientific_Name": "Rimicaris exoculata", "similarity": 0.85}]
            
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
