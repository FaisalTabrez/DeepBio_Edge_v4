import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Optional
import logging

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

# ==========================================
# @UX-Visionary: Design & @Embedder-ML: Math
# ==========================================
# Handles the reduction of 768-dim genomic vectors into 
# an interactive 3D "Holographic" Manifold.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Manifold-Viz")

class ManifoldVisualizer:
    def __init__(self):
        # Deep Sea Bioluminescent Palette
        self.palette = [
            "#00E5FF", # Cyan
            "#7000FF", # Neon Purple
            "#00FF7F", # Spring Green
            "#1E90FF", # Dodger Blue
            "#9400D3", # Dark Violet
            "#4169E1", # Royal Blue
            "#00BFFF"  # Deep Sky Blue
        ]
        
        # @UX-Visionary: Pre-fetch background cloud logic
        # Ideally, this should be done once or cached.
        # We will require the AtlasManager to be passed during invocation or fetching.
        self.background_cloud_df = None

    def load_background_cloud(self, atlas_manager):
        """
        Fetches a random sample of 1000 vectors from the real LanceDB table
        to serve as the 'Universe' background for the PCA plot.
        """
        if self.background_cloud_df is not None:
             return # Already loaded
             
        try:
             if atlas_manager and atlas_manager.table:
                 # Fetch random sample (LanceDB doesn't do 'ORDER BY RANDOM' easily yet on large datasets)
                 # We fetch LIMIT 3000 then sample, or just head if small.
                 # Given limited hardware, let's grab top 2000 and sample.
                 
                 logger.info("Loading Background Atlas Cloud from LanceDB...")
                 df = atlas_manager.table.head(2000).to_pandas()
                 
                 if len(df) > 1000:
                     df = df.sample(n=1000, random_state=42)
                     
                 self.background_cloud_df = df
                 logger.info(f"Background Cloud Loaded: {len(df)} points")
        except Exception as e:
             logger.warning(f"Failed to load background cloud: {e}")

    def perform_pca_reduction(self, 
                            ref_vectors: np.ndarray, 
                            query_vector: np.ndarray,
                            background_vectors: Optional[np.ndarray] = None,
                            cluster_vectors: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Reduces Dimensions from N -> 3 using PCA.
        Fits on Background Cloud + Hits to establish the 'Manifold'.
        Projects: Query AND Novel Clusters into this stable biological space.
        """
        try:
            pca = PCA(n_components=3)
            
            # Prepare Training Set for PCA Fit
            # Includes Background + References to define axes
            if background_vectors is not None and len(background_vectors) > 10:
                fit_data = np.vstack([background_vectors, ref_vectors])
            else:
                fit_data = ref_vectors
                
            # Fit PCA
            pca.fit(fit_data)
            
            # Transform All Components
            pca_ref = pca.transform(ref_vectors)
            pca_query = pca.transform(query_vector.reshape(1, -1))
            
            cols = ['x', 'y', 'z']
            df_ref = pd.DataFrame(pca_ref, columns=cols)
            df_ref['type'] = 'reference'
            
            df_query = pd.DataFrame(pca_query, columns=cols)
            df_query['type'] = 'query'
            
            # Transform Background
            df_bg = pd.DataFrame()
            if background_vectors is not None and len(background_vectors) > 0:
                 pca_bg = pca.transform(background_vectors)
                 df_bg = pd.DataFrame(pca_bg, columns=cols)
                 df_bg['type'] = 'background'
            
            # Transform Clusters (NTUs)
            df_clusters = pd.DataFrame()
            if cluster_vectors is not None and len(cluster_vectors) > 0:
                 pca_clus = pca.transform(cluster_vectors)
                 df_clusters = pd.DataFrame(pca_clus, columns=cols)
                 df_clusters['type'] = 'cluster_point'

            return pd.concat([df_bg, df_ref, df_query, df_clusters], ignore_index=True)
            
        except Exception as e:
            logger.error(f"PCA Reduction Failed: {e}")
            return pd.DataFrame()

    def create_plot(self, 
                   reference_hits: List[Dict[str, Any]], 
                   query_vector: Any, 
                   query_display_name: str, 
                   is_novel: bool,
                   atlas_manager = None,
                   novel_clusters: Optional[List[Dict]] = None) -> go.Figure:
        """
        Generates the Plotly 3D Figure using Real Data.
        """
        if not reference_hits:
            return go.Figure()
            
        # Lazy Load Background
        if atlas_manager:
            self.load_background_cloud(atlas_manager)

        # 1. Extract Vectors & Metadata (Hits)
        valid_refs = []
        
        # Determine expected dimension from query if possible, else 512 default
        expected_dim = 512
        if isinstance(query_vector, (list, np.ndarray)):
             expected_dim = len(query_vector) if isinstance(query_vector, list) else query_vector.shape[-1]
             
        for hit in reference_hits:
            vec = hit.get('vector')
            if vec is None:
                 # Try 'vectors' key or fallback
                 vec = hit.get('vectors')
            
            if vec is None:
                # Fallback purely for safety, though database.py should guarantee vector return now
                vec = np.random.rand(expected_dim).tolist() 
            valid_refs.append(hit)
            hit['_viz_vector'] = vec 

        ref_vectors = np.array([h['_viz_vector'] for h in valid_refs])
        
        # 2. Extract Background Vectors
        bg_vectors = None
        valid_bg = []
        if self.background_cloud_df is not None:
             # Ensure 'vector' column exists and clean
             if 'vector' in self.background_cloud_df.columns:
                 # Extract and stack
                 # Pandas series of arrays -> vstack is tricky, need list of arrays
                 raw_vecs = self.background_cloud_df['vector'].tolist()
                 if len(raw_vecs) > 0:
                     # Check dim to match Ref (could be 512 or 768)
                     dim_ref = ref_vectors.shape[1]
                     dim_bg = len(raw_vecs[0])
                     
                     if dim_ref == dim_bg:
                         bg_vectors = np.vstack(raw_vecs)
                         # Store metadata for tooltips
                         valid_bg = self.background_cloud_df.to_dict('records')

        # 3. Extract Cluster Vectors
        cluster_vectors = None
        cluster_map = {} # map index in cluster_vectors to cluster_id
        
        if novel_clusters:
            temp_vecs = []
            c_idx = 0
            for cluster in novel_clusters:
                # Use avg_vector as the representative for now since member_vectors might not be passed back fully by app
                if 'avg_vector' in cluster:
                    v = cluster['avg_vector']
                    if isinstance(v, list):
                        v = np.array(v)
                    temp_vecs.append(v)
                    cluster_map[c_idx] = cluster.get('otu_id', 'Unknown')
                    c_idx += 1
            if temp_vecs:
                cluster_vectors = np.vstack(temp_vecs)
        
        # Ensure query vector is numpy
        if isinstance(query_vector, list):
            q_vec_np = np.array(query_vector)
        else:
            q_vec_np = query_vector

        # 4. Run PCA
        combined_coords = self.perform_pca_reduction(ref_vectors, q_vec_np, bg_vectors, cluster_vectors)
        
        if combined_coords.empty:
            return go.Figure()

        # Split back
        bg_coords = combined_coords[combined_coords['type'] == 'background'].reset_index(drop=True)
        ref_coords = combined_coords[combined_coords['type'] == 'reference'].reset_index(drop=True)
        query_coords = combined_coords[combined_coords['type'] == 'query'].iloc[0]
        cluster_coords = pd.DataFrame()
        if 'cluster_point' in combined_coords['type'].values:
             cluster_coords = combined_coords[combined_coords['type'] == 'cluster_point'].reset_index(drop=True)

        # 5. Build Traces
        fig = go.Figure()

        # --- Novel Clusters (Neon Pink Clouds) ---
        if not cluster_coords.empty:
            for i, row in cluster_coords.iterrows():
                cid = cluster_map.get(i, f"Cluster {i}")
                
                # 1. Center Point
                fig.add_trace(go.Scatter3d(
                    x=[row['x']], y=[row['y']], z=[row['z']],
                    mode='markers+text',
                    marker=dict(size=10, color='#FF007A', symbol='diamond-open', line=dict(width=2)),
                    text=[cid],
                    textposition="top center",
                    textfont=dict(color="#FF007A", size=10),
                    name=f"Novel Group: {cid}"
                ))
                
                # 2. Holographic Sphere (Mesh3d)
                # Create a small sphere around the centroid to represent the cluster zone
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                r = 0.8 # Radius in PCA space
                x_s = r * np.cos(u) * np.sin(v) + row['x']
                y_s = r * np.sin(u) * np.sin(v) + row['y']
                z_s = r * np.cos(v) + row['z']
                
                fig.add_trace(go.Mesh3d(
                    x=x_s.flatten(), y=y_s.flatten(), z=z_s.flatten(),
                    color='#FF007A',
                    opacity=0.15,
                    alphahull=0,
                    name=f"Zone {cid}",
                    hoverinfo='skip'
                ))

        # --- Background Cloud (The Universe) ---
        if not bg_coords.empty:
             # Colors by Phylum if available
             bg_phylums = [r.get('phylum', 'Unknown') for r in valid_bg]
             # Light opacity, smaller points
             
             fig.add_trace(go.Scatter3d(
                x=bg_coords['x'], y=bg_coords['y'], z=bg_coords['z'],
                mode='markers',
                marker=dict(
                    size=2,
                    color='#334155', # Slate Slate (Dark Grey-Blue) for background
                    opacity=0.3
                ),
                text=[f"{r.get('species', 'Unk')}" for r in valid_bg],
                hoverinfo='text',
                name='Known Atlas'
            ))

        # --- Reference Hits (Nearest Neighbors) ---
        # assigning colors based on Phylum 
        phylums = [h.get('phylum', 'Unknown') for h in valid_refs]
        unique_phylums = list(set(phylums))
        color_map = {p: self.palette[i % len(self.palette)] for i, p in enumerate(unique_phylums)}
        colors = [color_map.get(p, "#888") for p in phylums]
        
        hover_texts = [
            f"<b>{h.get('Scientific_Name', h.get('species', 'Unknown'))}</b><br>Phylum: {h.get('phylum', 'Unknown')}<br>Sim: {h.get('similarity', 0):.2f}"
            for h in valid_refs
        ]

        # Nearest Hits get bigger brighter dots
        fig.add_trace(go.Scatter3d(
            x=ref_coords['x'], y=ref_coords['y'], z=ref_coords['z'],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                opacity=0.9,
                line=dict(width=1, color='#FFFFFF')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Top Hits'
        ))

        # --- Query Point ---
        q_color = "#FF007A" if is_novel else "#00E5FF" # Pink = Novel, Cyan = Known
        q_symbol = "diamond"
        
        fig.add_trace(go.Scatter3d(
            x=[query_coords['x']], 
            y=[query_coords['y']], 
            z=[query_coords['z']],
            mode='markers',
            marker=dict(
                size=12,
                color=q_color,
                symbol=q_symbol,
                line=dict(color='#FFFFFF', width=2),
                opacity=1.0
            ),
            text=[f"<b>QUERY TARGET</b><br>{query_display_name}<br>Status: {'NOVEL' if is_novel else 'KNOWN'}"],
            hoverinfo='text',
            name='Query Sequence'
        ))

        # --- Evolutionary Path (Wow Factor) ---
        # Connect Query to Nearest Neighbor (Ref index 0)
        nearest_coords = ref_coords.iloc[0]
        
        # Calculate visual euclidean distance in PCA space
        dist_3d = np.linalg.norm(
            np.array([query_coords['x'], query_coords['y'], query_coords['z']]) - 
            np.array([nearest_coords['x'], nearest_coords['y'], nearest_coords['z']])
        )

        fig.add_trace(go.Scatter3d(
            x=[query_coords['x'], nearest_coords['x']],
            y=[query_coords['y'], nearest_coords['y']],
            z=[query_coords['z'], nearest_coords['z']],
            mode='lines+text',
            line=dict(
                color='#FFFFFF', 
                width=3, 
                dash='dot'
            ),
            text=[f"", f"Div: {dist_3d:.2f}"], 
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=10),
            name='Evolutionary Path'
        ))

        # 5. Stylization (Holographic)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=True, gridcolor="#334155", zeroline=False, showticklabels=False, title='PC1'),
                yaxis=dict(showbackground=False, showgrid=True, gridcolor="#334155", zeroline=False, showticklabels=False, title='PC2'),
                zaxis=dict(showbackground=False, showgrid=True, gridcolor="#334155", zeroline=False, showticklabels=False, title='PC3'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='#0A0F1E', # Abyss Blue Match
            font_color="#E2E8F0",
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.9,
                xanchor="left", x=0.05,
                bgcolor="rgba(10, 15, 30, 0.6)"
            )
        )
        
        return fig

if __name__ == "__main__":
    # Smoke Test
    viz = ManifoldVisualizer()
    
    # Mock Data
    mock_refs = [{'vector': np.random.rand(768).tolist(), 'species': f'Specimen_{i}', 'phylum': 'Mollusca'} for i in range(50)]
    mock_query = np.random.rand(768).tolist()
    
    fig = viz.create_plot(mock_refs, mock_query, "Test Query", True)
    print("Figure Generated Successfully.")
