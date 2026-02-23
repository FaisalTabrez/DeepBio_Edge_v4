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
                if len(ref_vectors) > 0:
                    fit_data = np.vstack([background_vectors, ref_vectors])
                else:
                    fit_data = background_vectors
            elif len(ref_vectors) > 0:
                fit_data = ref_vectors
            else:
                 # Edge case: No data to fit (fresh DB)
                 # Fit on query + noise for stability
                 fit_data = np.vstack([query_vector, query_vector + np.random.normal(0, 0.01, query_vector.shape)])
                
            # Fit PCA
            if len(fit_data) < 3:
                return pd.DataFrame()

            pca.fit(fit_data)
            
            # --- TRANSFORM ---
            df_bg = pd.DataFrame()
            if background_vectors is not None and len(background_vectors) > 0:
                 pca_bg = pca.transform(background_vectors)
                 df_bg = pd.DataFrame(pca_bg, columns=['x', 'y', 'z'])
                 df_bg['type'] = 'background'
            
            df_ref = pd.DataFrame()
            if len(ref_vectors) > 0:
                pca_ref = pca.transform(ref_vectors)
                df_ref = pd.DataFrame(pca_ref, columns=['x', 'y', 'z'])
                df_ref['type'] = 'reference'
            
            pca_query = pca.transform(query_vector.reshape(1, -1))
            df_query = pd.DataFrame(pca_query, columns=['x', 'y', 'z'])
            df_query['type'] = 'query'
            
            # Transform Clusters (NTUs)
            df_clusters = pd.DataFrame()
            if cluster_vectors is not None and len(cluster_vectors) > 0:
                 pca_clus = pca.transform(cluster_vectors)
                 df_clusters = pd.DataFrame(pca_clus, columns=['x', 'y', 'z'])
                 df_clusters['type'] = 'cluster_point'

            return pd.concat([df_bg, df_ref, df_query, df_clusters], ignore_index=True)
            
        except Exception as e:
            logger.error(f"PCA Reduction Failed: {e}")
            return pd.DataFrame()

    def create_plot(self, 
                   reference_hits: List[Dict], 
                   query_vector: np.ndarray, 
                   query_display_name: str,
                   is_novel: bool,
                   atlas_manager: Any,
                   novel_clusters: Optional[List[Dict]] = None) -> go.Figure:
        """
        Generates the Plotly 3D Figure using Real Data.
        """
        # 1. Ensure Background
        if self.background_cloud_df is None:
            self.load_background_cloud(atlas_manager)
            
        # 2. Extract Vectors from Hits
        vectors = []
        valid_refs = []
        
        for hit in reference_hits:
            v = hit.get('vector')
            if v is None: v = hit.get('vectors') # Try fallback
            
            if v is not None:
                if isinstance(v, list):
                    vectors.append(np.array(v, dtype=np.float32))
                else:
                    vectors.append(v)
                valid_refs.append(hit)
                
        if vectors:
            ref_vecs_np = np.vstack(vectors)
        else:
            ref_vecs_np = np.empty((0, 768)) # Shape guess
            
        # 3. Get Background Vectors
        bg_vecs_np = None
        valid_bg = []
        if self.background_cloud_df is not None and not self.background_cloud_df.empty:
            if 'vector' in self.background_cloud_df.columns:
                 # Clean nans if any
                 bg_clean = self.background_cloud_df.dropna(subset=['vector'])
                 bg_list = bg_clean['vector'].tolist()
                 if bg_list:
                    bg_vecs_np = np.vstack(bg_list)
                    valid_bg = bg_clean.to_dict('records')

        # 4. Extract Cluster Vectors
        cluster_vectors = None
        cluster_map = {} 
        if novel_clusters:
            temp_vecs = []
            for i, cluster in enumerate(novel_clusters):
                if 'avg_vector' in cluster:
                    v = cluster['avg_vector']
                    if isinstance(v, list): v = np.array(v)
                    temp_vecs.append(v)
                    cluster_map[i] = cluster.get('otu_id', 'Unknown')
            if temp_vecs:
                cluster_vectors = np.vstack(temp_vecs)

        # 5. Run PCA - Dimension Reduction
        # Ensure query is numpy
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
            
        combined_coords = self.perform_pca_reduction(
            ref_vectors=ref_vecs_np,
            query_vector=query_vector,
            background_vectors=bg_vecs_np,
            cluster_vectors=cluster_vectors
        )
        
        if combined_coords.empty:
            return go.Figure()

        # 6. Split Coords back to logical groups
        bg_coords = combined_coords[combined_coords['type'] == 'background'].reset_index(drop=True)
        ref_coords = combined_coords[combined_coords['type'] == 'reference'].reset_index(drop=True)
        query_coords = combined_coords[combined_coords['type'] == 'query'].iloc[0]
        cluster_coords = pd.DataFrame()
        if 'cluster_point' in combined_coords['type'].values:
             cluster_coords = combined_coords[combined_coords['type'] == 'cluster_point'].reset_index(drop=True)

        # --- PLOTLY CONSTRUCTION ---
        fig = go.Figure()

        # A. Background Cloud (The Universe) - Dark & Subtle
        if not bg_coords.empty:
            # Create hover text
            hover_bg = [
                f"Record: {valid_bg[i].get('scientific_name', 'Unknown')}" 
                if i < len(valid_bg) else "Record: Unknown"
                for i in range(len(bg_coords))
            ]
            
            fig.add_trace(go.Scatter3d(
                x=bg_coords['x'], y=bg_coords['y'], z=bg_coords['z'],
                mode='markers',
                marker=dict(size=2, color='#334155', opacity=0.3),
                hoverinfo='text',
                text=hover_bg,
                name='Atlas Background'
            ))

        # B. Reference Context (The Neighbors) - Cyan #00E5FF
        if not ref_coords.empty:
            hover_ref = [
                f"<b>{h.get('scientific_name', 'Unknown')}</b><br>Sim: {h.get('similarity', 0):.2f}" 
                for h in valid_refs
            ]
            
            fig.add_trace(go.Scatter3d(
                x=ref_coords['x'], y=ref_coords['y'], z=ref_coords['z'],
                mode='markers',
                marker=dict(size=6, color='#00E5FF', opacity=0.9, line=dict(width=0.5, color='white')),
                hoverinfo='text',
                text=hover_ref,
                name='Genomic Neighborhood'
            ))

            # Optional: Convex Hull if enough points
            if len(ref_coords) >= 4 and ConvexHull:
                try:
                    hull = ConvexHull(ref_coords[['x','y','z']].values)
                    # For mesh3d, we need to extract vertex indices
                    # But simpler way for ConvexHull is to just plot vertices, 
                    # Mesh3d needs i,j,k indices which ConvexHull.simplices provides
                    
                    x_hull = ref_coords['x'].values
                    y_hull = ref_coords['y'].values
                    z_hull = ref_coords['z'].values
                    
                    simplices = hull.simplices
                    
                    fig.add_trace(go.Mesh3d(
                        x=x_hull, y=y_hull, z=z_hull,
                        i=simplices[:,0], j=simplices[:,1], k=simplices[:,2],
                        color='#00E5FF', opacity=0.1,
                        name='Phylogenetic Envelope',
                        hoverinfo='skip'
                    ))
                except Exception as e:
                    pass

        # C. Novel Cluster Centers - Neon Pink #FF007A
        if not cluster_coords.empty:
            for i, row in cluster_coords.iterrows():
                cid = cluster_map.get(i, f"Group {i}")
                fig.add_trace(go.Scatter3d(
                    x=[row['x']], y=[row['y']], z=[row['z']],
                    mode='markers+text',
                    marker=dict(size=8, color='#FF007A', symbol='diamond-open', line=dict(width=2)),
                    text=[cid],
                    textposition="top center",
                    textfont=dict(color="#FF007A", size=10),
                    name=f"NTU: {cid}"
                ))

        # D. The Query - Dynamic Color
        # Pink if Novel, Green if Known
        c_q = '#FF007A' if is_novel else '#00FF7F' 
        s_q = 'diamond' if is_novel else 'circle'
        status_txt = "NOVEL LINEAGE" if is_novel else "KNOWN SPECIES"
        
        fig.add_trace(go.Scatter3d(
            x=[query_coords['x']], y=[query_coords['y']], z=[query_coords['z']],
            mode='markers',
            marker=dict(size=15, color=c_q, symbol=s_q, line=dict(width=2, color='white'), opacity=1.0),
            name='Active Sequence',
            text=[f"<b>QUERY TARGET</b><br>{query_display_name}<br>Status: {status_txt}"],
            hoverinfo='text'
        ))

        # E. Evolutionary Line (Visual Guide from Query to Nearest Ref)
        if not ref_coords.empty:
            # Re-find the closest ref in PCA space (Euclidean)
            # This is purely visual
            q_pt = np.array([query_coords['x'], query_coords['y'], query_coords['z']])
            refs_pts = ref_coords[['x', 'y', 'z']].values
            
            dists = np.linalg.norm(refs_pts - q_pt, axis=1)
            nearest_idx = np.argmin(dists)
            nearest = ref_coords.iloc[nearest_idx]

            fig.add_trace(go.Scatter3d(
                x=[query_coords['x'], nearest['x']],
                y=[query_coords['y'], nearest['y']],
                z=[query_coords['z'], nearest['z']],
                mode='lines',
                line=dict(color='white', width=2, dash='dot'),
                hoverinfo='skip',
                showlegend=False
            ))

        # Layout Styling - Scientific / Lab
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                x=0.05, y=0.95,
                font=dict(color='#94A3B8', family="Consolas"),
                bgcolor='rgba(15, 23, 42, 0.8)',
                bordercolor='#334155',
                borderwidth=1
            )
        )
        
        return fig
