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

    def get_neighborhood_manifold(self, query_vector: np.ndarray, query_id: str, atlas_manager: Any, top_k: int = 500) -> Dict[str, Any]:
        """
        Fetches the top_k nearest neighbors for a query and fits a localized PCA model.
        Returns the projected coordinates and metadata.
        """
        try:
            if not atlas_manager or not atlas_manager.db:
                logger.error("Atlas Manager not available for neighborhood fetch.")
                return {}

            table_name = "reference_atlas_v100k"
            try:
                table = atlas_manager.db.open_table(table_name)
            except Exception:
                table = atlas_manager.table # Fallback
                
            if not table:
                logger.error("LanceDB table not found.")
                return {}

            # 1. Fetch Neighborhood
            # Ensure query_vector is a list of floats for LanceDB
            if isinstance(query_vector, np.ndarray):
                query_list = query_vector.flatten().tolist()
            else:
                query_list = query_vector

            results = table.search(query_list).limit(top_k).to_pandas()
            
            if results.empty:
                logger.warning(f"No neighbors found for query {query_id}")
                return {}

            # Normalize columns
            results.columns = [c.lower() for c in results.columns]
            
            # Extract vectors and metadata
            neighbor_vectors = np.array(results['vector'].tolist(), dtype=np.float32)
            
            # 2. Localized PCA
            # Combine query and neighbors for PCA fit
            query_np = np.array(query_list, dtype=np.float32).reshape(1, -1)
            fit_data = np.vstack([query_np, neighbor_vectors])
            
            pca = PCA(n_components=3)
            coords = pca.fit_transform(fit_data)
            
            # 3. Data Structuring
            query_coords = coords[0]
            neighbor_coords = coords[1:]
            
            # Build metadata list
            neighbors_meta = []
            for i, row in results.iterrows():
                neighbors_meta.append({
                    'scientific_name': row.get('scientific_name', 'Unknown'),
                    'phylum': row.get('phylum', 'Unknown'),
                    'x': float(neighbor_coords[i, 0]),
                    'y': float(neighbor_coords[i, 1]),
                    'z': float(neighbor_coords[i, 2]),
                    'distance': float(row.get('_distance', 0.0))
                })
                
            manifold_data = {
                'query_id': query_id,
                'query_coords': {
                    'x': float(query_coords[0]),
                    'y': float(query_coords[1]),
                    'z': float(query_coords[2])
                },
                'neighbors': neighbors_meta,
                'explained_variance': pca.explained_variance_ratio_.tolist()
            }
            
            # Store in session state if available
            import streamlit as st
            if 'manifold_cache' not in st.session_state:
                st.session_state['manifold_cache'] = {}
            st.session_state['manifold_cache'][query_id] = manifold_data
            
            logger.info(f"Localized Manifold generated for {query_id} with {len(neighbors_meta)} neighbors.")
            return manifold_data
            
        except Exception as e:
            logger.error(f"Failed to generate neighborhood manifold: {e}")
            return {}

    def create_localized_plot(self, query_id: str) -> go.Figure:
        """
        Renders the High-Resolution Zoom manifold for a specific query.
        """
        import streamlit as st
        
        if 'manifold_cache' not in st.session_state or query_id not in st.session_state['manifold_cache']:
            return go.Figure()
            
        data = st.session_state['manifold_cache'][query_id]
        
        q_coords = data['query_coords']
        neighbors = data['neighbors']
        
        # Determine if novel based on session state context
        is_novel = st.session_state.get('active_viz_novel', False)
        query_name = st.session_state.get('active_viz_name', query_id)
        
        fig = go.Figure()
        
        # 1. Plot 500 Neighbors (Neon Blue)
        if neighbors:
            df_n = pd.DataFrame(neighbors)
            
            hover_text = [
                f"<b>{row['scientific_name']}</b><br>Phylum: {row['phylum']}<br>Dist: {row['distance']:.3f}"
                for _, row in df_n.iterrows()
            ]
            
            fig.add_trace(go.Scatter3d(
                x=df_n['x'], y=df_n['y'], z=df_n['z'],
                mode='markers',
                marker=dict(
                    size=4, 
                    color='#00E5FF', # Neon Blue
                    opacity=0.6,
                    line=dict(width=0.5, color='rgba(255,255,255,0.2)')
                ),
                hoverinfo='text',
                text=hover_text,
                name='Local Neighborhood (500)'
            ))
            
            # 2. Identity Volume (Mesh3d around top 10)
            if len(df_n) >= 10 and ConvexHull:
                top_10 = df_n.nsmallest(10, 'distance')
                
                # Include query in the hull
                hull_pts = np.vstack([
                    top_10[['x', 'y', 'z']].values,
                    [q_coords['x'], q_coords['y'], q_coords['z']]
                ])
                
                try:
                    hull = ConvexHull(hull_pts)
                    simplices = hull.simplices
                    
                    fig.add_trace(go.Mesh3d(
                        x=hull_pts[:, 0], y=hull_pts[:, 1], z=hull_pts[:, 2],
                        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
                        color='#00FF7F' if not is_novel else '#FF007A',
                        opacity=0.15,
                        alphahull=5,
                        name='Identity Volume',
                        hoverinfo='skip'
                    ))
                except Exception as e:
                    logger.warning(f"Could not draw Identity Volume: {e}")

        # 3. Plot Query
        c_q = '#FF007A' if is_novel else '#00FF7F' 
        s_q = 'diamond' if is_novel else 'circle'
        status_txt = "NOVEL LINEAGE" if is_novel else "KNOWN SPECIES"
        
        fig.add_trace(go.Scatter3d(
            x=[q_coords['x']], y=[q_coords['y']], z=[q_coords['z']],
            mode='markers',
            marker=dict(
                size=18, 
                color=c_q, 
                symbol=s_q, 
                line=dict(width=3, color='white'), 
                opacity=1.0
            ),
            name='Active Sequence',
            text=[f"<b>QUERY TARGET</b><br>{query_name}<br>Status: {status_txt}"],
            hoverinfo='text'
        ))
        
        # Layout Styling
        layout_args: Dict[str, Any] = dict(
            scene=dict(
                xaxis=dict(visible=True, showgrid=True, gridcolor='#334155', showbackground=False, zeroline=False, showticklabels=False, title=''),
                yaxis=dict(visible=True, showgrid=True, gridcolor='#334155', showbackground=False, zeroline=False, showticklabels=False, title=''),
                zaxis=dict(visible=True, showgrid=True, gridcolor='#334155', showbackground=False, zeroline=False, showticklabels=False, title=''),
                bgcolor='rgba(0,0,0,0)',
                aspectmode='data',
                camera=dict(
                    center=dict(x=q_coords['x'], y=q_coords['y'], z=q_coords['z']),
                    eye=dict(x=q_coords['x'] + 1.0, y=q_coords['y'] + 1.0, z=q_coords['z'] + 1.0)
                )
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
        
        fig.update_layout(**layout_args)
        return fig
