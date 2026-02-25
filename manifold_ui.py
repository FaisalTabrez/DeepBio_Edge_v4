import os
import lancedb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# @UX-Visionary: Phase 3 - Genomic Manifold Optimization (100k Scale)
print("Initializing Deep-Sea Bioluminescent Manifold...")

DB_DIR = r"E:\DeepBio_Scan\data\db"

def get_stratified_sample(table, sample_size=5000):
    """
    Pulls a stratified sample proportional to the Phylum distribution.
    Fits PCA on the full 100k dataset, then projects the sample.
    """
    print("Loading 100k Reference Atlas for PCA fitting...")
    # Load the full dataset to fit the PCA accurately across the entire variance
    df_full = table.to_pandas()
    
    print("Fitting PCA(n_components=3) on the full 100k dataset...")
    pca = PCA(n_components=3)
    X_full = np.stack(df_full['Vector'].values)
    pca.fit(X_full)
    
    print(f"Extracting stratified sample of ~{sample_size} points based on Phylum...")
    # Calculate proportions
    phylum_counts = df_full['Phylum'].value_counts(normalize=True)
    
    # Sample proportionally
    sampled_dfs = []
    for phylum, prop in phylum_counts.items():
        n_samples = max(1, int(prop * sample_size))
        phylum_df = df_full[df_full['Phylum'] == phylum]
        if len(phylum_df) > n_samples:
            sampled_dfs.append(phylum_df.sample(n_samples, random_state=42))
        else:
            sampled_dfs.append(phylum_df)
            
    df_sample = pd.concat(sampled_dfs)
    
    # Transform the sample using the pre-fitted PCA
    print("Projecting sample into 3D space...")
    X_sample = np.stack(df_sample['Vector'].values)
    X_pca = pca.transform(X_sample)
    
    df_sample['PCA_x'] = X_pca[:, 0]
    df_sample['PCA_y'] = X_pca[:, 1]
    df_sample['PCA_z'] = X_pca[:, 2]
    
    return df_sample, pca

def plot_bioluminescent_manifold(df_sample, query_points=None):
    """
    Renders the 3D Scatter plot with WebGL, small markers, and low opacity
    to create the 'Bioluminescent Cloud' effect.
    """
    print("Rendering Bioluminescent Cloud UI...")
    
    fig = go.Figure()
    
    # Create a glowing color map for Phyla
    phyla = df_sample['Phylum'].unique()
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Vivid
    color_map = {phylum: colors[i % len(colors)] for i, phylum in enumerate(phyla)}
    
    # 1. Background Cloud (Stratified Sample)
    for phylum in phyla:
        df_p = df_sample[df_sample['Phylum'] == phylum]
        fig.add_trace(go.Scatter3d(
            x=df_p['PCA_x'], y=df_p['PCA_y'], z=df_p['PCA_z'],
            mode='markers',
            name=phylum,
            text=df_p['ScientificName'],
            hoverinfo='text+name',
            marker=dict(
                size=2,           # Small marker size
                opacity=0.5,      # Low opacity for cloud effect
                color=color_map[phylum],
                line=dict(width=0)
            )
        ))
        
    # 2. Query Points / HDBScan Clusters (if provided)
    if query_points is not None:
        fig.add_trace(go.Scatter3d(
            x=query_points['PCA_x'], y=query_points['PCA_y'], z=query_points['PCA_z'],
            mode='markers',
            name='Query / Clusters',
            text=query_points['Label'],
            marker=dict(
                size=6,
                opacity=1.0,
                color='#00FFFF', # Cyan glow for queries
                symbol='diamond',
                line=dict(color='white', width=1)
            )
        ))

    # 3. UI Styling - Deep Sea Theme
    fig.update_layout(
        title=dict(
            text="Genomic Manifold (100k Reference Atlas)",
            font=dict(color="white", size=20)
        ),
        template="plotly_dark",
        paper_bgcolor='#050505', # Deep sea black/gray
        plot_bgcolor='#050505',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False),
            bgcolor='#050505'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            itemsizing='constant', 
            font=dict(color='rgba(255,255,255,0.7)'),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

if __name__ == "__main__":
    # Connect to LanceDB
    db = lancedb.connect(DB_DIR)
    table = db.open_table("reference_atlas_v100k")
    
    # Get sample and PCA model
    df_sample, pca_model = get_stratified_sample(table, sample_size=5000)
    
    # Generate the UI
    fig = plot_bioluminescent_manifold(df_sample)
    
    # Save to HTML for viewing
    output_html = "genomic_manifold.html"
    fig.write_html(output_html)
    print(f"\n[SUCCESS] Manifold saved to {output_html}.")
    print("Open this file in your browser to view the 3D Bioluminescent Cloud!")