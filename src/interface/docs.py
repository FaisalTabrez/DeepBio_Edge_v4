import streamlit as st

def render_docs():
    """
    Renders the 'DeepBio-Scan: Technical Monograph & User Manual' documentation page.
    Adheres to @BioArch (scientific accuracy) and @UX-Visionary (high-tech aesthetic) personas.
    """
    
    # --- header ---
    st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="font-family: 'Helvetica Neue', sans-serif; letter-spacing: 2px; color: #E2E8F0;">
                DEEPBIO-SCAN<br>
                <span style="font-size: 0.5em; color: #00E5FF; text-transform: uppercase; letter-spacing: 4px;">
                    Technical Monograph & User Manual
                </span>
            </h1>
            <p style="color: #94A3B8; font-style: italic; font-size: 1.1em;">
                "An AI-Driven Framework for Deep-Sea eDNA Taxonomic Inference and Novel Biodiversity Discovery."
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- Section 1: The Scientific Challenge ---
    st.markdown("## 1. 🌊 The Scientific Challenge: The Abyss Gap")
    
    st.info("""
    ### ⚠️ The Trinity of Abyssal Bottlenecks
    Current taxonomic methods fail in the deep ocean due to three critical systemic failures:
    
    1.  **The Reference Gap**: Less than **1%** of deep-sea marine nematodes and meiofauna have sequenced reference genomes. BLAST checks against "empty libraries".
    2.  **The Computational Wall**: Alignment-based algorithms (BLAST/DIAMOND) are **O(N)** complexity. They cannot run in real-time on edge devices during a research cruise.
    3.  **The Metadata Crisis**: Public repositories (NCBI/GenBank) are flooded with *"Unknown Eukaryote"* or *"Uncultured Marine Organism"* entries, creating a feedback loop of ignorance.
    """)

    st.markdown("---")

    # --- Section 2: The AI Engine ---
    col_ai1, col_ai2 = st.columns([1.5, 1])
    
    with col_ai1:
        st.markdown("## 2. 🧠 The AI Engine: DNA as a Language")
        st.markdown("""
        **Foundation Model: Nucleotide Transformer (v2-50M)**
        
        DeepBio-Scan treats genomic sequences not as strings of characters, but as a **biological language**. Just as GPT-4 understands the semantic link between "King" and "Queen", our model understands the evolutionary link between *18S rRNA* motifs.
        
        *   **Context-Aware Embedding**: We project raw ACTG sequences into a **768-dimensional latent space**.
        *   **Evolutionary Distance**: In this space, Cosine Similarity $\\approx$ Evolutionary Divergence. Two sequences with no shared ancestors are orthogonal ($90^\\circ$), while variants of the same species are parallel ($0^\\circ$).
        """)
        
    with col_ai2:
        st.markdown("""
        <div class="glass-panel" style="padding: 20px; text-align: center;">
            <h3 style="color: #00E5FF; margin: 0;">VECTOR SEARCH</h3>
            <p style="font-size: 0.9em; color: #CBD5E1;">Indexing Strategy</p>
            <hr style="border-color: #334155;">
            <div style="text-align: left; font-family: 'Consolas', monospace; font-size: 0.85em; color: #94A3B8;">
                ENGINE: <span style="color: #FACC15;">LanceDB (Rust)</span><br>
                INDEX:  <span style="color: #FACC15;">IVF-PQ (Inverted File Product Quantization)</span><br>
                STORAGE: <span style="color: #FACC15;">NVMe / USB 3.2 (Disk-Native)</span><br>
                SPEED: <span style="color: #00E5FF;">4.2M Vectors / 12ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 3: The Triple-Tier Hybrid Engine ---
    st.markdown("## 3. 🛡️ The Triple-Tier Hybrid Engine")
    st.markdown("To solve the *Metadata Crisis*, we don't trust a single source. We use a consensus mechanism:")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🥇 TIER 1: Consensus")
        st.markdown("""
        **"The Democracy of Vectors"**
        We query the nearest **50 neighbors** in the specific 18S/COI latent space.
        
        *   If >80% share a Family, we assign that Family.
        *   Filtering of outliers using *Local Outlier Factor (LOF)*.
        """)
        
    with c2:
        st.markdown("### 🥈 TIER 2: The Oracle")
        st.markdown("""
        **"The Marine Specialist"**
        Cross-validation against **WoRMS (World Register of Marine Species)**.
        
        *   Strips out terrestrial contaminants (e.g., *Homo sapiens*, *Canis lupus*).
        *   Enforces strict marine-only taxonomy.
        """)
        
    with c3:
        st.markdown("### 🥉 TIER 3: Lineage")
        st.markdown("""
        **"The Phylogeny Builder"**
        Reconstruction of full lineage tree using **TaxonKit**.
        
        *   *Kingdom > Phylum > Class > Order > Family > Genus > Species*
        *   Standardizes synonymy and resolves defunct taxIDs.
        """)

    st.markdown("---")

    # --- Section 4: The 'Dark Taxa' Protocol ---
    st.markdown("## 4. 🔦 The 'Dark Taxa' Protocol")
    st.markdown("How we handle the unknown represents our biggest innovation. We don't discard 'No Hits'—we cluster them.")

    # Color Decoding with Visuals
    col_proto1, col_proto2 = st.columns([1, 1])
    
    with col_proto1:
        st.markdown("""
        <div style="background: rgba(0, 229, 255, 0.1); border-left: 5px solid #00E5FF; padding: 15px; margin-bottom: 20px;">
            <h3 style="color: #00E5FF; margin: 0;">🩵 CONFIRMED MARINE TAXON</h3>
            <p style="margin: 5px 0; color: #E2E8F0;">
                High confidence match (>90%) to existing reference atlas. Validated by WoRMS as marine.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_proto2:
        st.markdown("""
        <div style="background: rgba(255, 0, 122, 0.1); border-left: 5px solid #FF007A; padding: 15px; margin-bottom: 20px;">
            <h3 style="color: #FF007A; margin: 0;">🩷 POTENTIAL DARK TAXON (NTU)</h3>
            <p style="margin: 5px 0; color: #E2E8F0;">
                <b>Novel Taxonomic Unit.</b> Strong biological signal (high quality read), but <85% similarity to <i>any</i> known Earth organism.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🔬 Unsupervised Discovery (HDBSCAN)")
    st.markdown("""
    Passes marked as <span style='color:#FF007A'>**Dark Taxa**</span> are fed into the **Discovery Engine**:
    1.  **Density-Based Clustering**: We look for 'islands' of unknown sequences in the latent space.
    2.  **NTU Generation**: If $\\ge 5$ sequences form a dense cluster ($P < 0.05$), a new **NTU (Novel Taxonomic Unit)** is minted.
    3.  **Visualization**: These appear as new glowing nodes in the 3D Manifold Visualizer.
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("DeepBio-Scan v4.2 | Edge-Native Taxonomic Inference | Built for Expedition Use")
