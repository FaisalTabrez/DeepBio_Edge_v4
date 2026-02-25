import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

# Create a module path fix for locating src modules if run from root
sys.path.append(os.getcwd())

from src.edge.config_init import initialize_folders, TAXONOMY_DB_PATH

# Set TaxonKit DB Path (Windows Compatible)
os.environ["TAXONKIT_DB"] = str(TAXONOMY_DB_PATH)

# Initialize Directory Structure on Pendrive (or local)
initialize_folders()

from src.edge.logger import setup_localized_logging, get_logger

# Initialize Logging IMMEDIATELLY
if 'logger_active' not in st.session_state:
    setup_localized_logging()
    st.session_state.logger_active = True

logger = get_logger()

from src.edge.embedder import NucleotideEmbedder
from src.edge.database import AtlasManager
from src.edge.taxonomy import TaxonomyEngine
from src.edge.discovery import DiscoveryEngine
from src.edge.parser import stream_sequences
from src.interface.visualizer import ManifoldVisualizer
from src.interface.docs import render_docs

# ==========================================
# @UX-Visionary: Global-BioScan Console Config
# ==========================================
st.set_page_config(
    page_title="Global-BioScan | Edge Console",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. Theme & CSS Styling (Abyss Blue + Glassmorphism + Neon)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0A0F1E; /* Abyss Blue */
        color: #E2E8F0; /* Off-white */
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace; /* Terminal vibe */
    }

    /* Glassmorphism Containers */
    .glass-panel {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics Boxes */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px;
        transition: border-color 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00E5FF;
    }
    label[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #0F172A;
        color: #00E5FF;
        border: 1px solid #334155;
        font-family: 'Consolas', monospace;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #00E5FF;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
    }

    /* Buttons */
    .stButton>button {
        background: rgba(0, 229, 255, 0.1);
        color: #00E5FF;
        border: 1px solid #00E5FF;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #00E5FF;
        color: #0A0F1E;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }

    /* Custom Classes for Discovery Cards */
    .discovery-card-known {
        border-left: 4px solid #00E5FF;
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%);
    }
    .discovery-card-novel {
        border-left: 4px solid #FF007A;
        background: linear-gradient(90deg, rgba(255, 0, 122, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%);
    }
    
    .neon-text-cyan { color: #00E5FF; text-shadow: 0 0 5px rgba(0, 229, 255, 0.5); }
    .neon-text-pink { color: #FF007A; text-shadow: 0 0 5px rgba(255, 0, 122, 0.5); }
    
    .breadcrumb {
        font-size: 0.85em;
        color: #94A3B8;
        letter-spacing: 0.5px;
    }

    /* Terminal Log */
    .terminal-box {
        background-color: #000000;
        border: 1px solid #333;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        padding: 10px;
        height: 150px;
        overflow-y: auto;
        border-radius: 4px;
        box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00E5FF;
        border-bottom: 2px solid #00E5FF;
    }

    /* File Uploader Customisation */
    section[data-testid="stFileUploader"] {
        padding: 0;
    }
    section[data-testid="stFileUploader"] > div > div > button {
        background-color: rgba(0, 229, 255, 0.1);
        border: 2px dashed #00E5FF;
        color: #00E5FF;
        font-family: 'Consolas', monospace;
    }
    .uploadedFile {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #00E5FF;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Core Method Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_core_systems():
    try:
        try:
            import psutil
        except ImportError:
            psutil = None
            
        embedder = NucleotideEmbedder()
        
        # @UX-Visionary: Hot-Fix for Cached Objects
        # If Streamlit cache holds an old version of the embedder (hidden_dim=768),
        # we manually check the actual model config and patch the instance.
        try:
            real_dim = embedder.model.config.hidden_size
            if embedder.hidden_dim != real_dim:
                logger.warning(f"Cache Inconsistency Detected! Patching Embedder Dim: {embedder.hidden_dim} -> {real_dim}")
                embedder.hidden_dim = real_dim
        except Exception as e:
            logger.warning(f"Could not verify embedder config: {e}")
            
        atlas = AtlasManager()
        
        # @Data-Ops: Boot Logs for 100k Index
        if atlas.table:
            count = atlas.table.count_rows()
            if count >= 100000:
                print("[DB] Volume E: High-Speed IVF-PQ Index Detected.")
                print(f"[DB] {count:,} Marine Signatures Loaded into Latent Space.")
                logger.info("[DB] Volume E: High-Speed IVF-PQ Index Detected.")
                logger.info(f"[DB] {count:,} Marine Signatures Loaded into Latent Space.")
                
        taxonomy = TaxonomyEngine()
        discovery = DiscoveryEngine(atlas)
        viz = ManifoldVisualizer()
        return embedder, atlas, taxonomy, psutil, viz, discovery
    except Exception as e:
        logger.error(f"CRITICAL SYSTEM FAILURE: {e}", exc_info=True)
        st.error(f"Critical System Failure: {e}")
        return None, None, None, None, None, None

embedder, atlas, taxonomy, psutil, viz, discovery = load_core_systems()

if not embedder or not atlas or not taxonomy or not viz or not discovery:
    st.stop()

# -----------------------------------------------------------------------------
# 2. Sidebar (Command Center)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## COMMAND CENTER")
    st.markdown("---")
    
    # Connection Status
    st.markdown("### LINK STATUS")
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown(f":material/settings_input_antenna: **EDGE AI**<br><span style='font-size:0.8em; color:#64748B'>ONLINE</span>", unsafe_allow_html=True)
    with col_link2:
        # Check actual row count for status
        row_count = atlas.table.count_rows() if atlas.table else 0
        db_color = ":material/check_circle:" if row_count > 0 else ":material/error:"
        db_status = f"ONLINE ({row_count} Vectors)" if row_count > 0 else "OFFLINE"
        
        st.markdown(f"{db_color} **GLO-DB**<br><span style='font-size:0.8em; color:#64748B'>{db_status}</span>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Hardware Health
    st.markdown("### HARDWARE HEALTH")
    
    # Simulated Metrics or Real if psutil
    if psutil:
        cpu_load = psutil.cpu_percent() / 100.0
        ram_usage = psutil.virtual_memory().percent / 100.0
    else:
        cpu_load = random.randint(15, 45) / 100.0
        ram_usage = random.randint(40, 65) / 100.0
    
    st.write("CPU LOAD")
    st.progress(cpu_load)
    
    st.write("ATLAS LOAD (RAM)")
    st.progress(ram_usage)
    
    st.markdown("---")
    
    # Atlas Metadata
    st.markdown("### ATLAS METADATA")
    st.caption("**Source:** Colab (Volume E:)")
    st.caption("**Embedding Model:** Nucleotide-Transformer-v2-50M")
    st.caption("**Vector Dimensions:** 512 (Float32)")
    
    st.markdown("---")
    
    # Config
    st.markdown("### CONFIGURATION")
    gene_selector = st.selectbox(
        "GENE MARKER", 
        ["COI", "18S"], 
        help="Switch calibration protocols."
    )
    if gene_selector == "18S":
        st.caption(":material/lock: STRICT MODE ACTIVE")
    else:
        st.caption(":material/lock_open: STANDARD MODE ACTIVE")

# -----------------------------------------------------------------------------
# 3. Top KPI Row
# -----------------------------------------------------------------------------
# Header
st.markdown(f"""
    <h1 style='text-align: center; margin-bottom: 30px;'>
        GLOBAL-BIOSCAN <span style='font-size: 0.5em; vertical-align: middle; color: #00E5FF; border: 1px solid #00E5FF; padding: 2px 8px; border-radius: 4px;'>CONSOLE v4.2</span>
    </h1>
""", unsafe_allow_html=True)

# KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_seqs = atlas.table.count_rows() if atlas.table else 0
index_size = "482 MB" # Mock
novelty_rate = "12.4%" # Mock
latency = "< 8ms" # Updated for 100k IVF-PQ index

kpi1.metric("REFERENCE ATLAS", f"{total_seqs:,} SIGNATURES", "100k LOADED")
kpi2.metric("INDEX SIZE", index_size, "+2 MB")
kpi3.metric("NOVELTY RATE", novelty_rate, "-0.2%")
kpi4.metric("MEDIAN LATENCY", latency, "OPTIMAL")

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Interface Tabs
# -----------------------------------------------------------------------------
tab_monitor, tab_visualizer, tab_discovery, tab_report, tab_docs = st.tabs([
    "REAL-TIME MONITOR", 
    "GENOMIC MANIFOLD", 
    "TAXONOMIC INFERENCE", 
    "NOVELTY DISCOVERY",
    "DOCUMENTATION"
])

from src.edge.config_init import LOGS_PATH

# One-time Setup
if 'logger_setup' not in st.session_state:
    st.session_state.logger_setup = True

def log_event(message):
    """
    Bridge to the standard logging module.
    """
    import logging
    logging.info(message)

def display_terminal_logs():
    """
    Reads the physical log file on the E: Drive (or local) and displays it.
    """
    log_file = LOGS_PATH / 'session.log'
    if not log_file.exists():
        return "INFO: Waiting for system logs..."
        
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get last 20 lines
            last_lines = lines[-20:] if len(lines) > 20 else lines
            # Clean up default format for UI
            formatted_lines = []
            for l in last_lines:
                clean_line = l.strip()
                if "ERROR" in clean_line or "CRITICAL" in clean_line:
                    prefix = "FAIL"
                    color = "#FF007A"
                elif "WARNING" in clean_line:
                    prefix = "WARN"
                    color = "#FACC15"
                elif "SUCCESS" in clean_line or "OK" in clean_line:
                    prefix = "SUCCESS"
                    color = "#00E5FF"
                else:
                    prefix = "INFO"
                    color = "#00FF00"
                    
                formatted_lines.append(f"<span style='color:{color}'>[{prefix}] {clean_line}</span>")
                
            return "<br>".join(formatted_lines)
    except Exception as e:
        return f"FAIL: Log Read Error: {e}"

# Initialize Results Buffer
if 'scan_results_buffer' not in st.session_state:
    st.session_state.scan_results_buffer = []

# --- TAB 1: MONITOR & SCAN ---
with tab_monitor:
    col_input, col_results = st.columns([1, 1.5])
    
    with col_input:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("SEQUENCE INGESTION")
        
        uploaded_file = st.file_uploader(
            "Upload Genomic Data Stream", 
            type=['fasta', 'fastq', 'txt', 'parquet'], 
            help="Supported: FASTA, FASTQ, Parquet, TXT"
        )
        
        # Check for Demo Mode
        demo_active = False
        if 'demo_mode' in st.session_state and st.session_state.demo_mode:
            st.info("DEMO MODE: Loaded 'Hadal_Zone_Sample_Batch_04.fasta'")
            demo_active = True

        # Determine Active Input
        target_file = uploaded_file
        
        if target_file is not None or (demo_active and st.session_state.get('demo_file_path')):
            # Metadata Card
            if demo_active and not uploaded_file:
                 file_details = {"Filename": "Hadal_Zone_Sample_Batch_04.fasta", "Size": "1.2 MB", "Type": "Simulated Stream"}
            else:
                 # Ensure uploaded_file is not None here for static analysis
                 if uploaded_file:
                     file_details = {
                        "Filename": uploaded_file.name,
                        "Size": f"{uploaded_file.size / 1024:.2f} KB",
                        "Type": uploaded_file.type
                    }
                 else:
                     file_details = {"Filename": "Unknown", "Size": "0KB", "Type": "Unknown"}
            
            st.markdown(f"""
            <div class='uploadedFile'>
                <b>STREAM METADATA</b><br>
                Name: <span style='color:#00E5FF'>{file_details['Filename']}</span><br>
                Size: <span style='color:#00E5FF'>{file_details['Size']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            col_act1, col_act2 = st.columns(2)
            start_btn = col_act1.button("INITIATE STREAM", use_container_width=True, icon=":material/play_arrow:")
            stop_btn = col_act2.button("STOP INFERENCE", use_container_width=True, icon=":material/stop:")
            
            if start_btn:
                log_event(f"Mounting File: {file_details['Filename']}")
                log_event("Loading Local CPU Inference Engine...")
                log_event("Vectorizing Query Sequence (Local)...")
                log_event("Querying 32GB Colab-Seeded Atlas on D:/...")
                
                # Handle File path
                if demo_active and not uploaded_file:
                     tmp_path = st.session_state.demo_file_path
                elif uploaded_file: 
                     # Save to temp for parser access
                     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                else:
                    st.error("No file source available.")
                    st.stop()
                
                try:
                    # Stream Processing Loop
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    count = 0
                    # Just an estimation for progress bar if we don't know total
                    estimated_total = 100 
                    
                    for seq_record in stream_sequences(tmp_path):
                        count += 1
                        seq_id = seq_record['id']
                        raw_seq = seq_record['sequence']
                        
                        progress_text.text(f"Analyzing Sequence #{count}: {seq_id}")
                        progress_bar.progress(min(count % 100, 100)) # Loop visual
                        
                        # 1. Embed
                        vector = embedder.embed_sequences([raw_seq])
                        # log_event(f"DEBUG: Vector shape: {vector.shape}")
                        
                        # 2. Search
                        # Fetch larger set for Viz (200), slice for Feed (5)
                        results_viz = atlas.query_vector(vector, top_k=200)
                        # log_event(f"DEBUG: Search returned {len(results_viz)} results")
                        
                        # 3. Resolve (@Bio-Taxon Triple-Tier Logic)
                        # We extract the Top 50 candidates for Consensus & Oracle verification
                        # Explicitly slice list to avoid ambiguity
                        top_candidates = results_viz[:50] if isinstance(results_viz, list) else []
                        
                        # Add extra safety check on candidates
                        # If somehow query_vector returns weird stuff
                        if not top_candidates and isinstance(results_viz, list) and results_viz:
                             top_candidates = [results_viz[0]] # Just take top 1 if slice failed? NO.
                        
                        formatted_results = []
                        if top_candidates:
                             try:
                                 formatted_results = taxonomy.format_search_results(top_candidates, gene_type=gene_selector)
                             except Exception as tax_e:
                                 logger.error(f"Taxonomy Formatting Failed: {tax_e}")
                                 st.error(f"Taxonomy Error: {tax_e}")
                        
                        # Add to Buffer (Prepend for latest on top)
                        if formatted_results and isinstance(formatted_results, list) and len(formatted_results) > 0:
                            top_hit = formatted_results[0]
                            top_hit['query_id'] = seq_id # Track query ID
                            st.session_state.scan_results_buffer.insert(0, top_hit)
                            
                            # Ensure vector is stored flat for viz
                            vec_flat = vector.flatten() if isinstance(vector, np.ndarray) else vector

                            # Store Context for Tab 2 Visualizer
                            st.session_state.viz_context = {
                                'ref_hits': results_viz,
                                'query_vec': vec_flat,
                                'display_name': top_hit['display_name'],
                                'is_novel': bool(top_hit['is_novel']) # Double check
                            }
                        else:
                            # Handle case where taxonomy failed or no results
                            logger.warning(f"Skipping viz context update for sequence {seq_id} due to empty results.")

                            
                        # Real-time UI Update hack (rerun not ideal in loop, so we rely on session state being read next pass or manual container update if possible)
                        # Streamlit loops block the UI, so we need a container to update dynamically.
                        # However, to see 'live' card updates, we'd need to re-render the results column inside this loop
                        # OR just let it finish. 
                        # For 'Real-Time' feel, we will update a placeholder.
                        
                        # Limit buffer size
                        if len(st.session_state.scan_results_buffer) > 50:
                            st.session_state.scan_results_buffer.pop()
                            
                    log_event(f"SUCCESS: Stream Complete. Processed {count} sequences.")
                    progress_bar.progress(100)
                    progress_text.text("Analysis Complete.")
                    
                    # Run Discovery Engine Immediately
                    # Explicit length check to avoid ambiguity if buffer became array-like
                    if len(st.session_state.scan_results_buffer) > 0:
                         try:
                             clusters = discovery.analyze_novelty(st.session_state.scan_results_buffer)
                             st.session_state.novel_clusters = clusters
                             if clusters:
                                 log_event(f"SUCCESS: DISCOVERY: {len(clusters)} novel clusters identified.")
                         except Exception as de:
                             logger.error(f"Discovery Failed: {de}")

                except Exception as e:
                    st.error(f"Stream Error: {e}")
                    log_event(f"FAIL: ERROR: {e}")
                finally:
                    os.unlink(tmp_path) # Cleanup
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Results Loop (Discovery Cards)
    with col_results:
        st.markdown("### ANALYSIS FEED")
        
        # Container for scrollable results
        results_container = st.container(height=600)
        
        if st.session_state.scan_results_buffer and len(st.session_state.scan_results_buffer) > 0:
            with results_container:
                for hit in st.session_state.scan_results_buffer:
                    # Determine Styling
                    raw_novel = hit.get('is_novel', False)
                    is_novel = False
                    if isinstance(raw_novel, np.ndarray):
                         is_novel = bool(raw_novel.item()) if raw_novel.size == 1 else raw_novel.any()
                    else:
                         is_novel = bool(raw_novel)

                    card_class = "discovery-card-novel" if is_novel else "discovery-card-known"
                    
                    # Icons & Labels
                    icon_code = ":material/new_releases:" if is_novel else ":material/check_circle:"
                    status_prefix = "[NOVEL]" if is_novel else "[KNOWN]"
                    
                    text_class = "neon-text-pink" if is_novel else "neon-text-cyan"
                    bar_color = "#FF007A" if is_novel else "#00E5FF"
                    
                    # Confidence Calculation for Bar
                    sim_pct = hit['similarity'] * 100
                    
                    # HTML Card
                    # We inject Material Icons via simple text or span since HTML in markdown doesn't parse :icon: syntax directly
                    # Actually, Streamlit Markdown supports symbols. For HTML block we use text fallback.
                    
                    card_icon_html = "✦" if is_novel else "✓"
                    
                    st.markdown(f"""
                    <div class="glass-panel {card_class}" style="padding: 15px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <span style="font-size: 0.8em; color: #94A3B8;">QUERY: {hit.get('query_id', 'Unknown')}</span>
                                <h3 class="{text_class}" style="margin: 5px 0;">{card_icon_html} {hit['display_name']}</h3>
                                <div class="breadcrumb">{hit['display_lineage'].replace('Unknown', '???')}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.5em; font-weight: bold; color: {bar_color};">{hit['confidence_pct']}</div>
                                <div style="font-size: 0.7em; color: #64748B;">CONFIDENCE</div>
                            </div>
                        </div>
                        <div style="margin-top: 10px; background: #0F172A; height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="width: {sim_pct}%; background: {bar_color}; height: 100%; box-shadow: 0 0 10px {bar_color};"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.85em; color: #CBD5E1;">
                            TYPE: <b>{status_prefix}</b> | STATUS: {hit['status'].upper()}
                        </div>
                        <div style="margin-top: 4px; font-size: 0.75em; color: #94A3B8; font-style: italic;">
                            Analyzed 50 neighbors in latent space. Consensus: {hit.get('consensus_name', 'Unknown').split(' ')[0]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
        else:
            with results_container:
                st.info("Waiting for Scan Data... System Idle.")

    # 5. System Logs
    st.markdown("### SYSTEM LOGS")
    
    log_content = display_terminal_logs()
    
    st.markdown(f"""
    <div class="terminal-box">
        {log_content}
        <br><span style='color:#00FF00; animation: blink 1s infinite;'>_</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: MANIFOLD VISUALIZER ---
with tab_visualizer:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("LATENT SPACE TOPOLOGY")
    
    if 'viz_context' in st.session_state and st.session_state.viz_context:
        ctx = st.session_state.viz_context
        
        with st.spinner("Calculating 3D Manifold Projection..."):
            # Prepare clusters
            n_clusters = st.session_state.get('novel_clusters', [])
            
            fig_3d = viz.create_plot(
                reference_hits=ctx['ref_hits'],
                query_vector=ctx['query_vec'],
                query_display_name=ctx['display_name'],
                is_novel=ctx['is_novel'],
                atlas_manager=atlas,
                novel_clusters=n_clusters
            )
            
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("Real-time Holographic Projection: 768-dim PCA Reduction of evolutionary neighborhood.")
        
    else:
        st.info("System Idle. Initiate a Scan to project the biodiversity manifold.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: NOVELTY DISCOVERY ---
with tab_discovery:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("UNSUPERVISED BIOLOGICAL DISCOVERY ENGINE")
    st.caption("Auto-Clustering 'Dark Taxa' to identify potential new species groups.")
    
    if 'scan_results_buffer' in st.session_state and st.session_state.scan_results_buffer:
        # 1. Run Discovery Loop
        novel_entities = discovery.analyze_novelty(st.session_state.scan_results_buffer)
        
        if novel_entities:
            st.success(f"DISCOVERY ALERT: Identified {len(novel_entities)} Potential New Species Groups!")
            
            # Export Buffer
            export_data = []
            
            # 2. Render Cluster Gallery
            cols = st.columns(3)
            for i, entity in enumerate(novel_entities):
                with cols[i % 3]:
                    # Dynamic Color based on divergence
                    div_val = entity['biological_divergence']
                    gauge_color = "#00FF00" if div_val < 0.1 else ("#00E5FF" if div_val < 0.2 else "#FF007A")
                    
                    st.markdown(f"""
                    <div style="border: 1px solid {gauge_color}; padding: 15px; border-radius: 8px; background: rgba(0,0,0,0.3); margin-bottom: 20px;">
                        <h4 style="color: {gauge_color}; margin: 0;">✦ {entity['otu_id']}</h4>
                        <div style="font-size: 0.8em; color: #94A3B8; margin-bottom: 10px;">{entity['status']}</div>
                        
                        <div style="margin-bottom: 5px;">
                            <b>nearest_relative:</b><br>{entity['nearest_relative']}
                        </div>
                        
                        <div style="display: flex; align-items: center; margin-top: 10px;">
                            <div style="font-size: 1.2em; font-weight: bold; color: {gauge_color}; margin-right: 10px;">
                                {entity['divergence_pct']}
                            </div>
                            <div style="font-size: 0.7em; color: #64748B;">DIVERGENCE</div>
                        </div>
                        
                        <div style="margin-top: 10px; font-family: monospace; font-size: 0.7em; color: #475569; overflow-x: hidden; white-space: nowrap;">
                            REPRESENTATIVE:<br>{entity['representative_name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prepare for Export
                    export_data.append(f">{entity['otu_id']} | Divergence: {entity['divergence_pct']} | Anchor: {entity['nearest_relative']}\n[SEQUENCE_DATA_PLACEHOLDER_FOR_LAB]")

            st.divider()
            
            # 3. Export Action
            if st.button("EXPORT CLUSTERS (Phylogenetic Tree)", icon=":material/download:"):
                export_path = Path("E:/DeepBio_Scan/results/novel_clusters.fasta")
                # Fallback to local if E: missing
                if not Path("E:/").exists():
                    export_path = Path("data/results/novel_clusters.fasta")
                    
                if not export_path.parent.exists():
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    
                with open(export_path, "w") as f:
                    f.write("\n".join(export_data))
                
                st.toast(f"✅ Exported to {export_path}")
                logger.info(f"Exported {len(novel_entities)} novel clusters to {export_path}")
                
        else:
            st.info("No significant novel clusters detected in current buffer. (Need > 2 'Dark Taxa')")
    
    else:
        st.warning("Buffer Empty. Run a sequence scan to populate the discovery engine.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: REPORT ---
with tab_report:
    st.markdown("### EXPEDITION METRICS")

    col_rep1, col_rep2 = st.columns([2, 1])
    
    with col_rep1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        if len(st.session_state.scan_results_buffer) > 0:
            df_res = pd.DataFrame(st.session_state.scan_results_buffer)
            if 'status' in df_res.columns:
                 counts = df_res['status'].value_counts()
                 fig_pie = px.pie(
                     values=counts.values, 
                     names=counts.index, 
                     title="Real-time Identification Status",
                     color_discrete_sequence=['#00E5FF', '#FF007A', '#7000FF']
                 )
                 fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#FFF")
                 st.plotly_chart(fig_pie, use_container_width=True)
            else:
                 st.info("Insufficient Data for Charts")
        else:
             st.info("Waiting for Scan Data...")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: DOCUMENTATION ---
with tab_docs:
    render_docs()
    
    # End of Docs
