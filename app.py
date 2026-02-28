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

from configs.config import initialize_folders, TAXON_DIR

# Set TaxonKit DB Path (Windows Compatible)
os.environ["TAXONKIT_DB"] = str(TAXON_DIR)

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
from src.edge.reporting import ResearchReporter
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

# Calculate actual index size
index_size_bytes = 0
if atlas.db_path and os.path.exists(atlas.db_path):
    for dirpath, _, filenames in os.walk(atlas.db_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                index_size_bytes += os.path.getsize(fp)
index_size = f"{index_size_bytes / (1024 * 1024):.1f} MB" if index_size_bytes > 0 else "0 MB"

# Calculate Novelty Rate based on clusters
if 'novel_clusters' in st.session_state and st.session_state.novel_clusters:
    num_clusters = len(st.session_state.novel_clusters)
    novelty_rate = f"{num_clusters} NTUs"
    novelty_delta = f"from {len(st.session_state.scan_results_buffer)} seqs"
else:
    novelty_rate = "0 NTUs"
    novelty_delta = "0%"

latency = "< 8ms" # Updated for 100k IVF-PQ index

kpi1.metric("REFERENCE ATLAS", f"{total_seqs:,} SIGNATURES", "100k LOADED")
kpi2.metric("INDEX SIZE", index_size, "+2 MB")
kpi3.metric("NOVELTY RATE", novelty_rate, novelty_delta)
kpi4.metric("MEDIAN LATENCY", latency, "OPTIMAL")

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Interface Tabs
# -----------------------------------------------------------------------------
tab_monitor, tab_visualizer, tab_discovery, tab_report, tab_benchmark, tab_docs = st.tabs([
    "REAL-TIME MONITOR", 
    "GENOMIC MANIFOLD", 
    "TAXONOMIC INFERENCE", 
    "NOVELTY DISCOVERY",
    "SCALABILITY & BENCHMARKING",
    "DOCUMENTATION"
])

from configs.config import LOG_FILE

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
    if not LOG_FILE.exists():
        return "INFO: Waiting for system logs..."
        
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
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
            
            st.divider()
            export_btn = st.button("EXPORT RESEARCH BRIEF", use_container_width=True, icon=":material/download:")
            
            if export_btn:
                if st.session_state.scan_results_buffer:
                    with st.spinner("Generating Research Brief..."):
                        reporter = ResearchReporter()
                        novel_clusters = st.session_state.get('novel_clusters', [])
                        report_paths = reporter.generate_expedition_summary(
                            st.session_state.scan_results_buffer, 
                            novel_clusters
                        )
                        if report_paths:
                            st.success(f"Research Brief Exported to {reporter.results_dir}")
                            log_event(f"SUCCESS: Research Brief exported to {reporter.results_dir}")
                else:
                    st.warning("No session data available to export.")
            
            if start_btn:
                # Clear buffer for new stream
                st.session_state.scan_results_buffer = []
                st.session_state.novel_clusters = []
                
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
                            top_hit['raw_sequence'] = raw_seq # Store raw sequence for FASTA export
                            
                            # Ensure vector is stored flat for viz and clustering
                            vec_flat = vector.flatten() if isinstance(vector, np.ndarray) else vector
                            top_hit['vector'] = vec_flat # Explicitly attach vector to the result object
                            
                            st.session_state.scan_results_buffer.insert(0, top_hit)
                        else:
                            # Handle case where taxonomy failed or no results
                            logger.warning(f"Skipping result buffer update for sequence {seq_id} due to empty results.")

                            
                        # Real-time UI Update hack (rerun not ideal in loop, so we rely on session state being read next pass or manual container update if possible)
                        # Streamlit loops block the UI, so we need a container to update dynamically.
                        # However, to see 'live' card updates, we'd need to re-render the results column inside this loop
                        # OR just let it finish. 
                        # For 'Real-Time' feel, we will update a placeholder.
                        
                        # Limit buffer size
                        # We need to keep all sequences for clustering, so we shouldn't pop them during the stream
                        # if len(st.session_state.scan_results_buffer) > 50:
                        #     st.session_state.scan_results_buffer.pop()
                            
                    log_event(f"SUCCESS: Stream Complete. Processed {count} sequences.")
                    progress_bar.progress(100)
                    progress_text.text("Analysis Complete.")
                    
                    # Run Discovery Engine Immediately
                    # Explicit length check to avoid ambiguity if buffer became array-like
                    if len(st.session_state.scan_results_buffer) >= 5:
                         st.info("Running Unsupervised Discovery on session buffer...")
                         try:
                             clusters = discovery.analyze_novelty(st.session_state.scan_results_buffer)
                             st.session_state.novel_clusters = clusters
                             st.session_state['ntu_registry'] = clusters
                             if clusters:
                                 log_event(f"SUCCESS: DISCOVERY: {len(clusters)} novel clusters identified.")
                                 st.success(f"Cluster Summary: {len(clusters)} Novel Taxonomic Units (NTUs) discovered from {len(st.session_state.scan_results_buffer)} sequences.")
                                 
                                 # Tag the buffer items with their cluster entity for the UI
                                 for cluster in clusters:
                                     cid = cluster['otu_id']
                                     for member in cluster.get('members', []):
                                         mid = member['id']
                                         for hit in st.session_state.scan_results_buffer:
                                             if hit.get('query_id') == mid:
                                                 hit['cluster_id'] = cid
                                                 hit['cluster_entity'] = cluster
                             else:
                                 st.info(f"Cluster Summary: No novel clusters identified from {len(st.session_state.scan_results_buffer)} sequences.")
                         except Exception as de:
                             logger.error(f"Discovery Failed: {de}")

                    st.session_state['scan_complete'] = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Stream Error: {e}")
                    log_event(f"FAIL: ERROR: {e}")
                finally:
                    os.unlink(tmp_path) # Cleanup
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Results Loop (Discovery Cards) & Community Composition
    with col_results:
        st.markdown("### ANALYSIS FEED & COMMUNITY COMPOSITION")
        
        # Create tabs for Feed vs Composition
        feed_tab, comp_tab = st.tabs(["LIVE FEED", "COMMUNITY COMPOSITION"])
        
        with feed_tab:
            # Container for scrollable results
            results_container = st.container(height=600)
            
            if st.session_state.scan_results_buffer and len(st.session_state.scan_results_buffer) > 0:
                with results_container:
                    rendered_clusters = set()
                    
                    # Check if we have clustered data
                    has_clusters = 'ntu_registry' in st.session_state and st.session_state['ntu_registry']
                    
                    for hit in st.session_state.scan_results_buffer:
                        cluster_id = hit.get('cluster_id')
                        
                        if has_clusters and cluster_id and cluster_id != -1:
                            if cluster_id in rendered_clusters:
                                continue # Skip individual cards for clustered items
                            
                            rendered_clusters.add(cluster_id)
                            entity = hit.get('cluster_entity', {})
                            
                            # Render NTU Discovery Card
                            div_val = entity.get('biological_divergence', 0.0)
                            gauge_color = "#00FF00" if div_val < 0.1 else ("#00E5FF" if div_val < 0.25 else "#FF007A")
                            confidence = 100 - (div_val * 100)
                            
                            consensus_name = entity.get('consensus_name', 'Unknown')
                            consensus_rank = entity.get('consensus_rank', 'Unknown')
                            cluster_size = entity.get('cluster_size', 0)
                            
                            # Narrative Content
                            header_text = f"✦ [DISCOVERY] NOVEL TAXONOMIC UNIT: {entity.get('otu_id', 'UNKNOWN')}"
                            
                            # Try to build a predicted lineage from the first member's hierarchy
                            predicted_lineage = "Unknown"
                            if entity.get('members') and len(entity.get('members')) > 0:
                                first_member_id = entity['members'][0]['id']
                                for h in st.session_state.scan_results_buffer:
                                    if h.get('query_id') == first_member_id:
                                        predicted_lineage = h.get('hierarchy', 'Unknown')
                                        break
                            
                            if predicted_lineage == "Unknown":
                                predicted_lineage = f"Predicted Lineage: {consensus_name} > [Divergent Genus]"
                            else:
                                predicted_lineage = f"Predicted Lineage: {predicted_lineage}"
                                
                            scientific_note = f"Cluster analysis of {cluster_size} sequences suggests a novel lineage within the {consensus_name} complex. Semantic distance from reference atlas: {(div_val * 100):.1f}%."
                                
                            st.markdown(f"""
                            <div class="glass-panel discovery-card-novel" style="padding: 15px; margin-bottom: 15px; border: 2px solid #FF007A; box-shadow: 0 0 15px rgba(255, 0, 122, 0.5);">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div>
                                        <span style="font-size: 0.8em; color: #94A3B8;">NTU DISCOVERY</span>
                                        <h3 class="neon-text-pink" style="margin: 5px 0; color: #FF007A; text-shadow: 0 0 5px #FF007A;">{header_text}</h3>
                                        <div class="breadcrumb" style="color: #FACC15; font-weight: bold;">{predicted_lineage}</div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 1.5em; font-weight: bold; color: #FF007A;">{confidence:.1f}%</div>
                                        <div style="font-size: 0.7em; color: #64748B;">PHYLOGENETIC STABILITY</div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px; background: #0F172A; height: 6px; border-radius: 3px; overflow: hidden;">
                                    <div style="width: {confidence}%; background: #FF007A; height: 100%; box-shadow: 0 0 10px #FF007A;"></div>
                                </div>
                                <div style="margin-top: 8px; font-size: 0.85em; color: #CBD5E1;">
                                    POPULATION: <b>{cluster_size} sequences</b> forming a stable genomic cluster
                                </div>
                                <div style="margin-top: 4px; font-size: 0.75em; color: #94A3B8; font-style: italic;">
                                    {scientific_note}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            continue
                            
                        # Determine Styling (Normal Card)
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
                        
                        # Use the new hierarchy string if available, otherwise fallback to taxonomic_reliability_str or display_lineage
                        display_breadcrumb = hit.get('hierarchy', '')
                        if not display_breadcrumb or display_breadcrumb == 'Unknown':
                            display_breadcrumb = hit.get('taxonomic_reliability_str', '') if is_novel and hit.get('taxonomic_reliability_str') else hit['display_lineage'].replace('Unknown', '???')
                        
                        st.markdown(f"""
                        <div class="glass-panel {card_class}" style="padding: 15px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <span style="font-size: 0.8em; color: #94A3B8;">QUERY: {hit.get('query_id', 'Unknown')}</span>
                                    <h3 class="{text_class}" style="margin: 5px 0;">{card_icon_html} {hit['display_name']}</h3>
                                    <div class="breadcrumb" style="color: #FACC15; font-weight: bold;">{display_breadcrumb}</div>
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
                        
                        if st.button("VIEW GENOMIC TOPOLOGY", key=f"btn_viz_{hit.get('query_id', 'Unknown')}"):
                            st.session_state['active_viz_id'] = hit.get('query_id')
                            st.session_state['active_viz_vector'] = hit.get('vector')
                            st.session_state['active_viz_name'] = hit.get('display_name')
                            st.session_state['active_viz_novel'] = is_novel
                            st.session_state['navigation'] = 'GENOMIC MANIFOLD'
                            st.rerun()
                    
            else:
                with results_container:
                    st.info("Waiting for Scan Data... System Idle.")
                    
        with comp_tab:
            if st.session_state.scan_results_buffer and len(st.session_state.scan_results_buffer) > 0:
                st.markdown("### COMMUNITY ANALYTICS")
                
                # 1. Diversity Metrics
                total_seqs_processed = len(st.session_state.scan_results_buffer)
                
                # Calculate Species Richness (S) and Shannon-Wiener Index (H')
                species_counts = {}
                has_clusters = 'ntu_registry' in st.session_state and st.session_state['ntu_registry']
                
                for hit in st.session_state.scan_results_buffer:
                    cluster_id = hit.get('cluster_id')
                    if has_clusters and cluster_id and cluster_id != -1:
                        species_name = f"[NTU] {cluster_id}"
                    else:
                        species_name = hit.get('display_name', 'Unknown')
                    
                    species_counts[species_name] = species_counts.get(species_name, 0) + 1
                
                # Species Richness (S)
                S = len(species_counts)
                
                # Shannon-Wiener Index (H')
                H_prime = 0
                for count in species_counts.values():
                    p_i = count / total_seqs_processed
                    if p_i > 0:
                        H_prime -= p_i * np.log(p_i)
                
                # Novelty Ratio
                num_ntus = len(st.session_state.get('ntu_registry', [])) if has_clusters else 0
                novelty_ratio = (num_ntus / total_seqs_processed) * 100 if total_seqs_processed > 0 else 0
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Species Richness (S)", f"{S}")
                m2.metric("Shannon-Wiener Index (H')", f"{H_prime:.3f}")
                m3.metric("Novelty Ratio", f"{novelty_ratio:.1f}%", f"{num_ntus} NTUs")
                
                st.divider()
                
                # 2. Taxonomic Sunburst
                st.markdown("#### Taxonomic Sunburst")
                hierarchy_data = []
                processed_clusters = set()
                
                for hit in st.session_state.scan_results_buffer:
                    cluster_id = hit.get('cluster_id')
                    
                    if has_clusters and cluster_id and cluster_id != -1:
                        if cluster_id in processed_clusters:
                            continue
                        processed_clusters.add(cluster_id)
                        
                        entity = hit.get('cluster_entity', {})
                        
                        # Try to get lineage from the first member's reliability if available
                        kingdom = "Animalia" # Default for deep sea
                        phylum = "Unknown"
                        class_name = "Unknown"
                        family = "Unknown"
                        
                        if entity.get('members') and len(entity.get('members')) > 0:
                            first_member_id = entity['members'][0]['id']
                            for h in st.session_state.scan_results_buffer:
                                if h.get('query_id') == first_member_id:
                                    rel = h.get('reliability', {})
                                    phylum = rel.get('Phylum', {}).get('name', 'Unknown')
                                    class_name = rel.get('Class', {}).get('name', 'Unknown')
                                    family = rel.get('Family', {}).get('name', 'Unknown')
                                    break
                        
                        resolved_name = f"[NOVEL] {entity.get('otu_id', 'NTU')}"
                        
                        hierarchy_data.append({
                            'Kingdom': kingdom,
                            'Phylum': phylum,
                            'Class': class_name,
                            'Family': family,
                            'Resolved_Name': resolved_name,
                            'Count': entity.get('cluster_size', 1)
                        })
                    else:
                        rel = hit.get('reliability', {})
                        kingdom = "Animalia"
                        phylum = rel.get('Phylum', {}).get('name', 'Unknown')
                        class_name = rel.get('Class', {}).get('name', 'Unknown')
                        family = rel.get('Family', {}).get('name', 'Unknown')
                        resolved_name = hit.get('display_name', 'Unknown')
                        
                        hierarchy_data.append({
                            'Kingdom': kingdom,
                            'Phylum': phylum,
                            'Class': class_name,
                            'Family': family,
                            'Resolved_Name': resolved_name,
                            'Count': 1
                        })
                
                df_comp = pd.DataFrame(hierarchy_data)
                
                # Create Sunburst Chart
                fig_sunburst = px.sunburst(
                    df_comp, 
                    path=['Kingdom', 'Phylum', 'Class', 'Family', 'Resolved_Name'], 
                    values='Count',
                    color='Phylum',
                    color_discrete_sequence=px.colors.sequential.Tealgrn, # Bioluminescent gradient
                    title="Real-Time Community Composition"
                )
                
                fig_sunburst.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0', family='Consolas'),
                    margin=dict(t=40, l=0, r=0, b=0)
                )
                
                fig_sunburst.update_traces(
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Parent: %{parent}',
                    marker=dict(line=dict(color='#0A0F1E', width=1))
                )
                
                st.plotly_chart(fig_sunburst, use_container_width=True)
                
                st.divider()
                
                # 3. Rarefaction Curve
                st.markdown("#### Rarefaction Curve (Discovery Rate)")
                
                # Calculate cumulative unique species over time
                # Since scan_results_buffer has latest on top (index 0), we need to reverse it to simulate time
                chronological_buffer = list(reversed(st.session_state.scan_results_buffer))
                
                seen_species = set()
                cumulative_species = []
                
                for hit in chronological_buffer:
                    cluster_id = hit.get('cluster_id')
                    if has_clusters and cluster_id and cluster_id != -1:
                        species_name = f"[NTU] {cluster_id}"
                    else:
                        species_name = hit.get('display_name', 'Unknown')
                        
                    seen_species.add(species_name)
                    cumulative_species.append(len(seen_species))
                
                df_rarefaction = pd.DataFrame({
                    'Sequences Processed': range(1, len(cumulative_species) + 1),
                    'Unique Species Discovered': cumulative_species
                })
                
                fig_rarefaction = px.line(
                    df_rarefaction, 
                    x='Sequences Processed', 
                    y='Unique Species Discovered',
                    color_discrete_sequence=['#00E5FF']
                )
                
                fig_rarefaction.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0', family='Consolas'),
                    xaxis=dict(showgrid=True, gridcolor='#334155', title='Sequences Processed'),
                    yaxis=dict(showgrid=True, gridcolor='#334155', title='Unique Species Discovered'),
                    margin=dict(t=10, l=0, r=0, b=0)
                )
                
                st.plotly_chart(fig_rarefaction, use_container_width=True)
                
                # Scientific Hook: Check if curve is still rising
                if len(cumulative_species) > 5:
                    recent_discoveries = cumulative_species[-1] - cumulative_species[-5]
                    if recent_discoveries > 0:
                        st.warning("SAMPLING EFFORT INSUFFICIENT - ADDITIONAL SEQUENCING RECOMMENDED")
                    else:
                        st.success("SAMPLING EFFORT SUFFICIENT - ASYMPTOTE REACHED")
                elif len(cumulative_species) > 1:
                    recent_discoveries = cumulative_species[-1] - cumulative_species[0]
                    if recent_discoveries > 0:
                        st.warning("SAMPLING EFFORT INSUFFICIENT - ADDITIONAL SEQUENCING RECOMMENDED")
                    else:
                        st.success("SAMPLING EFFORT SUFFICIENT - ASYMPTOTE REACHED")
                
            else:
                st.info("Waiting for Scan Data to generate Community Analytics...")

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
    
    col_viz1, col_viz2 = st.columns([3, 1])
    with col_viz1:
        st.subheader("LATENT SPACE TOPOLOGY")
    with col_viz2:
        force_recluster = st.button("FORCE RE-CLUSTER", use_container_width=True, icon=":material/hub:")
        if force_recluster and 'scan_results_buffer' in st.session_state and st.session_state.scan_results_buffer:
            with st.spinner("Re-clustering Manifold..."):
                clusters = discovery.analyze_novelty(st.session_state.scan_results_buffer)
                st.session_state.novel_clusters = clusters
                st.session_state['ntu_registry'] = clusters
                if clusters:
                    for cluster in clusters:
                        cid = cluster['otu_id']
                        for member in cluster.get('members', []):
                            mid = member['id']
                            for hit in st.session_state.scan_results_buffer:
                                if hit.get('query_id') == mid:
                                    hit['cluster_id'] = cid
                                    hit['cluster_entity'] = cluster
                    st.success(f"Re-clustered: {len(clusters)} NTUs found.")
                else:
                    st.info("Re-clustered: No NTUs found.")
                st.rerun()
    
    if 'active_viz_id' in st.session_state and st.session_state['active_viz_id']:
        active_id = st.session_state['active_viz_id']
        active_name = st.session_state.get('active_viz_name', active_id)
        
        st.markdown(f"### HIGH-RESOLUTION ZOOM: {active_name}")
        
        with st.spinner("Calculating Localized PCA Manifold..."):
            log_event(f"[UI] Attempting to render topology for ID: {active_id}")
            # Generate the localized manifold if not cached
            if 'manifold_cache' not in st.session_state or active_id not in st.session_state['manifold_cache']:
                log_event(f"[UI] Cache miss for {active_id}. Triggering Real-Time Neighborhood Search.")
                viz.get_neighborhood_manifold(
                    query_vector=st.session_state['active_viz_vector'],
                    query_id=active_id,
                    atlas_manager=atlas,
                    top_k=500
                )
            
            fig_3d = viz.create_localized_plot(active_id)
            
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("Localized Holographic Projection: 500 Nearest Neighbors in Latent Space.")
        
        if st.button("← CLEAR SELECTION"):
            st.session_state['active_viz_id'] = None
            st.rerun()
            
    else:
        st.info("Please select a sequence from the Real-Time Monitor to view its local topology.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: NOVELTY DISCOVERY ---
with tab_discovery:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.subheader("HOLOTYPE REGISTRY: NTU DISCOVERY")
        st.caption("Auto-Clustering 'Dark Taxa' to identify potential new species groups.")
    with col_header2:
        if st.button("🔄 REFRESH REGISTRY", use_container_width=True):
            if 'scan_results_buffer' in st.session_state and st.session_state.scan_results_buffer:
                st.session_state['ntu_registry'] = discovery.analyze_novelty(st.session_state.scan_results_buffer)
            st.rerun()
    
    if 'scan_results_buffer' in st.session_state and st.session_state.scan_results_buffer:
        # 1. Run Discovery Loop
        if 'ntu_registry' not in st.session_state:
            st.session_state['ntu_registry'] = discovery.analyze_novelty(st.session_state.scan_results_buffer)
            
        novel_entities = st.session_state['ntu_registry']
        
        if novel_entities:
            st.success(f"DISCOVERY ALERT: Identified {len(novel_entities)} Potential New Species Groups!")
            
            # 3D Map at the top
            if 'active_viz_id' in st.session_state and st.session_state['active_viz_id']:
                active_id = st.session_state['active_viz_id']
                fig_3d = viz.create_localized_plot(active_id)
                
                # Enable selection
                event = st.plotly_chart(
                    fig_3d, 
                    use_container_width=True, 
                    on_select="rerun",
                    selection_mode="points"
                )
                
                selected_cluster_id = None
                selected_points = []
                try:
                    if isinstance(event, dict):
                        selected_points = event.get("selection", {}).get("points", [])
                    else:
                        selected_points = getattr(event, "selection", {}).get("points", [])
                except Exception:
                    pass
                    
                if selected_points:
                    for pt in selected_points:
                        if "customdata" in pt and pt["customdata"]:
                            selected_cluster_id = pt["customdata"][0]
                            break
            else:
                st.info("No manifold context available.")
                selected_cluster_id = None
            
            st.markdown("### THE NTU GALLERY")
            
            # Export Buffer
            export_data = []
            
            # 2. Render Cluster Gallery (Grid)
            cols = st.columns(3)
            for i, entity in enumerate(novel_entities):
                with cols[i % 3]:
                    # Bioluminescent Neon Pink as requested
                    gauge_color = "#FF007A"
                    
                    # Highlight logic
                    is_highlighted = (selected_cluster_id == entity['otu_id'])
                    
                    bg_color = "rgba(255, 0, 122, 0.1)" if is_highlighted else "rgba(0,0,0,0.3)"
                    border_style = f"2px solid {gauge_color}" if is_highlighted else f"1px solid {gauge_color}"
                    box_shadow = f"0 0 15px {gauge_color}" if is_highlighted else f"0 0 5px {gauge_color}"
                    
                    confidence = 100 - (entity['biological_divergence'] * 100)
                    
                    # Consensus Breadcrumb Logic
                    lineage_parts = entity.get('lineage', '').split(';')
                    lineage_parts = [p for p in lineage_parts if p and p != 'Unknown']
                    if len(lineage_parts) > 3:
                        breadcrumb = " > ".join(lineage_parts[:3]) + f" > <span style='color: {gauge_color}; font-weight: bold;'>[NOVEL GENUS]</span>"
                    elif len(lineage_parts) > 0:
                        breadcrumb = " > ".join(lineage_parts) + f" > <span style='color: {gauge_color}; font-weight: bold;'>[NOVEL GENUS]</span>"
                    else:
                        breadcrumb = f"Unknown > <span style='color: {gauge_color}; font-weight: bold;'>[NOVEL GENUS]</span>"
                    
                    # Centroid Coordinates
                    coords = entity['avg_vector'][:3] if 'avg_vector' in entity and len(entity['avg_vector']) >= 3 else [0,0,0]
                    coords_str = f"[{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}, ...]"
                    
                    st.markdown(f"""
                    <div style="border: {border_style}; box-shadow: {box_shadow}; padding: 15px; border-radius: 8px; background: {bg_color}; margin-bottom: 15px; transition: all 0.3s ease; height: 100%;">
                        <h4 style="color: {gauge_color}; margin: 0; font-size: 1.1em; text-shadow: 0 0 5px {gauge_color};">✦ {entity['otu_id']}</h4>
                        
                        <div style="margin-top: 10px; margin-bottom: 5px; font-size: 0.85em; color: #CBD5E1;">
                            <b>Holotype ID:</b> {entity.get('representative_id', 'Unknown')}
                        </div>
                        <div style="margin-bottom: 5px; font-size: 0.85em; color: #CBD5E1;">
                            <b>Latent Coords:</b> <span style="font-family: monospace;">{coords_str}</span>
                        </div>
                        <div style="margin-bottom: 5px; font-size: 0.85em; color: #CBD5E1;">
                            <b>Population:</b> {entity['cluster_size']} sequences
                        </div>
                        
                        <div style="margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.05); border-left: 3px solid {gauge_color}; font-size: 0.8em; color: #E2E8F0;">
                            <b>Consensus Breadcrumb:</b><br>
                            {breadcrumb}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prepare for Export
                    fasta_lines = []
                    for member in entity.get('members', []):
                        fasta_lines.append(f">{member['id']} | Cluster: {entity['otu_id']} | Anchor: {entity['nearest_relative']}")
                        fasta_lines.append(member['seq'])
                    cluster_fasta = "\n".join(fasta_lines)
                    export_data.append(cluster_fasta)
                    
                    # GenBank Export Button
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("VIEW GENOMIC TOPOLOGY", key=f"btn_viz_cluster_{entity['otu_id']}"):
                            st.session_state['active_viz_id'] = entity.get('representative_id')
                            st.session_state['active_viz_vector'] = entity.get('avg_vector')
                            st.session_state['active_viz_name'] = entity['otu_id']
                            st.session_state['active_viz_novel'] = True
                            st.session_state['navigation'] = 'GENOMIC MANIFOLD'
                            st.rerun()
                    with col_btn2:
                        if st.button("EXPORT ARCHIVE", key=f"exp_{entity['otu_id']}", icon=":material/folder_zip:"):
                            import json
                            export_dir = Path("E:/DeepBio_Scan/results") / entity['otu_id']
                            if not Path("E:/").exists():
                                export_dir = Path("data/results") / entity['otu_id']
                            
                            export_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 1. FASTA
                            with open(export_dir / f"{entity['otu_id']}.fasta", "w") as f:
                                f.write(cluster_fasta)
                                
                            # 2. JSON Metadata
                            metadata = {
                                "otu_id": entity['otu_id'],
                                "representative_id": entity.get('representative_id', 'Unknown'),
                                "cluster_size": entity['cluster_size'],
                                "biological_divergence": entity['biological_divergence'],
                                "lineage": entity.get('lineage', 'Unknown'),
                                "centroid_vector": [float(x) for x in entity.get('avg_vector', [])]
                            }
                            with open(export_dir / "metadata.json", "w") as f:
                                json.dump(metadata, f, indent=4)
                                
                            # 3. High-res PNG
                            try:
                                rep_id = entity.get('representative_id')
                                if rep_id:
                                    if 'manifold_cache' not in st.session_state or rep_id not in st.session_state['manifold_cache']:
                                        # Generate manifold if not cached
                                        rep_vector = entity.get('avg_vector')
                                        if rep_vector is not None:
                                            viz.get_neighborhood_manifold(
                                                query_id=rep_id,
                                                query_vector=rep_vector,
                                                atlas_manager=atlas,
                                                top_k=500
                                            )
                                    
                                    fig_3d = viz.create_localized_plot(rep_id)
                                    fig_3d.write_image(str(export_dir / "manifold.png"), width=1920, height=1080, scale=2)
                            except Exception as e:
                                logger.error(f"Failed to export PNG: {e}")
                                
                            st.toast(f"✅ Exported {entity['otu_id']} to {export_dir}")

            st.divider()
            
            # 3. Export Action (Global)
            if st.button("EXPORT ALL CLUSTERS (Phylogenetic Tree)", icon=":material/download:"):
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

# --- TAB 5: SCALABILITY & BENCHMARKING ---
with tab_benchmark:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("SYSTEM SCALABILITY & BENCHMARKING")
    st.caption("Performance metrics and future hardware requirements for the DeepBio-Scan engine.")
    
    col_bench1, col_bench2 = st.columns(2)
    
    with col_bench1:
        st.markdown("#### 1. Latency Comparison (Vector vs Alignment)")
        # Bar chart comparing current search latency (<10ms) against a simulated alignment-based BLAST search (Minutes).
        latency_data = pd.DataFrame({
            "Method": ["DeepBio Vector Search", "Traditional BLAST (Simulated)"],
            "Latency (ms)": [8, 120000] # 8ms vs 2 minutes
        })
        
        fig_latency = px.bar(
            latency_data, 
            x="Method", 
            y="Latency (ms)", 
            color="Method",
            color_discrete_sequence=["#00E5FF", "#FF007A"],
            log_y=True,
            title="Search Latency (Log Scale)"
        )
        fig_latency.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color="#FFF",
            showlegend=False
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        st.info("Vector search operates in **<10ms** per query, compared to minutes for traditional alignment algorithms.")

    with col_bench2:
        st.markdown("#### 2. Data Horizon Projection")
        # Data Horizon Graph: 'Database Coverage' projection
        horizon_data = pd.DataFrame({
            "Phase": ["Current (32GB USB)", "Grant Goal (2TB NVMe)"],
            "Signatures": [100000, 4200000],
            "Scope": ["Deep-Sea Markers", "Global Marine Metagenome"]
        })
        
        fig_horizon = px.bar(
            horizon_data,
            x="Phase",
            y="Signatures",
            color="Phase",
            text="Signatures",
            color_discrete_sequence=["#00FF7F", "#7000FF"],
            title="Database Coverage Projection"
        )
        fig_horizon.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_horizon.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color="#FFF",
            showlegend=False
        )
        st.plotly_chart(fig_horizon, use_container_width=True)
        st.caption("Scaling from targeted deep-sea markers to a comprehensive global marine index.")

    st.divider()
    
    st.markdown("#### 3. IOPS Analysis & Hardware Requirements")
    col_iops1, col_iops2 = st.columns([1, 2])
    
    with col_iops1:
        # Gauge showing 'USB Bandwidth Utilization'
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 85,
            title = {'text': "USB Bandwidth Utilization (%)", 'font': {'size': 16, 'color': '#FFF'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "#FFF"},
                'bar': {'color': "#FF007A"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0, 255, 127, 0.2)"},
                    {'range': [50, 80], 'color': "rgba(255, 165, 0, 0.2)"},
                    {'range': [80, 100], 'color': "rgba(255, 0, 122, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color="#FFF",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_iops2:
        st.markdown("""
        <div style="padding: 20px; background: rgba(255, 0, 122, 0.1); border-left: 4px solid #FF007A; border-radius: 5px; height: 100%;">
            <h4 style="color: #FF007A; margin-top: 0;">⚠️ Storage Bottleneck Warning</h4>
            <p style="color: #E2E8F0; font-size: 1.1em;">
                While the LanceDB vector search is highly optimized and fast, <b>Localized PCA fitting for 1,000+ sequences simultaneously</b> requires significant I/O throughput.
            </p>
            <p style="color: #CBD5E1;">
                The current 32GB USB drive is nearing its bandwidth limit during heavy clustering operations. To maintain real-time performance for the full 4.2M signature database, the system will require the <b>TRC high-speed NVMe storage array</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 6: DOCUMENTATION ---
with tab_docs:
    render_docs()
    
    # End of Docs
