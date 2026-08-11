import streamlit as st
import time
import os
import sys

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_triage_langgraph import get_agents, run_case
import settings

st.set_page_config(
    page_title="Flipkart Grid 8.0 - Return Triage",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Headers */
    h1, h2, h3 {
        color: #F8F9FA;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Verdict Cards */
    .verdict-approve {
        background: linear-gradient(135deg, #00C853 0%, #009624 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .verdict-reject {
        background: linear-gradient(135deg, #FF3D00 0%, #DD2C00 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 61, 0, 0.3);
    }
    .verdict-escalate {
        background: linear-gradient(135deg, #FFC107 0%, #FF8F00 100%);
        padding: 20px;
        border-radius: 12px;
        color: #1A1A1A;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
    }
    
    /* Reasoning Box */
    .reasoning-box {
        background-color: #1E2329;
        border-left: 5px solid #2962FF;
        padding: 20px;
        border-radius: 8px;
        color: #E2E8F0;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Stats/Metrics */
    .metric-card {
        background-color: #161B22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .metric-label {
        color: #8B949E;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #58A6FF;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ─── INIT AGENTS ───
@st.cache_resource
def load_system(_dummy=1):
    # Load all models and indexes only once
    with st.spinner("Initializing AI Agents and loading Machine Learning models..."):
        get_agents()
    return True

system_ready = load_system()

# ─── SIDEBAR ───
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### 🤖 Multi-Agent Triage")
    st.markdown("An intelligent system powered by LangGraph, LightGBM, and Gemini to automate return request adjudications.")
    st.markdown("---")
    st.markdown("#### Agent Pipeline:")
    st.markdown("✅ **Data Agent** (PostgreSQL/Feature Store)")
    st.markdown("✅ **Risk Agent** (LightGBM)")
    st.markdown("✅ **RAG Agent** (Company Policies)")
    st.markdown("✅ **Orchestrator** (Gemini 3.5 Flash)")
    
# ─── MAIN APP ───
st.title("📦 Return Fraud Triage Dashboard")
st.markdown("Process e-commerce return requests autonomously bridging deterministic rules and human-like reasoning.")

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Single Query'

mode_col1, mode_col2 = st.columns(2)
with mode_col1:
    if st.button("Single Query", use_container_width=True, type="primary" if st.session_state.app_mode == 'Single Query' else "secondary"):
        st.session_state.app_mode = 'Single Query'
with mode_col2:
    if st.button("Batch Processing", use_container_width=True, type="primary" if st.session_state.app_mode == 'Batch Processing' else "secondary"):
        st.session_state.app_mode = 'Batch Processing'

analyze_btn = False
return_id_input = ""

if st.session_state.app_mode == 'Single Query':
    col1, col2 = st.columns([3, 1])
    with col1:
        return_id_input = st.text_input("Enter Return ID (e.g., RET_09900061)", value="RET_09900061")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Request", use_container_width=True, type="primary")

elif st.session_state.app_mode == 'Batch Processing':
    st.markdown("### 🗂️ Batch Processing")
    num_queries = st.number_input("Batch Size (Number of queries to run)", min_value=1, max_value=100, value=5)
    batch_btn = st.button("▶️ Run Batch Processing", type="primary")
    
    if batch_btn:
        import pandas as pd
        import os
        from sqlalchemy import create_engine
        import json
        import settings
        
        # Check already processed returns
        processed_ids = set()
        log_file_path = os.path.join(settings.OUTPUT_DIR, 'batch_justification_log.jsonl')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'return_id' in data:
                                processed_ids.add(data['return_id'])
                        except:
                            pass
        
        engine = create_engine(settings.DB_URI)
        query = "SELECT return_id FROM returns ORDER BY return_request_date DESC, return_id"
        df = pd.read_sql(query, engine)
        
        unprocessed_df = df[~df['return_id'].isin(processed_ids)]
        batch_df = unprocessed_df.head(num_queries)
        
        if batch_df.empty:
            st.warning("No unprocessed return requests found in the database. All caught up!")
        else:
            st.info(f"Found {len(batch_df)} unprocessed cases. Starting batch run...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, row in batch_df.iterrows():
                rid = row['return_id']
                status_text.markdown(f"**Processing {rid} ({len(results)+1}/{len(batch_df)})...**")
                
                img_path = None
                custom_text = None
                try:
                    probe_q = "SELECT image_path, customer_custom_text FROM adversarial_probe_set WHERE assigned_return_id = %s"
                    match = pd.read_sql(probe_q, engine, params=(rid,))
                    if not match.empty:
                        mapped_img = match.iloc[0].get('image_path', '')
                        if pd.notna(mapped_img) and str(mapped_img).strip() != '':
                            img_path = str(mapped_img).strip()
                        mapped_text = match.iloc[0].get('customer_custom_text', '')
                        if pd.notna(mapped_text) and str(mapped_text).strip() != '':
                            custom_text = str(mapped_text).strip()
                except Exception:
                    pass
                
                try:
                    res = run_case(rid, image_path=img_path, reason_text=custom_text, log_filename='batch_justification_log.jsonl')
                    results.append(res)
                except Exception as e:
                    st.error(f"Error processing {rid}: {e}")
                
                progress_bar.progress((len(results)) / len(batch_df))
            
            status_text.success(f"✅ Batch Processing Complete! Processed {len(results)} cases.")
            
            # Summary Metrics
            approved = sum(1 for r in results if r['verdict'] == 'Auto-Approve')
            escalated = sum(1 for r in results if r['verdict'] == 'Escalate')
            rejected = sum(1 for r in results if r['verdict'] == 'Auto-Reject')
            
            st.markdown("### Batch Run Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Auto-Approved ✅", approved)
            c2.metric("Escalated ⚠️", escalated)
            c3.metric("Auto-Rejected 🚫", rejected)

if analyze_btn and return_id_input:
    # Check if there is an image mapped to this return_id
    import pandas as pd
    import os
    from sqlalchemy import create_engine
    import settings
    
    img_path = None
    custom_text = None
    try:
        engine = create_engine(settings.DB_URI)
        query = "SELECT image_path, customer_custom_text FROM adversarial_probe_set WHERE assigned_return_id = %s"
        match = pd.read_sql(query, engine, params=(return_id_input,))
        if not match.empty:
            mapped_img = match.iloc[0].get('image_path', '')
            if pd.notna(mapped_img) and str(mapped_img).strip() != '':
                img_path = str(mapped_img).strip()
            
            mapped_text = match.iloc[0].get('customer_custom_text', '')
            if pd.notna(mapped_text) and str(mapped_text).strip() != '':
                custom_text = str(mapped_text).strip()
    except Exception as e:
        pass

    if img_path and os.path.exists(img_path):
        st.info(f"📸 Image proof attached: {os.path.basename(img_path)}")
        st.image(img_path, width=200)
        
    if custom_text:
        st.info(f"💬 Customer Comment: \"{custom_text}\"")

    # ─── PROGRESS SIMULATION & EXECUTION ───
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("🔍 **Fetching case data and historical features...**")
    progress_bar.progress(20)
    time.sleep(0.5)
    
    status_text.markdown("🧮 **Running LightGBM Risk Scoring & Rule Engine...**")
    progress_bar.progress(40)
    
    status_text.markdown("🤖 **Invoking LangGraph Orchestrator...**")
    progress_bar.progress(60)
    
    # Run the actual backend
    start_time = time.time()
    try:
        result = run_case(return_id_input, image_path=img_path, reason_text=custom_text)
        
        status_text.markdown("📚 **MCP Tools Querying Company Policies...**")
        progress_bar.progress(80)
        
        elapsed = time.time() - start_time
        progress_bar.progress(100)
        status_text.success(f"✅ Triage complete in {elapsed:.2f} seconds!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        # ─── DISPLAY RESULTS ───
        st.markdown("---")
        
        # Verdict Banner
        verdict = result['verdict']
        if verdict == 'Auto-Approve':
            st.markdown("<div class='verdict-approve'><h2>✅ AUTO-APPROVE</h2></div>", unsafe_allow_html=True)
        elif verdict == 'Auto-Reject':
            st.markdown("<div class='verdict-reject'><h2>🚫 AUTO-REJECT</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict-escalate'><h2>⚠️ ESCALATE TO HUMAN</h2></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Core Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Latency</div><div class='metric-value'>{result['elapsed_seconds']}s</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total API Calls</div><div class='metric-value'>{result['num_llm_calls']}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Confidence</div><div class='metric-value'>{result['confidence']}</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Combined Risk Score</div><div class='metric-value'>{result['combined_score']:.3f}</div></div>", unsafe_allow_html=True)
        with m5:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>LGBM Fraud Prob</div><div class='metric-value'>{result['lgbm_fraud_prob']:.1%}</div></div>", unsafe_allow_html=True)
            
        st.markdown("### 🧠 Agent Reasoning Chain")
        st.markdown(f"<div class='reasoning-box'>{result['reasoning'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # SHAP and Rules Row
        st.markdown("### 🔍 Evidence & Explainability")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("#### Top SHAP Features (LightGBM)")
            for shap in result['top_shap']:
                direction_color = "🔴" if "increases" in shap['direction'] else "🟢"
                st.markdown(f"- {direction_color} **{shap['feature']}**: {shap['value']:.2f} *(Impact: {shap['shap_value']:+.3f})*")
                
        with e2:
            st.markdown("#### Triggered Rules")
            if result['rules_triggered']:
                for r in result['rules_triggered']:
                    st.warning(f"**{r['rule_name']}**: {r['description']}")
            else:
                st.success("No suspicious heuristic rules triggered.")
                
    except Exception as e:
        status_text.error(f"❌ Error processing return case: {str(e)}")
        progress_bar.empty()
