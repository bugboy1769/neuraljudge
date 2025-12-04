import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json

# Import Modules
from judge_0 import LogicScore, Constraint
from llm import LlamaWrapper
import config_manager as cm

st.set_page_config(page_title="Judge-0", layout="wide")

# --- Load Configuration ---
config = cm.load_config()
if not config:
    st.error("Could not load config.json. Please ensure it exists.")
    st.stop()

# --- Sidebar: Model Config ---
st.sidebar.header("Configuration")
model_dir = "models"
available_models = [f for f in os.listdir(model_dir) if f.endswith(".gguf")] if os.path.exists(model_dir) else []

selected_model = st.sidebar.selectbox("Select Model", available_models)
n_ctx = st.sidebar.number_input("Context Window", value=2048, step=512)

# Load Model (Cached)
@st.cache_resource
def load_model(path, ctx):
    return LlamaWrapper(model_path=path, n_ctx=ctx)

llm = None
if selected_model:
    model_path = os.path.join(model_dir, selected_model)
    try:
        with st.spinner(f"Loading {selected_model}..."):
            llm = load_model(model_path, n_ctx)
        st.sidebar.success("Model Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# --- Main Content ---
st.title("Judge-0: Neural Evaluation Engine")

# 1. The Scenario (Fixed Anchor)
st.subheader("1. The Scenario")
col_s1, col_s2 = st.columns([1, 1])
with col_s1:
    question = st.text_area("Question (Anchor)", value=config.get("default_query", ""), height=100)
with col_s2:
    context = st.text_area("Context (Source Truth)", value=config.get("default_context", ""), height=100)

# 2. The Candidate (Variable)
st.subheader("2. The Candidate")
answer = st.text_area("LLM Answer", height=150, placeholder="Paste the response to evaluate here...")

# 3. Constraints & Config (Collapsible)
with st.expander("3. Constraints & System Configuration", expanded=False):
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.markdown("**System Message**")
        system_msg = st.text_area("System Prompt", value=config.get("system_message", ""), height=100)
        
        st.markdown("**Label Map**")
        label_map_str = st.text_area("Label Map (JSON)", json.dumps(config.get("label_map", {}), indent=2))
        try:
            label_map = json.loads(label_map_str)
        except:
            st.error("Invalid JSON in Label Map")
            label_map = {}

    with col_c2:
        st.markdown("**Constraints**")
        # Load from config if session state is empty
        if "constraints" not in st.session_state:
            st.session_state.constraints = config.get("constraints", [])
        
        edited_df = st.data_editor(
            pd.DataFrame(st.session_state.constraints),
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.constraints = edited_df.to_dict("records")

    # Save Config Button
    if st.button("Save Configuration to JSON"):
        new_config = {
            "system_message": system_msg,
            "default_query": question,
            "default_context": context,
            "constraints": st.session_state.constraints,
            "label_map": label_map
        }
        cm.save_config(new_config)
        st.success("Configuration saved to config.json!")

# --- Tabs for Modes ---
tab_judge, tab_tune = st.tabs(["Judge", "Tuning"])

# Shared Logic for Running the Judge
def run_judge_logic():
    constraints = [Constraint(r["name"], float(r["weight"]), r["description"]) for r in st.session_state.constraints]
    engine = LogicScore(constraints, label_map)
    
    labels = []
    progress_text = "Judging..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, constraint in enumerate(constraints):
        context_block = f"\nContext: {context}" if context else ""
        prompt = f"""{system_msg}

Question: {question}{context_block}
Answer: {answer}

Constraint: {constraint.name}
Description: {constraint.description}
Rubric: {json.dumps(label_map)}

Task: Rate the Answer based on the Constraint. Reply ONLY with one of the Rubric labels.
Verdict:"""
        verdict = llm.get_verdict(prompt)
        labels.append(verdict)
        my_bar.progress((i + 1) / len(constraints), text=f"Judging {constraint.name}...")
        
    my_bar.empty()
    return engine, engine.score(labels)

# --- Tab 1: Standard Judge ---
with tab_judge:
    if st.button("Run Evaluation", type="primary", disabled=not llm, key="btn_judge"):
        if not question or not answer:
            st.warning("Please provide Question and Answer.")
        else:
            engine, result = run_judge_logic()
            
            st.divider()
            st.header(f"Final Score: {result['normalized_score']}/100")
            
            # Radar & Table (Same as before)
            categories = [x['constraint'] for x in result['breakdown']]
            raw_scores = [x['score'] for x in result['breakdown']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=raw_scores, theta=categories, fill='toself', name='Score'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
            st.plotly_chart(fig)
            st.table(pd.DataFrame(result['breakdown']))

# --- Tab 2: Neural Tuning ---
with tab_tune:
    st.markdown("### Train the Judge")
    st.info("Provide a 'Ground Truth' score to teach the system how to weight these constraints.")
    
    if st.button("Run Forward Pass", key="btn_tune_run", disabled=not llm):
        if not question or not answer:
            st.warning("Please provide Question and Answer.")
        else:
            engine, result = run_judge_logic()
            st.session_state['last_run'] = result
            st.session_state['last_engine'] = engine
            st.success(f"Predicted Score: {result['normalized_score']}")

    if 'last_run' in st.session_state:
        st.divider()
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Predicted Score", st.session_state['last_run']['normalized_score'])
        with col_t2:
            target_score = st.number_input("Target Score (Ground Truth)", min_value=0.0, max_value=100.0, value=90.0)
        
        learning_rate = st.slider("Learning Rate (Alpha)", 0.001, 0.1, 0.01, format="%.3f")
        
        if st.button("Optimize Weights (Backprop)", type="primary"):
            engine = st.session_state['last_engine']
            breakdown = st.session_state['last_run']['breakdown']
            
            # Run Backprop
            new_constraints = engine.tune_weights(breakdown, target_score, alpha=learning_rate)
            
            # Update Session State & Config
            st.session_state.constraints = [{"name": c.name, "weight": c.weight, "description": c.description} for c in new_constraints]
            
            # Auto-save to config
            new_config = {
                "system_message": system_msg,
                "default_query": question,
                "default_context": context,
                "constraints": st.session_state.constraints,
                "label_map": label_map
            }
            cm.save_config(new_config)
            
            st.success("Weights Updated & Saved!")
            st.dataframe(pd.DataFrame(st.session_state.constraints))
            st.rerun()