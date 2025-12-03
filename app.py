import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

#Import Modules
from judge_0 import LogicScore, Constraint
from llm import LlamaWrapper

st.set_page_config(page_title="Judge-0", layout="wide")

# --- Sidebar: Model Config ---
st.sidebar.header("Configuration")
model_dir="models"
available_models=[f for f in os.listdir(model_dir) if f.endswith(".gguf")] if os.path.exists(model_dir) else []

selected_model=st.sidebar.selectbox("Select Model", available_models)
n_ctx=st.sidebar.number_input("Context Window", value=2048, step=512)

#Load Model (Cached)
@st.cache_resource
def load_model(path, ctx):
    return LlamaWrapper(model_path=path, n_ctx=ctx)

llm=None
if selected_model:
    model_path=os.path.join(model_dir, selected_model)
    try:
        with st.spinner(f"Loading {selected_model}..."):
            llm=load_model(model_path, n_ctx)
        st.sidebar.success("Model Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# --- Main Content ---
st.title("Judge-0: LLM Eval Engine")

col1, col2=st.columns([1,1])

with col1:
    st.subheader("1. Input")
    question=st.text_area("Question", height=100, placeholder="User Query ...")
    answer=st.text_area("Answer", height=100, placeholder="LLM Answer ...")

with col2:
    st.subheader("2. Constraints (The Rules)")

    #Default Constraints
    if "constraints" not in st.session_state:
        st.session_state.constraints=[
            {"name": "Factuality", "weight": 0.5, "desc":"Is the answer factually correct?"},
            {"name": "Tone", "weight": 0.2, "desc": "Is the tone professional?"},
            {"name": "Conciseness", "weight": 0.3, "desc": "Is the answer brief and to the point?"}
        ]
    
    #Editor
    edited_df=st.data_editor(
        pd.DataFrame(st.session_state.constraints),
        num_rows="dynamic",
        use_container_width=True
    )
    st.session_state.constraints=edited_df.to_dict("records")

    st.subheader("3. Label Map")
    #Simple dictionary editor
    label_map_str=st.text_area("Label Map (JSON Style)", '{"High": 1.0, "Mid": 0.5, "Low": 0.0, "Non": 0.0}')
    try:
        import json
        label_map=json.loads(label_map_str)
    except:
        st.error("Invalid JSON in Label Map")
        label_map={}

# --- Execution ---
if st.button("Run Evaluation", type="primary", disabled=not llm):
    if not question or not answer:
        st.warning("Please provide both the question and answer")
    else:
        #1. Setup Engine
        constraints = [Constraint(r["name"], float(r["weight"]), r["desc"]) for r in st.session_state.constraints]
        engine = LogicScore(constraints, label_map)
        
        # 2. Run Judge for each constraint
        labels = []
        progress_bar = st.progress(0)
        
        for i, constraint in enumerate(constraints):
            # Construct Prompt
            prompt = f"""User: You are an impartial judge.
Question: {question}
Answer: {answer}

Constraint: {constraint.name}
Description: {constraint.description}
Rubric: {label_map_str}

Task: Rate the Answer based on the Constraint. Reply ONLY with one of the Rubric labels.
Verdict:"""
            
            # Call LLM
            verdict = llm.get_verdict(prompt)
            labels.append(verdict)
            progress_bar.progress((i + 1) / len(constraints))
            
        # 3. Calculate Score
        result = engine.score(labels)
        
        # 4. Display Results
        st.divider()
        st.header(f"Final Score: {result['normalized_score']}/100")
        
        # Radar Chart
        categories = [x['constraint'] for x in result['breakdown']]
        scores = [x['weighted_score'] for x in result['breakdown']] # Or raw scores? Let's use raw scores for radar
        raw_scores = [x['score'] for x in result['breakdown']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=raw_scores,
            theta=categories,
            fill='toself',
            name='Score'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig)
        
        # Detailed Table
        st.table(pd.DataFrame(result['breakdown']))
        
