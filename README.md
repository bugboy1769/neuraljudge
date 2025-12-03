# Judge-0: Deterministic LLM Evaluation Engine

**Judge-0** is a local, configurable evaluation framework for LLM outputs (e.g., RAG systems). It replaces subjective "vibe checks" with a deterministic, weighted scoring system based on the **Logic Score Method**.

## The Theory: Logic Score Method

This project implements the evaluation framework proposed by me in **[A Proposed Logic Score Method for LLM Inferencing](https://shwetabhsingh.substack.com/p/a-proposed-logic-score-method-for)**.

Instead of asking an LLM for a generic score (which is prone to hallucination and inconsistency), Judge-0 breaks evaluation into two vectors:

1.  **Constraint Vector ($W$)**: A set of specific rules (e.g., "Factuality", "Tone") with assigned weights.
2.  **Label Vector ($L$)**: Qualitative judgments from a small, smart "Judge" LLM (e.g., "High", "Mid", "Low").

The final score is calculated using the **Hadamard Product** of the weights and the numerical values of the labels:

$$ Score = \sum_{i=1}^{n} (Weight_i \times LabelScore_i) $$

We use **Fuzzy Matching** to map the Judge LLM's natural language output to the strict scoring rubric, ensuring robustness against minor variations in the model's response.

## Quick Start

### 1. Prerequisites
-   **Python 3.10+**
-   **Apple Silicon Mac** (Recommended for Metal acceleration)
-   **GGUF Model**: Download a quantized model (e.g., [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)) to the `models/` directory.

### 2. Installation
```bash
pip install -r requirements.txt
```
*Note: For Apple Silicon acceleration, ensure `llama-cpp-python` is installed with `CMAKE_ARGS="-DLLAMA_METAL=on"`.*

### 3. Usage
Run the dashboard:
```bash
streamlit run app.py
```

1.  **Select Model**: Load your GGUF model from the sidebar.
2.  **Input**: Paste the Question and the LLM Answer you want to judge.
3.  **Configure**: Adjust the Constraints (Rules) and their Weights.
4.  **Judge**: Click "Run Evaluation" to see the Radar Chart and Score Breakdown.

## Tech Stack
-   **Engine**: `llama-cpp-python` (Local Inference)
-   **Logic**: `numpy` & `fuzzywuzzy` (Scoring & Robustness)
-   **UI**: `Streamlit` & `Plotly` (Dashboard & Visualization)
