# NeuralJudge: Interpretable LLM Evaluation with Distillation

**NeuralJudge** is a local, tunable evaluation framework for LLM outputs that enables **model distillation** through interpretable constraint weights. It replaces subjective "vibe checks" with a deterministic, weighted scoring system based on the **Logic Score Method**.

## The Theory: Logic Score Method

This project implements the evaluation framework proposed by me in **[A Proposed Logic Score Method for LLM Inferencing](https://shwetabhsingh.substack.com/p/a-proposed-logic-score-method-for)**.

Instead of asking an LLM for a generic score (which is prone to hallucination and inconsistency), NeuralJudge breaks evaluation into two vectors:

1.  **Constraint Vector ($W$)**: A set of specific rules (e.g., "Factuality", "Tone") with assigned weights.
2.  **Label Vector ($L$)**: Qualitative judgments from a small "Judge" LLM (e.g., "High", "Mid", "Low").

The final score is calculated using the **Hadamard Product** of the weights and the numerical values of the labels:

$$ Score = \sum_{i=1}^{n} (Weight_i \times LabelScore_i) $$

We use **Fuzzy Matching** to map the Judge LLM's natural language output to the strict scoring rubric, ensuring robustness against minor variations in the model's response.

## Model Distillation Architecture

NeuralJudge implements a form of **interpretable distillation**:

**Teacher Model**: Use a powerful model (GPT-4, Claude, etc.) to provide ground truth scores for your evaluation scenarios.

**Student Model**: A small, efficient model (Phi-3, Gemma-2B) provides simple label classifications ("High", "Mid", "Low") for each constraint.

**Knowledge Transfer**: The "intelligence" is compressed into the **constraint weight vector**, which is trained via backpropagation to match the Teacher's evaluation strategy.

**Key Advantage**: Unlike traditional distillation, the learned weights remain **human-interpretable** and **modifiable**. You can inspect why a score is what it is, and manually override weights if needed.

**Cost Efficiency**: After training, you only need the small Student model for inference, reducing evaluation costs by up to 90% while maintaining alignment with the Teacher's preferences.

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
4.  **Evaluate**: Click "Run Evaluation" to see the Radar Chart and Score Breakdown.
5.  **Tune**: Switch to the "Tuning" tab to train weights using ground truth scores.

## Neural Tuning (Backpropagation)

NeuralJudge models the evaluation process as a single **linear neuron** where inputs are the Judge's scores and weights are the constraint importance.

$$ Score = \sum (w_i \cdot x_i) $$

**Design Choice**: We use a "White-Box" linear layer instead of a black-box neural network to maintain interpretability. The weights directly correspond to human-understandable constraints (e.g., Factuality).

**Training Mechanism**:
1.  **Forward Pass**: The system calculates a predicted score based on current weights.
2.  **Loss Calculation**: You provide a "Ground Truth" score. The system calculates the error (Delta).
3.  **Backpropagation**: Weights are updated using Gradient Descent.

$$ w_i^{new} = w_i + \alpha \cdot (Score_{target} - Score_{predicted}) \cdot x_i $$

This ensures proper **credit assignment**: only constraints that contributed to the score (active inputs) receive weight updates.

## Tech Stack
-   **Engine**: `llama-cpp-python` (Local Inference)
-   **Logic**: `numpy` & `fuzzywuzzy` (Scoring & Robustness)
-   **UI**: `Streamlit` & `Plotly` (Dashboard & Visualization)
