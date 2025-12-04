# The Neural Judge: Theory & Math

## 1. Similar Work
The concept of a "Trainable Judge" aligns with recent research in **LLM Calibration** and **Reward Modeling**:
-   **LLM-Rubric (Ye et al., 2023)**: Uses a small neural network to "calibrate" LLM probability distributions against human preferences.
-   **Rubrics as Rewards (RaR)**: Uses structured rubrics to generate reward signals for RLHF.
-   **Judge-0 Approach**: Implements a **Linear Calibration Layer** on top of qualitative LLM outputs. This is a "White-Box" approach where the weights are interpretable, unlike a "Black-Box" neural net.

## 2. The Neuron Analogy
We can map the Judge-0 architecture directly to a single **Artificial Neuron**:

$$ y = f(\sum (w_i \cdot x_i) + b) $$

| Neural Component | Judge-0 Component | Explanation |
| :--- | :--- | :--- |
| **Inputs ($x_i$)** | **Judge Scores** | The numerical value of the label (e.g., "High" $\rightarrow$ 1.0, "Low" $\rightarrow$ 0.0). These are the incoming signals. |
| **Weights ($w_i$)** | **Constraint Weights** | The importance of each signal (e.g., Factuality=0.8). These are the trainable parameters. |
| **Bias ($b$)** | *None (Implicit 0)* | We assume a score of 0 if all inputs are 0. |
| **Activation ($f$)** | **Identity** | The linear sum is the final score. (The *Judge LLM* acts as the non-linear feature extractor *before* this layer). |
| **Output ($y$)** | **Final Score** | The weighted sum (0-100). |

## 3. Backpropagation: The Credit Assignment
**Question**: *"Won't every weight be updated by the same amount?"*
**Answer**: **No.** The update depends on the **Input ($x_i$)**.

### The Math
Let the Loss Function ($E$) be the Mean Squared Error between the Predicted Score ($y$) and the Ground Truth ($y_{true}$):
$$ E = \frac{1}{2} (y_{true} - y)^2 $$

We update weight $w_i$ using Gradient Descent:
$$ w_i^{new} = w_i - \alpha \frac{\partial E}{\partial w_i} $$

Using the Chain Rule:
$$ \frac{\partial E}{\partial w_i} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial w_i} $$

1.  **The Error Term** ($\frac{\partial E}{\partial y}$):
    $$ \frac{\partial E}{\partial y} = -(y_{true} - y) = -\delta $$
    *(This represents the global error magnitude)*

2.  **The Input Term** ($\frac{\partial y}{\partial w_i}$):
    Since $y = \sum w_k x_k$, the derivative with respect to $w_i$ is simply $x_i$.
    $$ \frac{\partial y}{\partial w_i} = x_i $$

### The Update Rule
$$ \Delta w_i = \alpha \cdot \delta \cdot x_i $$

### Physical Interpretation
The weight update is proportional to the **Input Score ($x_i$)**:
-   **Active Constraint ($x_i \approx 1.0$)**: If a constraint scored "High", it contributed significantly to the final score. It receives a **large update** proportional to the error.
-   **Inactive Constraint ($x_i \approx 0.0$)**: If a constraint scored "Low" (or "Non"), it did not contribute to the score. It receives **little to no update**.

**Example**:
-   Target: 90, Predicted: 50. Error $\delta = 40$.
-   **Factuality** scored "High" (1.0). Update $\propto 40 \times 1.0 = 40$. (Big boost).
-   **Tone** scored "Low" (0.0). Update $\propto 40 \times 0.0 = 0$. (No change).

This ensures that we only tune the weights of constraints that are actually active in the current scenario.
