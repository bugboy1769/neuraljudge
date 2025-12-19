"""
NeuralJudge CLI - Headless Training Pipeline

Usage:
    python main.py --dataset data/ml_qa.json --epochs 5 --lr 0.01
"""

import argparse
import json
import os
from typing import List, Dict, Any

from judge_0 import LogicScore, Constraint
from llm import LlamaWrapper
from teacher_llm import TeacherLLM
import config_manager as cm


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load Q/A dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def run_student_forward_pass(
    llm: LlamaWrapper,
    engine: LogicScore,
    question: str,
    answer: str,
    context: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the Student model to get labels for each constraint.
    Returns the scoring result with breakdown.
    """
    labels = []
    system_msg = config.get("system_message", "")
    label_map = config.get("label_map", {})
    
    for constraint in engine.constraints:
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
    
    return engine.score(labels)


def train_epoch(
    student_llm: LlamaWrapper,
    teacher_llm: TeacherLLM,
    dataset: List[Dict[str, Any]],
    config: Dict[str, Any],
    learning_rate: float
) -> float:
    """
    Run one training epoch over the dataset.
    Returns the average loss for the epoch.
    """
    constraints = [
        Constraint(c["name"], float(c["weight"]), c["description"])
        for c in config.get("constraints", [])
    ]
    engine = LogicScore(constraints, config.get("label_map", {}))
    
    total_loss = 0.0
    
    for i, sample in enumerate(dataset):
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        context = sample.get("context", "")
        
        # 1. Student forward pass
        result = run_student_forward_pass(
            student_llm, engine, question, answer, context, config
        )
        predicted_score = result["normalized_score"]
        
        # 2. Teacher ground truth
        ground_truth = teacher_llm.get_ground_truth_score(question, answer, context)
        
        # 3. Calculate loss
        loss = (ground_truth - predicted_score) ** 2
        total_loss += loss
        
        # 4. Backprop
        updated_constraints = engine.tune_weights(
            result["breakdown"],
            ground_truth,
            alpha=learning_rate
        )
        
        # Update engine constraints for next sample
        engine.constraints = updated_constraints
        
        print(f"  Sample {i+1}/{len(dataset)}: Pred={predicted_score:.1f}, Truth={ground_truth:.1f}, Loss={loss:.2f}")
    
    # Update config with new weights
    config["constraints"] = [
        {"name": c.name, "weight": c.weight, "description": c.description}
        for c in engine.constraints
    ]
    
    return total_loss / len(dataset) if dataset else 0.0


def main():
    parser = argparse.ArgumentParser(description="NeuralJudge Training CLI")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.5, help="Learning rate decay per epoch")
    parser.add_argument("--student-model", type=str, required=True, help="Path to Student GGUF model")
    parser.add_argument("--teacher-url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--teacher-model", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the teacher model on vLLM")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for all)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.abspath(args.config)
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load dataset
    dataset_path = os.path.abspath(args.dataset)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"  Loaded {len(dataset)} samples")
    
    # Initialize Student LLM
    print(f"Loading Student model: {args.student_model}...")
    student_llm = LlamaWrapper(args.student_model, n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers)
    
    # Initialize Teacher LLM
    print(f"Connecting to Teacher LLM at {args.teacher_url} (Model: {args.teacher_model})...")
    teacher_llm = TeacherLLM(base_url=args.teacher_url, model_name=args.teacher_model)
    if not teacher_llm.health_check():
        print("WARNING: Teacher LLM server not responding. Make sure vLLM is running.")
    
    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}")
    print("=" * 50)
    
    learning_rate = args.lr
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={learning_rate:.4f})")
        print("-" * 40)
        
        avg_loss = train_epoch(student_llm, teacher_llm, dataset, config, learning_rate)
        
        print(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
        
        # Decay learning rate
        learning_rate *= args.lr_decay
    
    # Save updated config
    print("\nSaving updated weights to config.json...")
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nTraining complete!")
    print("Final weights:")
    for c in config["constraints"]:
        print(f"  {c['name']}: {c['weight']:.4f}")


if __name__ == "__main__":
    main()
