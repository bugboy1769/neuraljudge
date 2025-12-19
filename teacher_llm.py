"""
NeuralJudge Teacher LLM Wrapper

Uses vLLM for high-parameter model inference to generate ground truth scores.
"""

import requests
from typing import Optional

class TeacherLLM:
    """
    Wrapper for vLLM server to get ground truth scores from a Teacher model.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "default"):
        """
        Initialize the Teacher LLM wrapper.
        
        Args:
            base_url: vLLM server URL (default: http://localhost:8000)
            model_name: Name of the model to use
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.completions_url = f"{self.base_url}/v1/completions"
    
    def get_ground_truth_score(self, question: str, answer: str, context: str = "") -> float:
        """
        Get a ground truth score (0-100) from the Teacher model.
        
        Args:
            question: The question being answered
            answer: The LLM-generated answer to evaluate
            context: Optional context/source material
            
        Returns:
            A float score between 0 and 100
        """
        context_block = f"\nContext: {context}" if context else ""
        
        prompt = f"""You are an expert evaluator for Machine Learning and Deep Learning content.
Evaluate the following answer on a scale of 0-100 based on:
- Technical accuracy
- Clarity of explanation
- Depth of coverage
- Practical relevance

Question: {question}{context_block}
Answer: {answer}

Provide ONLY a numerical score between 0 and 100. No explanation.
Score:"""

        try:
            response = requests.post(
                self.completions_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "stop": ["\n"]
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            score_text = result["choices"][0]["text"].strip()
            
            # Parse the score
            score = float(score_text)
            return max(0.0, min(100.0, score))  # Clamp to 0-100
            
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"Error getting ground truth: {e}")
            return 50.0  # Default fallback score
    
    def health_check(self) -> bool:
        """Check if the vLLM server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
