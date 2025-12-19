import os
from llama_cpp import Llama
from typing import Optional

class LlamaWrapper:
    def __init__(self, model_path:str, n_ctx:int=2048, n_gpu_layers:int=-1):
        """
        Initialize the Llama model.
        Args:
            model_path: Path to the .gguf model file
            n_ctx: Context window size (default 2048)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        # Ensure path is absolute if it starts with / or ~
        model_path = os.path.expanduser(model_path)
        model_path = os.path.abspath(model_path)
        
        self.model=Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
    
    def get_verdict(self, prompt:str)->str:
        """
        Run the LLM on the prompt and get the text response.
        """
        output=self.model(
            prompt,
            max_tokens=50,
            temperature=0.1, #deterministic
            stop=["\n", "Constraint:", "User:"] #Stop generation early
        )
        return output["choices"][0]["text"].strip()

    
