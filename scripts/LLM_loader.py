from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLM_loader:
    _instance = None  # Singleton instance

    def __new__(cls, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        if cls._instance is None:
            print("Initializing model loader...")

            cls._instance = super(LLM_loader, cls).__new__(cls)
            cls._instance.model_name = model_name

            # Check if a GPU is available
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {cls._instance.device}")

            # Load model and tokenizer
            print("Loading model...")
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(cls._instance.model_name)
            cls._instance.model = AutoModelForCausalLM.from_pretrained(
                cls._instance.model_name, 
                torch_dtype=torch.float16 
            ).to(cls._instance.device)
            
            print("Model loaded successfully.")
        
        return cls._instance

    def get_model(self):
        """Returns the loaded model and tokenizer."""
        return self.model, self.tokenizer