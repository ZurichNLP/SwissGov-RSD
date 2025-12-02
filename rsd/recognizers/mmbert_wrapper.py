"""
Wrapper for mmBERT models to extract token embeddings.
This allows mmBERT models that are only available with fill-mask task to be used with diffalign 
by extracting hidden states.
"""

import torch
from typing import List, Union, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
import transformers

# Set availability flag
MMBERT_AVAILABLE = True


class MmBERTWrapper:
    """
    A wrapper that makes mmBERT models compatible with the 
    FeatureExtractionPipeline interface. This allows mmBERT models to be used with 
    diffalign by extracting hidden states.
    """
    
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        """
        Initialize the mmBERT model wrapper.
        
        Args:
            model_name_or_path: Name or path of the mmBERT model
            device: Device to use ("auto", "cpu", "cuda", etc.)
        """
        self.model_name_or_path = model_name_or_path
        
        # Load the model and tokenizer
        print(f"Loading mmBERT model: {model_name_or_path}")
        
        try:
            # Try to load the model with trust_remote_code=True first
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load with trust_remote_code=True: {e}")
            print("Trying to load without trust_remote_code...")
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=False,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=False
                )
            except Exception as e2:
                print(f"Failed to load without trust_remote_code: {e2}")
                # If the large model is not available, try the base model
                if "large" in model_name_or_path:
                    base_model_name = model_name_or_path.replace("large", "base")
                    print(f"Trying base model instead: {base_model_name}")
                    try:
                        self.model = AutoModelForMaskedLM.from_pretrained(
                            base_model_name, 
                            trust_remote_code=True,
                            device_map="auto"
                        )
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            base_model_name, 
                            trust_remote_code=True
                        )
                        self.model_name_or_path = base_model_name  # Update the model name
                        print(f"Successfully loaded base model: {base_model_name}")
                    except Exception as e3:
                        print(f"Failed to load base model: {e3}")
                        raise e2  # Raise the original error
                else:
                    raise e2  # Raise the original error
        
        # Set device - for models with device_map="auto", we don't move the model manually
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # For models loaded with device_map="auto", we need to determine the device of the first parameter
        # to ensure inputs are on the same device
        first_param_device = next(self.model.parameters()).device
        self._device = first_param_device
        
        print(f"mmBERT model loaded on device: {self._device}")
    
    def __call__(self, sentences: List[str], **kwargs) -> torch.Tensor:
        """
        Extract token embeddings from the mmBERT model.
        
        Args:
            sentences: List of input sentences
            **kwargs: Additional arguments (layer, etc.)
            
        Returns:
            torch.Tensor: Token embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Get the layer to extract embeddings from (default to last layer)
        layer = kwargs.get('layer', -1)
        
        # Tokenize the sentences
        model_inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        model_inputs = model_inputs.to(self._device)
        
        # Get hidden states from the model
        with torch.no_grad():
            outputs = self.model(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Select the desired layer
            if layer < 0:
                layer = len(hidden_states) + layer
            token_embeddings = hidden_states[layer]
        
        return token_embeddings
    
    @property
    def device(self):
        """Get the device the model is on."""
        return next(self.model.parameters()).device


def is_mmbert_model(model_name_or_path: str) -> bool:
    """
    Check if a model is an mmBERT model that needs special handling.
    
    Args:
        model_name_or_path: Name or path of the model
        
    Returns:
        bool: True if this is an mmBERT model that needs the wrapper
    """
    model_name_str = str(model_name_or_path).lower()
    return ("mmbert" in model_name_str or 
            "jhu-clsp" in model_name_str)


def create_mmbert_pipeline(model_name_or_path: str, device: str = "auto"):
    """
    Create a pipeline-like object for mmBERT models.
    
    Args:
        model_name_or_path: Name or path of the mmBERT model
        device: Device to use
        
    Returns:
        MmBERTWrapper: A wrapper that behaves like a pipeline
    """
    return MmBERTWrapper(model_name_or_path, device=device)
