"""
Wrapper for sentence-transformers models to work with diffalign.
This allows models like LaBSE that are designed for sentence-transformers to be used with diffalign.
"""

import torch
from typing import List, Union, Optional
from transformers import FeatureExtractionPipeline, Pipeline, AutoTokenizer, AutoModel
import transformers

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SentenceTransformerWrapper:
    """
    A wrapper that makes sentence-transformers models compatible with the FeatureExtractionPipeline interface.
    This allows sentence-transformers models to be used with diffalign.
    """
    
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        """
        Initialize the sentence-transformer wrapper.
        
        Args:
            model_name_or_path: Name or path of the sentence-transformer model
            device: Device to use ("auto", "cpu", "cuda", etc.)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        
        self.model_name_or_path = model_name_or_path
        self.sentence_transformer = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
        
        # Extract the underlying model and tokenizer for compatibility
        # The sentence-transformer wraps the model in a DenseTransformer module
        self.model = self.sentence_transformer._modules['0'].auto_model
        self.tokenizer = self.sentence_transformer._modules['0'].tokenizer
        
        # Debug info (can be removed in production)
        # print(f"Debug: Model type: {type(self.model)}")
        # print(f"Debug: Tokenizer type: {type(self.tokenizer)}")
        # print(f"Debug: Model device: {next(self.model.parameters()).device}")
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def __call__(self, inputs, **kwargs):
        """
        Make the wrapper callable like a transformers pipeline.
        This method extracts token-level embeddings from the sentence-transformer model.
        """
        # Handle both single strings and lists of strings
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.tokenizer.model_max_length,  # use model's actual max length
            trust_remote_code=True
        )
        
        # Move to device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        # Get model outputs with hidden states
        with torch.no_grad():
            outputs = self.model(**model_inputs, output_hidden_states=True, **kwargs)
        
        # Return the last hidden state (token-level embeddings)
        # This matches what FeatureExtractionPipeline returns
        # Shape should be [batch_size, seq_len, hidden_size]
        token_embeddings = outputs.hidden_states[-1]
        
        # Debug info (can be removed in production)
        # print(f"Debug: Input sentences: {inputs}")
        # print(f"Debug: Token embeddings shape: {token_embeddings.shape}")
        # print(f"Debug: Input IDs shape: {model_inputs['input_ids'].shape}")
        
        # Ensure we return the correct shape for diffalign
        # diffalign expects token-level embeddings for each sentence
        return token_embeddings
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Encode sentences using the sentence-transformer model.
        This returns sentence-level embeddings (pooled).
        """
        return self.sentence_transformer.encode(sentences, **kwargs)
    
    def to(self, device):
        """Move the model to a specific device."""
        self.device = torch.device(device)
        self.model = self.model.to(device)
        self.sentence_transformer = self.sentence_transformer.to(device)
        return self


def create_sentence_transformer_pipeline(model_name_or_path: str, device: str = "auto") -> FeatureExtractionPipeline:
    """
    Create a FeatureExtractionPipeline-compatible wrapper for sentence-transformers models.
    
    Args:
        model_name_or_path: Name or path of the sentence-transformer model
        device: Device to use
        
    Returns:
        A pipeline-like object that can be used with diffalign
    """
    wrapper = SentenceTransformerWrapper(model_name_or_path, device)
    
    # Create a mock pipeline object that has the expected interface
    class MockPipeline:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.model = wrapper.model
            self.tokenizer = wrapper.tokenizer
            self.device = wrapper.device
            
        def __call__(self, inputs, **kwargs):
            return self.wrapper(inputs, **kwargs)
            
        def to(self, device):
            self.wrapper.to(device)
            self.device = self.wrapper.device
            return self
    
    return MockPipeline(wrapper)


def is_sentence_transformer_model(model_name_or_path: str) -> bool:
    """
    Check if a model is a sentence-transformer model.
    
    Args:
        model_name_or_path: Name or path of the model
        
    Returns:
        True if this is likely a sentence-transformer model
    """
    # Common sentence-transformer model prefixes and patterns
    sentence_transformer_patterns = [
        "sentence-transformers/",
        "LaBSE",
        "all-MiniLM",
        "all-mpnet",
        "paraphrase-",
        "distilbert-base-nli",
        "roberta-base-nli",
        "bert-base-nli",
        "stsb-",
        "nli-",
        "multilingual-",
        "google/embeddinggemma-"
    ]
    
    model_name_lower = model_name_or_path.lower()
    return any(pattern.lower() in model_name_lower for pattern in sentence_transformer_patterns)
