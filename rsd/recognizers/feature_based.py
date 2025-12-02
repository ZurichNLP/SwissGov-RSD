import itertools
from typing import List, Union

import torch
import transformers
from transformers import FeatureExtractionPipeline, Pipeline, AutoTokenizer, AutoModel
from torch.nn import DataParallel

from rsd.recognizers.base import DifferenceRecognizer
from rsd.recognizers.utils import DifferenceSample

# Try to import fastembed for sparse models
try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

# Try to import sentence-transformer wrapper
try:
    from rsd.recognizers.sentence_transformer_wrapper import (
        create_sentence_transformer_pipeline, 
        is_sentence_transformer_model,
        SENTENCE_TRANSFORMERS_AVAILABLE
    )
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    def is_sentence_transformer_model(model_name_or_path: str) -> bool:
        return False
    def create_sentence_transformer_pipeline(model_name_or_path: str, device: str = "auto"):
        raise ImportError("sentence-transformers wrapper not available")

# Try to import mmBERT wrapper
try:
    from rsd.recognizers.mmbert_wrapper import (
        create_mmbert_pipeline,
        is_mmbert_model,
        MMBERT_AVAILABLE
    )
    MMBERT_AVAILABLE = True
except ImportError:
    MMBERT_AVAILABLE = False
    def is_mmbert_model(model_name_or_path: str) -> bool:
        return False
    def create_mmbert_pipeline(model_name_or_path: str, device: str = "auto"):
        raise ImportError("mmBERT wrapper not available")

Ngram = List[int]  # A span of subword indices


class FeatureExtractionRecognizer(DifferenceRecognizer):

    def __init__(self,
                 model_name_or_path: str = None,
                 pipeline: Union[FeatureExtractionPipeline, Pipeline] = None,
                 layer: int = -1,
                 batch_size: int = 16,
                 ):
        assert model_name_or_path is not None or pipeline is not None
        if pipeline is None:
            # Check if this is a sparse model that needs fastembed
            model_name_str = str(model_name_or_path)
            print(f"Loading model: {model_name_str}")
            if model_name_str == "Qdrant/bm25":
                if FASTEMBED_AVAILABLE:
                    print("Loading sparse BM25 model with fastembed...")
                    # Try to use GPU if available
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Using device: {device}")
                    self.sparse_model = SparseTextEmbedding(model_name=model_name_str)
                    self.pipeline = None  # No traditional pipeline for sparse models
                    #self.is_sparse = True
                else:
                    raise ImportError("Failed to load Qdrant/bm25 model. Please install fastembed: pip install fastembed (or fastembed-gpu for GPU support)")
            elif is_mmbert_model(model_name_str):
                # Check if this is an mmBERT model that needs special handling
                if MMBERT_AVAILABLE:
                    print(f"Loading mmBERT model: {model_name_str}")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Using device: {device}")
                    pipeline = create_mmbert_pipeline(model_name_str, device=device)
                    #self.is_sparse = False
                    self.is_sentence_transformer = False
                    self.is_mmbert = True
                else:
                    raise ImportError(f"Failed to load mmBERT model {model_name_str}. mmBERT wrapper not available")
            elif is_sentence_transformer_model(model_name_str):
                # Check if this is a sentence-transformer model
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    print(f"Loading sentence-transformer model: {model_name_str}")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Using device: {device}")
                    pipeline = create_sentence_transformer_pipeline(model_name_str, device=device)
                    #self.is_sparse = False
                    self.is_sentence_transformer = True
                    self.is_mmbert = False
                else:
                    raise ImportError(f"Failed to load sentence-transformer model {model_name_str}. Please install sentence-transformers: pip install sentence-transformers")
            else:
                print(f"Model {model_name_str} not detected as sentence-transformer, using traditional transformer path")
                # Standard HuggingFace transformers model
                # Explicitly load tokenizer and model with trust_remote_code=True
                tokenizer = AutoTokenizer.from_pretrained(model_name_str, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name_str, trust_remote_code=True)
                try:
                    if "facebook/MEXMA" in model_name_str:
                        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", trust_remote_code=True, use_fast=False)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model_name_str, trust_remote_code=True)
                except Exception as e:
                    print(f"Failed to load tokenizer with trust_remote_code=True: {e}")
                    print("Trying to load tokenizer without trust_remote_code...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name_str, trust_remote_code=False)
                
                model = AutoModel.from_pretrained(model_name_str, trust_remote_code=True, device_map="auto")
                
                print(f"Model successfully distributed across {torch.cuda.device_count()} GPUs")

                pipeline = transformers.pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task="feature-extraction",
                    device_map="auto",
                )
                #self.is_sparse = False
                self.is_sentence_transformer = False
                self.is_mmbert = False
        else:
            #self.is_sparse = False
            self.is_sentence_transformer = False
            self.is_mmbert = False
            
        self.pipeline = pipeline
        self.layer = layer
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path
        

    def encode_batch(self, sentences: List[str], **kwargs) -> torch.Tensor:
        """if self.is_sparse:
            # Handle sparse models (like BM25)
            embeddings = []
            for sentence in sentences:
                # Get sparse embedding from fastembed
                embedding_generator = self.sparse_model.embed(sentence)
                # Convert generator to list and get the first embedding
                embedding_list = list(embedding_generator)
                print(f"Debug: embedding_list type: {type(embedding_list)}, length: {len(embedding_list)}")
                if len(embedding_list) > 0:
                    embedding = embedding_list[0]
                    print(f"Debug: embedding type: {type(embedding)}, shape: {getattr(embedding, 'shape', 'no shape')}")
                else:
                    # Fallback: create a zero embedding if generator is empty
                    embedding = torch.zeros(1000, dtype=torch.float32)  # Default size
                    embeddings.append(embedding)
                    continue
                
                # Convert SparseEmbedding to dense tensor
                if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                    # This is a SparseEmbedding object from fastembed
                    # Convert to dense vector
                    max_index = max(embedding.indices) if len(embedding.indices) > 0 else 0
                    dense_vector = torch.zeros(max_index + 1, dtype=torch.float32)
                    for idx, val in zip(embedding.indices, embedding.values):
                        dense_vector[idx] = val
                    embedding = dense_vector
                elif hasattr(embedding, 'toarray'):
                    # If it's a scipy sparse matrix, convert to dense
                    embedding = torch.tensor(embedding.toarray(), dtype=torch.float32)
                elif hasattr(embedding, 'numpy'):
                    # If it's a numpy array
                    embedding = torch.tensor(embedding.numpy(), dtype=torch.float32)
                elif isinstance(embedding, (list, tuple)):
                    # If it's a list or tuple
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                else:
                    # If it's already a tensor or other type
                    try:
                        embedding = torch.tensor(embedding, dtype=torch.float32)
                    except:
                        # Last resort: create a zero embedding
                        embedding = torch.zeros(1000, dtype=torch.float32)
                
                embeddings.append(embedding)
            
            # For sparse models, we need to create a sequence of embeddings
            # Since BM25 gives us one embedding per sentence, we'll create word-level embeddings
            batch_embeddings = []
            for i, embedding in enumerate(embeddings):
                # Get the number of words in the sentence
                words = sentences[i].split()
                # Create word-level embeddings by splitting the sentence embedding
                # For BM25, we'll create simple word-level features
                word_embeddings = []
                for word in words:
                    # Create a simple word embedding based on the word's position and the overall sentence embedding
                    word_embedding = embedding.clone()  # Start with the sentence embedding
                    # Add some word-specific variation (this is a simplified approach)
                    word_hash = hash(word) % 1000  # Simple hash-based feature
                    word_embedding[word_hash % len(word_embedding)] += 0.1  # Add small variation
                    word_embeddings.append(word_embedding)
                
                batch_embeddings.extend(word_embeddings)
            
            # Stack all embeddings
            batch_embeddings = torch.stack(batch_embeddings)
            return batch_embeddings"""
        if hasattr(self, 'is_mmbert') and self.is_mmbert:
            # Handle mmBERT models
            # Use the wrapper's __call__ method to get token-level embeddings
            return self.pipeline(sentences, layer=self.layer, **kwargs)
        elif hasattr(self, 'is_sentence_transformer') and self.is_sentence_transformer:
            # Handle sentence-transformer models
            # Process all sentences as a batch to get consistent output format
            embeddings = self.pipeline(sentences, **kwargs)
            # The sentence transformer wrapper returns [batch_size, seq_len, hidden_size]
            # This should match the format expected by DiffAlign
            return embeddings
        else:
            # Handle traditional transformer models
            if str(self.model_name_or_path) == "voidism/diffcse-roberta-base-sts":
                model_inputs = self.pipeline.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                model_inputs = self.pipeline.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

            model_inputs = model_inputs.to(self.pipeline.device)
            
            outputs = self.pipeline.model(**model_inputs, output_hidden_states=True, **kwargs)
            
            return outputs.hidden_states[self.layer]

    def predict(self,
                a: str,
                b: str,
                **kwargs,
                ) -> DifferenceSample:
        return self.predict_all([a], [b], **kwargs)[0]

    def predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        samples = []
        for i in range(0, len(a), self.batch_size):
            samples.extend(self._predict_all(
                a[i:i + self.batch_size],
                b[i:i + self.batch_size],
                **kwargs,
            ))
        return samples

    @torch.no_grad()
    def _predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        raise NotImplementedError

    def _pool(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param token_embeddings: batch x seq_len x dim
        :param mask: batch x seq_len; 1 if token should be included in the pooling
        :return: batch x dim
        Do only sum and do not divide by the number of tokens because cosine similarity is length-invariant.
        """
        return torch.sum(token_embeddings * mask.unsqueeze(-1), dim=1)

    def _get_subwords_by_word(self, sentence: str) -> List[Ngram]:
        """
        :return: For each word in the sentence, the positions of the subwords that make up the word.
        """
        """if self.is_sparse:
            # For sparse models, we'll use simple word-level tokenization
            # since BM25 doesn't have subword tokenization
            words = sentence.split()
            return [[i] for i in range(len(words))]"""
        
        # Handle different model types
        if hasattr(self, 'is_mmbert') and self.is_mmbert:
            # Use the tokenizer from the mmBERT wrapper
            tokenizer = self.pipeline.tokenizer
        elif hasattr(self, 'is_sentence_transformer') and self.is_sentence_transformer:
            # Use the tokenizer from the sentence-transformer wrapper
            tokenizer = self.pipeline.tokenizer
        else:
            # Use the pipeline tokenizer for traditional models
            tokenizer = self.pipeline.tokenizer
        
        if str(self.model_name_or_path) == "voidism/diffcse-roberta-base-sts":
            batch_encoding = tokenizer(
                sentence,
                padding=True,
                truncation=True,
                max_length=512,
            )
        else:
            batch_encoding = tokenizer(
                sentence,
                padding=True,
                truncation=True,
            )
        subword_ids: List[List[int]] = []

        # Debug output for sentence-transformer models (can be removed in production)
        # if hasattr(self, 'is_sentence_transformer') and self.is_sentence_transformer:
        #     print(f"Debug: Tokenizing sentence: '{sentence}'")
        #     print(f"Debug: Word IDs: {batch_encoding.encodings[0].word_ids}")
        #     print(f"Debug: Tokens: {batch_encoding.encodings[0].tokens}")
        #     print(f"Debug: Tokenizer type: {type(tokenizer)}")

        # For BERT-based tokenizers (like LaBSE), use word_ids to group tokens
        if isinstance(tokenizer, transformers.BertTokenizerFast) or \
           isinstance(tokenizer, transformers.BertTokenizer) or \
           "bert" in str(type(tokenizer)).lower():
            # Group tokens by word_id
            current_word_id = None
            for subword_idx in range(len(batch_encoding.encodings[0].word_ids)):
                word_id = batch_encoding.encodings[0].word_ids[subword_idx]
                if word_id is None:  # Special token
                    continue
                if word_id != current_word_id:
                    # New word, start a new group
                    subword_ids.append([subword_idx])
                    current_word_id = word_id
                else:
                    # Same word, add to current group
                    subword_ids[-1].append(subword_idx)
        elif "gemma" in str(type(tokenizer)).lower():
            # For Gemma tokenizers, use SentencePiece-style tokenization
            # Tokens starting with ▁ are word boundaries
            for subword_idx in range(len(batch_encoding.encodings[0].tokens)):
                token = batch_encoding.encodings[0].tokens[subword_idx]
                if token in tokenizer.all_special_tokens:  # Skip special tokens
                    continue
                if token.startswith("▁") or subword_idx == 1:  # Word boundary or first token
                    subword_ids.append([subword_idx])
                else:
                    # Same word, add to current group
                    if subword_ids:
                        subword_ids[-1].append(subword_idx)
                    else:
                        subword_ids.append([subword_idx])
        elif hasattr(self, 'is_mmbert') and self.is_mmbert:
            # For mmBERT models with SentencePiece tokenization, use word_ids
            current_word_id = None
            for subword_idx in range(len(batch_encoding.encodings[0].word_ids)):
                word_id = batch_encoding.encodings[0].word_ids[subword_idx]
                if word_id is None:  # Special token
                    continue
                if word_id != current_word_id:
                    # New word, start a new group
                    subword_ids.append([subword_idx])
                    current_word_id = word_id
                else:
                    # Same word, add to current group
                    subword_ids[-1].append(subword_idx)
        else:
            # Original logic for RoBERTa-based tokenizers
            for subword_idx in range(len(batch_encoding.encodings[0].word_ids)):
                if batch_encoding.encodings[0].word_ids[subword_idx] is None:  # Special token
                    continue
                char_idx = batch_encoding.encodings[0].offsets[subword_idx][0]
                if isinstance(tokenizer, transformers.XLMRobertaTokenizerFast) or \
                        isinstance(tokenizer, transformers.XLMRobertaTokenizer):
                    token = batch_encoding.encodings[0].tokens[subword_idx]
                    is_tail = not token.startswith("▁") and token not in tokenizer.all_special_tokens
                elif isinstance(tokenizer, transformers.RobertaTokenizerFast) or \
                        isinstance(tokenizer, transformers.RobertaTokenizer):
                    token = batch_encoding.encodings[0].tokens[subword_idx]
                    is_tail = not token.startswith("Ġ") and token not in tokenizer.all_special_tokens
                elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) or \
                        isinstance(tokenizer, transformers.PreTrainedTokenizer):
                    token = batch_encoding.encodings[0].tokens[subword_idx]
                    is_tail = not token.startswith("Ġ") and token not in tokenizer.all_special_tokens
                else:
                    is_tail = char_idx > 0 and char_idx == batch_encoding.encodings[0].offsets[subword_idx - 1][1]
                if is_tail and len(subword_ids) > 0:
                    subword_ids[-1].append(subword_idx)
                else:
                    subword_ids.append([subword_idx])
        
        # Debug output for sentence-transformer models (can be removed in production)
        # if hasattr(self, 'is_sentence_transformer') and self.is_sentence_transformer:
        #     print(f"Debug: Subword IDs: {subword_ids}")
        
        return subword_ids

    def _get_ngrams(self, subwords_by_word: List[Ngram]) -> List[Ngram]:
        """
        :return: For each subword ngram in the sentence, the positions of the subwords that make up the ngram.
        """
        subwords = list(itertools.chain.from_iterable(subwords_by_word))
        # Always return at least one ngram (reduce n if necessary)
        min_n = min(self.min_n, len(subwords))
        ngrams = []
        for n in range(min_n, self.max_n + 1):
            for i in range(len(subwords) - n + 1):
                ngrams.append(subwords[i:i + n])
        return ngrams

    def _subword_labels_to_word_labels(self, subword_labels: torch.Tensor, subwords_by_words: List[Ngram]) -> List[float]:
        """
        :param subword_labels: num_subwords
        :param subwords_by_words: num_words x num_subwords
        :return: num_words
        """
        labels = []
        for subword_indices in subwords_by_words:
            label = subword_labels[subword_indices].mean().item()
            labels.append(label)
        return labels
