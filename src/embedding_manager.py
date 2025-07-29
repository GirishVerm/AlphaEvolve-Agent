"""
Embedding manager module for handling different embedding models.
"""
import os
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Union

# Import real embedding providers
try:
    # OpenAI embeddings
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

try:
    # HuggingFace embeddings
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch = None

try:
    # Sentence Transformer embeddings
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class OpenAIEmbedding:
    """OpenAI embedding implementation."""
    
    def __init__(self, model_name="text-embedding-3-large", api_key=None, dimensions=None, embed_batch_size=10, **kwargs):
        """
        Initialize OpenAI embedding.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            dimensions: Optional embedding dimensions
            embed_batch_size: Batch size for embeddings
            **kwargs: Additional parameters
            
        Raises:
            ImportError: If OpenAI package is not installed
            ValueError: If API key is not provided and not in environment variables
        """
        if openai is None:
            raise ImportError("OpenAI is required. Please install it with 'pip install openai'")
            
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Validate API key is present
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided either as parameter or in environment variables")
            
        self.dimensions = dimensions
        self.embed_batch_size = embed_batch_size
        self.kwargs = kwargs
        
        # Create OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            ValueError: If text is empty
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
            
        params = {"model": self.model_name, "input": text}
        if self.dimensions:
            params["dimensions"] = self.dimensions
            
        try:
            response = self.client.embeddings.create(**params)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {str(e)}")
            raise
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding values for each text
            
        Raises:
            ValueError: If texts list is empty or contains empty strings
            Exception: If API call fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
            
        # Validate input texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
        
        all_embeddings = []
        
        # Process in batches
        try:
            for i in range(0, len(texts), self.embed_batch_size):
                batch = texts[i:i+self.embed_batch_size]
                params = {"model": self.model_name, "input": batch}
                if self.dimensions:
                    params["dimensions"] = self.dimensions
                    
                response = self.client.embeddings.create(**params)
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
                
            return all_embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {str(e)}")
            raise


class HuggingFaceEmbedding:
    """HuggingFace embedding implementation."""
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu", **kwargs):
        """
        Initialize HuggingFace embedding.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda, etc.)
            **kwargs: Additional parameters
        """
        if torch is None:
            raise ImportError("PyTorch and transformers are required. Please install them with 'pip install torch transformers'")
            
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Model output
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        # Tokenize and convert to tensor
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling and convert to list
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings[0].cpu().tolist()
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding values for each text
        """
        # Tokenize and convert to tensor
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling and convert to list
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Alias for get_text_embedding."""
        return self.get_text_embedding(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias for get_text_embeddings."""
        return self.get_text_embeddings(texts)


class SentenceTransformerEmbedding:
    """Sentence Transformer embedding implementation."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu", **kwargs):
        """
        Initialize Sentence Transformer embedding.
        
        Args:
            model_name: Model name
            device: Device to use (cpu, cuda, etc.)
            **kwargs: Additional parameters
        """
        if SentenceTransformer is None:
            raise ImportError("Sentence-Transformers is required. Please install it with 'pip install sentence-transformers'")
            
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        return self.model.encode(text).tolist()
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding values for each text
        """
        return self.model.encode(texts).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Alias for get_text_embedding."""
        return self.get_text_embedding(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias for get_text_embeddings."""
        return self.get_text_embeddings(texts)


class EmbeddingManager:
    """Manages embedding models and provides a unified interface for generating embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding manager with the specified configuration.
        
        Args:
            config: Configuration dictionary with embedding settings
            
        Raises:
            ValueError: If no suitable embedding model can be initialized
        """
        self.config = config
        self.embedding_config = config.get("embeddings", {})
        self.provider = self.embedding_config.get("provider", "openai")
        
        # Extract model name from config
        self.model_name = self.embedding_config.get("model", "text-embedding-3-large")
        
        # Setup caching
        self.cache_dir = self.embedding_config.get("cache_dir")
        self.cache_size_limit = self.embedding_config.get("cache_size_limit", 1000)
        self._memory_cache = {}  # LRU cache could be used here for production
        
        # Initialize the embedding model
        self.embedding_model = self._init_embedding_model()
        if self.embedding_model is None:
            raise ValueError("Failed to initialize any embedding model")
        
        logger.info(f"Embedding manager initialized with provider: {self.provider}, model: {self.model_name}")
    
    def _init_embedding_model(self):
        """
        Initialize the embedding model based on the configuration.
        
        Returns:
            Embedding model instance or None if initialization fails
        """
        # Try the requested provider first
        if self.provider == "openai":
            try:
                return self._init_openai_embeddings()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                if not self.embedding_config.get("allow_fallback", True):
                    raise
        elif self.provider == "huggingface":
            try:
                return self._init_huggingface_embeddings()
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
                if not self.embedding_config.get("allow_fallback", True):
                    raise
        elif self.provider == "local":
            try:
                return self._init_local_embeddings()
            except Exception as e:
                logger.error(f"Failed to initialize local embeddings: {e}")
                if not self.embedding_config.get("allow_fallback", True):
                    raise
        else:
            logger.warning(f"Unknown embedding provider: {self.provider}. Trying fallbacks.")
        
        # If we reach here, try fallbacks in order if allowed
        if self.embedding_config.get("allow_fallback", True):
            # Try local models first as they don't require API keys
            if self.provider != "local":
                try:
                    logger.info("Falling back to local embedding model")
                    return self._init_local_embeddings()
                except Exception as e:
                    logger.warning(f"Failed to initialize local fallback: {e}")
            
            # Try HuggingFace next
            if self.provider != "huggingface":
                try:
                    logger.info("Falling back to HuggingFace embedding model")
                    return self._init_huggingface_embeddings()
                except Exception as e:
                    logger.warning(f"Failed to initialize HuggingFace fallback: {e}")
            
            # Try OpenAI as last resort
            if self.provider != "openai":
                try:
                    logger.info("Falling back to OpenAI embedding model")
                    return self._init_openai_embeddings()
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI fallback: {e}")
        
        # If we reach here, all initialization attempts failed
        return None
    
    def _init_openai_embeddings(self):
        """
        Initialize OpenAI embeddings.
        
        Returns:
            OpenAIEmbedding instance
            
        Raises:
            ValueError: If API key is missing and required
        """
        openai_config = self.embedding_config.get("openai", {})
        model_name = openai_config.get("model_name", "text-embedding-3-large")
        dimensions = openai_config.get("dimensions")
        
        embed_kwargs = {}
        if dimensions:
            embed_kwargs["dimensions"] = dimensions
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables")
            
            # If API key is required and not present, raise error
            if openai_config.get("require_api_key", True):
                raise ValueError("OpenAI API key is required but not found in environment variables")
            
        # Create the embedding model
        return OpenAIEmbedding(
            model_name=model_name,
            embed_batch_size=openai_config.get("embed_batch_size", 10),
            api_key=api_key,
            **embed_kwargs
        )
    
    def _init_huggingface_embeddings(self):
        """Initialize HuggingFace embeddings."""
        hf_config = self.embedding_config.get("huggingface", {})
        model_name = hf_config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        device = hf_config.get("device", "cpu")
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            device=device
        )
    
    def _init_local_embeddings(self):
        """Initialize local sentence transformer embeddings."""
        local_config = self.embedding_config.get("local", {})
        model_name = local_config.get("model_name", "all-MiniLM-L6-v2")
        device = local_config.get("device", "cpu")
        
        return SentenceTransformerEmbedding(
            model_name=model_name,
            device=device
        )
    
    def get_embeddings(self):
        """
        Get the configured embedding model.
        
        Returns:
            Embedding model instance
        """
        return self.embedding_model
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key as string
        """
        # Use a hash of the text + model name as the cache key
        text_to_hash = f"{self.model_name}:{text}"
        return hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Optional[str]:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            File path or None if caching is disabled
        """
        if not self.cache_dir:
            return None
            
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except (OSError, IOError) as e:
            logger.warning(f"Failed to create cache directory: {e}")
            return None
        
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from file cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found/error
        """
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path or not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.warning(f"Error reading from cache: {e}")
            # Remove corrupted cache file
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except OSError:
                pass
                
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to file cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path:
            return
            
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(embedding, f)
        except (IOError, OSError) as e:
            logger.warning(f"Error saving to cache: {e}")
            
    def _manage_memory_cache(self):
        """
        Manage the memory cache size to prevent unbounded growth.
        Removes oldest entries when cache exceeds size limit.
        """
        if len(self._memory_cache) > self.cache_size_limit:
            # Simple approach: remove oldest entries (first 20%)
            items_to_remove = int(self.cache_size_limit * 0.2)
            for _ in range(items_to_remove):
                if self._memory_cache:
                    self._memory_cache.pop(next(iter(self._memory_cache)))
                    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query text.
        
        Args:
            query: The query text to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            ValueError: If the query is empty or invalid
            Exception: If the embedding model fails
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query")
            
        # First check in-memory cache
        if query in self._memory_cache:
            return self._memory_cache[query]
        
        # Then check file cache if enabled
        cached_embedding = self._get_from_cache(query)
        if cached_embedding is not None:
            self._memory_cache[query] = cached_embedding
            self._manage_memory_cache()
            return cached_embedding
            
        # If not in cache, generate embedding
        embedding = self.embedding_model.embed_query(query)
        
        # Update caches
        self._memory_cache[query] = embedding
        self._manage_memory_cache()
        self._save_to_cache(query, embedding)
        
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of document texts.
        
        Args:
            documents: List of document texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If documents list is empty or contains invalid entries
            Exception: If the embedding model fails
        """
        if not documents:
            return []
            
        # Check cache first
        cached_embeddings = []
        documents_to_embed = []
        document_indices = []
        
        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                raise ValueError(f"Document at index {i} is empty or invalid")
                
            # Check memory cache first
            if doc in self._memory_cache:
                cached_embeddings.append((i, self._memory_cache[doc]))
                continue
                
            # Check file cache
            cached_embedding = self._get_from_cache(doc)
            if cached_embedding is not None:
                self._memory_cache[doc] = cached_embedding
                cached_embeddings.append((i, cached_embedding))
                continue
                
            # Not in cache, add to list to embed
            documents_to_embed.append(doc)
            document_indices.append(i)
        
        # Create results array with None placeholders
        results = [None] * len(documents)
        
        # Add cached embeddings to results
        for idx, embedding in cached_embeddings:
            results[idx] = embedding
        
        # If there are documents to embed, do so now
        if documents_to_embed:
            try:
                new_embeddings = self.embedding_model.embed_documents(documents_to_embed)
                
                # Add new embeddings to results and cache
                for i, (doc_idx, doc) in enumerate(zip(document_indices, documents_to_embed)):
                    embedding = new_embeddings[i]
                    results[doc_idx] = embedding
                    self._memory_cache[doc] = embedding
                    self._save_to_cache(doc, embedding)
                    
                self._manage_memory_cache()
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
        
        return results
    
    # Alias methods for compatibility
    def get_query_embedding(self, query: str) -> List[float]:
        """Alias for embed_query."""
        return self.embed_query(query)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Alias for embed_query."""
        return self.embed_query(text)