"""
Document processor module for handling document loading, chunking, and embedding.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import LlamaIndex components
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.readers.file.base import SimpleDirectoryReader

# Import local modules
from src.embedding_manager import EmbeddingManager
from src.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, chunking, and embedding generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration dictionary with the following structure:
                - general:
                    - persist_dir: Directory to store processed documents (required)
                    - temp_dir: Directory for temporary files (default: "test_temp")
                - chunking:
                    - strategy: Chunking strategy, either "sentence" or "fixed" (default: "sentence")
                    - chunk_size: Size of chunks in tokens for sentence strategy (default: 1024)
                    - chunk_overlap: Overlap between chunks in tokens (default: 200)
                    - paragraph_separator: Separator between paragraphs (default: "\n\n")
                    - fixed_chunk_size: Size of chunks for fixed strategy (default: 512)
                - document_loading:
                    - supported_extensions: List of file extensions to process (default: [".txt", ".md", ".pdf"])
                    - recursive: Whether to process subdirectories (default: True)
                    - ignore_hidden_files: Whether to ignore hidden files (default: True)
        """
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.storage_manager = StorageManager(config)
        
        # Set up directory paths with defaults if not provided
        if "general" not in config:
            logger.warning("No 'general' section in config, using defaults")
            config["general"] = {}
        
        if "persist_dir" not in config.get("general", {}):
            logger.warning("No 'persist_dir' specified in config, using default: 'test_storage'")
            config["general"]["persist_dir"] = "test_storage"
        
        if "temp_dir" not in config["general"]:
            logger.warning("No 'temp_dir' specified in config, using default: 'test_temp'")
            config["general"]["temp_dir"] = "test_temp"
        
        self.persist_dir = Path(config["general"]["persist_dir"])
        self.temp_dir = Path(config["general"]["temp_dir"])
        
        # Ensure directories exist
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set up document loading parameters
        doc_loading_config = config.get("document_loading", {})
        default_extensions = [".txt", ".md", ".pdf"]
        self.supported_extensions = doc_loading_config.get("supported_extensions", default_extensions)
        if self.supported_extensions != default_extensions:
            logger.info(f"Using custom supported extensions: {self.supported_extensions}")
        
        self.recursive = doc_loading_config.get("recursive", True)
        self.ignore_hidden = doc_loading_config.get("ignore_hidden_files", True)
        
        # Extract chunking parameters
        chunk_config = config.get("chunking", {})
        self.chunk_size = chunk_config.get("chunk_size", 1024)
        self.chunk_overlap = chunk_config.get("chunk_overlap", 200)
        
        # Initialize node parser based on configuration
        self.node_parser = self._init_node_parser()
        
        logger.info("Document processor initialized")
    
    def _init_node_parser(self):
        """
        Initialize the node parser based on configuration.
        
        Returns:
            A node parser instance configured according to the settings.
        """
        chunk_config = self.config.get("chunking", {})
        chunk_strategy = chunk_config.get("strategy", "sentence")
        
        if chunk_strategy == "sentence":
            logger.debug(f"Using SentenceSplitter with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            return SentenceSplitter(
                chunk_size=chunk_config.get("chunk_size", 512),
                chunk_overlap=chunk_config.get("chunk_overlap", 50),
                paragraph_separator=chunk_config.get("paragraph_separator", "\n\n")
                # Note: sentence_separator is not a valid parameter for SentenceSplitter
            )
        elif chunk_strategy == "fixed":
            logger.debug(f"Using TokenTextSplitter with chunk_size={chunk_config.get('fixed_chunk_size', 512)}")
            return TokenTextSplitter(
                chunk_size=chunk_config.get("fixed_chunk_size", 512),
                chunk_overlap=chunk_config.get("chunk_overlap", 50)
            )
        else:
            logger.warning(f"Unknown chunking strategy: {chunk_strategy}, falling back to sentence strategy")
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n"
            )

    def _parse_custom_metadata(self, metadata_input: Optional[str]) -> Dict[str, Any]:
        """
        Parse custom metadata from input string or file.
        
        Args:
            metadata_input: JSON string or file path containing metadata
            
        Returns:
            Dictionary of metadata
        """
        if not metadata_input:
            return {}
            
        try:
            # Check if input is a file path
            if os.path.exists(metadata_input):
                with open(metadata_input, 'r') as f:
                    return json.load(f)
            else:
                # Assume input is a JSON string
                return json.loads(metadata_input)
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return {}

    def process_file(self, file_path: Union[str, Path], custom_metadata: Dict[str, Any] = None) -> bool:
        """
        Process a single file and return success status.
        
        This method loads a file, chunks it into nodes, embeds those nodes, and adds them
        to the storage system.
        
        Args:
            file_path: Path to the file to process
            custom_metadata: Optional metadata dictionary to add to the document
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Processing file: {file_path}")
            
            # Create base metadata with file info
            doc_metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "file_size": file_path.stat().st_size,
                "file_path": str(file_path.absolute())
            }
            
            # Add custom metadata
            if custom_metadata:
                if not isinstance(custom_metadata, dict):
                    logger.error(f"Invalid metadata format, expected dict but got {type(custom_metadata)}")
                    return False
                doc_metadata.update(custom_metadata)
            
            # Load document using SimpleDirectoryReader
            logger.debug(f"Loading document: {file_path}")
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            documents = reader.load_data()
            
            # Update metadata for each document
            for doc in documents:
                doc.metadata.update(doc_metadata)
            
            # Chunk documents into nodes
            logger.debug(f"Chunking document into nodes")
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes from document")
            
            # Ensure each node has complete metadata
            for node in nodes:
                node.metadata.update(doc_metadata)
            
            # Generate embeddings and store nodes
            logger.debug(f"Adding nodes to storage")
            self.storage_manager.add_nodes(nodes, self.embedding_manager.get_embeddings())
            
            logger.info(f"Successfully processed {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False

    def process_directory(
        self, 
        input_dir: Union[str, Path], 
        recursive: bool = True,
        custom_metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Process all documents in a directory.
        
        Args:
            input_dir: Directory containing documents to process
            recursive: Whether to process subdirectories
            custom_metadata: Custom metadata dictionary to add to all documents
            
        Returns:
            True if all files were processed successfully, False otherwise
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists() or not input_path.is_dir():
                logger.error(f"Input directory doesn't exist: {input_path}")
                return False
            
            logger.info(f"Processing documents from {input_path}")
            
            # Get all files in the directory
            files = self._get_files(input_path, recursive)
            if not files:
                logger.warning(f"No files found in {input_path}")
                return True
            
            # Process each file
            success_count = 0
            total_files = len(files)
            logger.info(f"Found {total_files} files to process")
            
            for file_path in files:
                if self.process_file(file_path, custom_metadata):
                    success_count += 1
            
            logger.info(f"Successfully processed {success_count} out of {total_files} files from {input_path}")
            
            # Return True if all files were processed successfully
            return success_count == total_files
        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            return False
    
    def _get_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        Get all files in a directory.
        
        Args:
            directory: Directory to scan
            recursive: Whether to include subdirectories
            
        Returns:
            List of file paths
        """
        pattern = "**/*" if recursive else "*"
        return [f for f in directory.glob(pattern) if f.is_file()]
    
    def process_document(self, file_path: Union[str, Path], custom_metadata: Dict[str, Any] = None) -> None:
        """
        Process a single document: load, chunk, embed, and store.
        
        Args:
            file_path: Path to document file
            custom_metadata: Custom metadata to add to document
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Create base metadata with file info
            file_path = Path(file_path)
            doc_metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "file_size": file_path.stat().st_size,
                "file_path": str(file_path.absolute())
            }
            
            # Add custom metadata
            if custom_metadata:
                doc_metadata.update(custom_metadata)
            
            # Load document based on file type
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents = [Document(text=text, metadata=doc_metadata)]
            elif file_path.suffix.lower() in ['.pdf', '.docx', '.pptx']:
                # For these document types, we'd typically use a specialized reader
                # This is a simplified version that assumes we have a reader function
                documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                # Update metadata for each document
                for doc in documents:
                    doc.metadata.update(doc_metadata)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return
            
            # Chunk documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Ensure each node has complete metadata
            for node in nodes:
                node.metadata.update(doc_metadata)
            
            logger.info(f"Created {len(nodes)} chunks from {file_path}")
            
            # Generate embeddings and store nodes
            self.storage_manager.add_nodes(nodes, self.embedding_manager.get_embeddings())
            
            logger.info(f"Successfully processed {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise 