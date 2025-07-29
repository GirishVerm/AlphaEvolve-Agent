#!/usr/bin/env python3
"""
Main entry point for the Document Processing Framework.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import logging
import yaml
from dotenv import load_dotenv

# Import framework modules
from src.document_processor import DocumentProcessor
from src.retrieval_engine import RetrievalEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Set log level from config
        if config.get("general", {}).get("log_level"):
            numeric_level = getattr(logging, config["general"]["log_level"].upper(), None)
            if isinstance(numeric_level, int):
                logging.getLogger().setLevel(numeric_level)
                
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)


def process_command(args):
    """Process documents based on provided arguments."""
    logger.info(f"Loading config from {args.config_path}")
    config = load_config(args.config_path)
    
    # Create directories if they don't exist
    os.makedirs(config["general"]["persist_dir"], exist_ok=True)
    os.makedirs(config["general"]["temp_dir"], exist_ok=True)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Initialize document processor
    processor = DocumentProcessor(config)
    
    # Process documents
    logger.info(f"Processing documents from {input_dir}")
    recursive = config.get("document_loading", {}).get("recursive", True)
    processor.process_directory(
        str(input_dir),
        recursive=recursive,
        metadata=args.metadata
    )
    
    logger.info("Document processing complete!")


def search_command(args):
    """Perform search based on provided query."""
    logger.info(f"Loading config from {args.config_path}")
    config = load_config(args.config_path)
    
    # Initialize retrieval engine
    retrieval_engine = RetrievalEngine(config)
    
    # Perform search
    logger.info(f"Searching for: {args.query}")
    results = retrieval_engine.search(
        args.query,
        top_k=args.top_k or config.get("retrieval", {}).get("top_k", 5),
        filters=args.filters,
        search_type=getattr(args, "search_type", "hybrid")
    )
    
    # Display results
    print("\nSearch Results:")
    print("==============")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Text: {result.text[:200]}..." if len(result.text) > 200 else f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")
    
    logger.info("Search complete!")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Document Processing Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--input_dir", required=True, help="Input directory containing documents")
    process_parser.add_argument("--config_path", default="config.yaml", help="Path to configuration file")
    process_parser.add_argument("--metadata", type=str, help="JSON string or file path for custom metadata")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search processed documents")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--config_path", default="config.yaml", help="Path to configuration file")
    search_parser.add_argument("--top_k", type=int, help="Number of results to return")
    search_parser.add_argument("--filters", type=str, help="Filters in format 'field:value,field:value'")
    search_parser.add_argument("--search_type", type=str, help="Search type", default="hybrid")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 