#!/usr/bin/env python3
"""
CLI script for managing agents.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for local imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
#sys.path.insert(0, str(Path(__file__).parent))

# Import agent module
from src.agent import Agent, AgentManager
from src.agent_templates import (
    get_template_categories,
    get_templates,
    get_template,
    create_agent_from_template,
    copy_sources_to_agent
)
from src.retrieval_engine import RetrievalEngine

def create_agent_command(args):
    """Command to create a new agent."""
    agent_manager = AgentManager(args.config)
    
    # Check if using a template
    if args.template:
        template_type = args.template
        template = get_template(template_type)
        
        if not template:
            logger.error(f"Unknown template type: {template_type}")
            print(f"\nAvailable templates: {', '.join(get_templates().keys())}")
            return
        
        # Collect template variables
        template_vars = {}
        if template_type == "expert":
            template_vars["agent_name"] = args.agent_id.replace('_', ' ').title()
            template_vars["domain_name"] = input("Enter domain name (e.g., 'Machine Learning'): ")
            template_vars["expertise_point_1"] = input("Enter first expertise point: ")
            template_vars["expertise_point_2"] = input("Enter second expertise point: ")
            template_vars["expertise_point_3"] = input("Enter third expertise point: ")
        
        elif template_type == "character":
            template_vars["character_name"] = args.agent_id.replace('_', ' ').title()
            template_vars["character_description"] = input("Enter character description: ")
            template_vars["background_point_1"] = input("Enter first background point: ")
            template_vars["background_point_2"] = input("Enter second background point: ")
            template_vars["background_point_3"] = input("Enter third background point: ")
            template_vars["personality_point_1"] = input("Enter first personality trait: ")
            template_vars["personality_point_2"] = input("Enter second personality trait: ")
            template_vars["personality_point_3"] = input("Enter third personality trait: ")
            template_vars["knowledge_point_1"] = input("Enter first knowledge area: ")
            template_vars["knowledge_point_2"] = input("Enter second knowledge area: ")
            template_vars["knowledge_point_3"] = input("Enter third knowledge area: ")
        
        elif template_type == "assistant":
            template_vars["assistant_name"] = args.agent_id.replace('_', ' ').title()
            template_vars["specialty"] = input("Enter specialty (e.g., 'Travel Planning'): ")
            template_vars["capability_1"] = input("Enter first capability: ")
            template_vars["capability_2"] = input("Enter second capability: ")
            template_vars["capability_3"] = input("Enter third capability: ")
        
        # Create agent from template
        agent_dir = create_agent_from_template(
            agent_id=args.agent_id,
            template_type=template_type,
            template_variables=template_vars,
            llm_model=args.llm_model
        )
        
        # Now load the agent
        agent = agent_manager.create_agent(args.agent_id)
        
        print(f"\nAgent '{args.agent_id}' created from {template_type} template at {agent_dir}")
    else:
        # Create agent normally
        agent = agent_manager.create_agent(args.agent_id, args.agent_config)
        logger.info(f"Agent '{args.agent_id}' created at {agent.agent_dir}")
    
    # Create directories
    logger.info(f"Sources directory: {agent.sources_dir}")
    print(f"\nTo add sources for this agent, place documents in: {agent.sources_dir}")
    print(f"To customize the agent's persona, edit: {agent.persona_path}")
    print(f"To customize the agent's greeting, edit: {agent.greeting_path}")
    
    if args.llm_model:
        print(f"Agent is configured to use LLM model: {args.llm_model}")
        # Ensure LLM model is saved to config
        import yaml
        config_path = Path(agent.agent_dir) / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        
        # Update LLM config
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["provider"] = "openai"  # Default provider
        config["llm"]["model"] = args.llm_model
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

def list_templates_command(args):
    """Command to list available templates."""
    categories = get_template_categories()
    templates = get_templates()
    
    print("\nAvailable Agent Templates:")
    print("=========================")
    
    for template_id, template in templates.items():
        category_desc = categories.get(template_id, "")
        print(f"\n[{template_id}] {category_desc}")
        print(f"  {template['description']}")

def add_sources_command(args):
    """Command to add source files to an agent."""
    agent_manager = AgentManager(args.config)
    
    # Get agent
    agent = agent_manager.get_agent(args.agent_id)
    if not agent:
        logger.error(f"Agent '{args.agent_id}' not found")
        return
    
    # Copy source files
    source_files = args.sources
    copied_count = copy_sources_to_agent(args.agent_id, source_files)
    
    logger.info(f"Copied {copied_count} source files to agent '{args.agent_id}'")
    print(f"\nTo process these sources, run:")
    print(f"python src/manage_agents.py process {args.agent_id}")

def process_sources_command(args):
    """Command to process an agent's sources."""
    agent_manager = AgentManager(args.config)
    
    # Get agent
    agent = agent_manager.get_agent(args.agent_id)
    if not agent:
        logger.error(f"Agent '{args.agent_id}' not found")
        return
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            # Check if input is a file path
            if os.path.exists(args.metadata):
                with open(args.metadata, 'r') as f:
                    metadata = json.load(f)
            else:
                # Assume input is a JSON string
                metadata = json.loads(args.metadata)
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return
    
    # Count source files
    source_count = len(list(Path(agent.sources_dir).glob("*.*")))
    if source_count == 0:
        logger.warning(f"No source files found in {agent.sources_dir}")
        print(f"\nTo add sources, copy files to {agent.sources_dir} or run:")
        print(f"python src/manage_agents.py add-sources {args.agent_id} file1.txt file2.pdf ...")
        if not args.force:
            return
    
    # Process sources
    print(f"Processing {source_count} source files...")
    agent.process_sources(recursive=args.recursive, metadata=metadata)
    
    logger.info(f"Sources processed for agent '{args.agent_id}'")
    print(f"\nYou can now search or chat with this agent:")
    print(f"python src/manage_agents.py chat {args.agent_id}")

def process_command(args):
    """
    Command to process a file or directory - for backward compatibility with tests.
    This bridges the gap between the test expectations and the actual implementation.
    """
    agent_manager = AgentManager(args.config)
    
    # Get agent
    agent = agent_manager.get_agent(args.agent_id)
    if not agent:
        logger.error(f"Agent '{args.agent_id}' not found")
        return
    
    path = Path(args.path) if hasattr(args, 'path') else None
    
    # Process file or directory
    if path and path.is_file():
        print(f"Processing file: {path}")
        agent.document_processor.process_file(str(path), metadata=args.metadata)
        print("Processing complete.")
    elif path and path.is_dir():
        print(f"Processing directory: {path}")
        agent.document_processor.process_directory(str(path), recursive=args.recursive, metadata=args.metadata)
        print("Processing complete.")
    else:
        # Fall back to processing sources directory
        process_sources_command(args)

def list_agents_command(args):
    """Command to list all available agents."""
    agent_manager = AgentManager(args.config)
    
    agents = agent_manager.list_agents()
    
    if not agents:
        print("No agents found.")
        print("\nTo create an agent, run:")
        print("python src/manage_agents.py create agent_id [--template expert|character|assistant]")
        return
    
    print("\nAvailable agents:")
    for agent_id in agents:
        # Try to get agent to load details
        agent = agent_manager.get_agent(agent_id)
        if agent:
            persona_first_line = agent.persona.split('\n')[0].replace('# ', '')
            print(f"- {agent_id}: {persona_first_line}")
        else:
            print(f"- {agent_id}")

def search_command(args):
    """Command to search an agent's knowledge base."""
    agent_manager = AgentManager(args.config)
    
    # Get agent
    agent = agent_manager.get_agent(args.agent_id)
    if not agent:
        logger.error(f"Agent '{args.agent_id}' not found")
        return
    
    # Perform search
    results = agent.search(
        query=args.query,
        top_k=args.top_k,
        filters=args.filters,
        search_type=args.search_type
    )
    
    # Display results
    print(f"\nSearch Results for '{args.query}':")
    print("=" * 50)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Text: {result.text[:200]}..." if len(result.text) > 200 else f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")

def chat_command(args):
    """Command to chat with an agent."""
    agent_manager = AgentManager(args.config)
    
    # Get agent
    agent = agent_manager.get_agent(args.agent_id)
    if not agent:
        logger.error(f"Agent '{args.agent_id}' not found")
        return
    
    # Override retrieval settings if specified
    if args.search_type or args.top_k or args.semantic_weight is not None or args.keyword_weight is not None:
        logger.info("Customizing agent retrieval settings for this session")
        
        # Update retrieval config
        if "retrieval" not in agent.config:
            agent.config["retrieval"] = {}
        
        # Override search type
        if args.search_type:
            agent.config["retrieval"]["search_type"] = args.search_type
            logger.info(f"Setting search type to: {args.search_type}")
        
        # Override top_k
        if args.top_k:
            agent.config["retrieval"]["top_k"] = args.top_k
            logger.info(f"Setting top_k to: {args.top_k}")
        
        # Override hybrid weights
        if args.semantic_weight is not None or args.keyword_weight is not None:
            if "hybrid" not in agent.config["retrieval"]:
                agent.config["retrieval"]["hybrid"] = {}
            
            if args.semantic_weight is not None:
                agent.config["retrieval"]["hybrid"]["semantic_weight"] = args.semantic_weight
                logger.info(f"Setting semantic weight to: {args.semantic_weight}")
            
            if args.keyword_weight is not None:
                agent.config["retrieval"]["hybrid"]["keyword_weight"] = args.keyword_weight
                logger.info(f"Setting keyword weight to: {args.keyword_weight}")
        
        # Re-initialize the retrieval engine with the updated config
        agent.retrieval_engine = RetrievalEngine(agent.config)
    
    # Get agent name and greeting
    agent_name, greeting = agent.start_chat()
    
    # Print retrieval settings info
    search_type = agent.config.get("retrieval", {}).get("search_type", "hybrid")
    top_k = agent.config.get("retrieval", {}).get("top_k", 5)
    print(f"\nUsing {search_type} search with top {top_k} results for retrieval")
    
    if search_type == "hybrid":
        hybrid_config = agent.config.get("retrieval", {}).get("hybrid", {})
        semantic_weight = hybrid_config.get("semantic_weight", 0.7)
        keyword_weight = hybrid_config.get("keyword_weight", 0.3)
        print(f"Hybrid settings: semantic weight={semantic_weight}, keyword weight={keyword_weight}")
    
    # Print agent greeting
    print(f"\nChatting with {agent_name}")
    print("=" * 50)
    print(f"{agent_name}: {greeting}")
    print("\nType 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to reset the conversation history.")
    
    # Start chat session
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            print("\nEnding chat session.")
            break
        
        # Check for clear command
        if user_input.lower() == 'clear':
            agent.clear_conversation_history()
            print("\nConversation history cleared.")
            continue
        
        # Get agent response
        try:
            response = agent.chat(user_input)
            print(f"\n{agent_name}: {response}")
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            print("\nSorry, I encountered an error. Please try again.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Agent Management CLI")
    parser.add_argument('--config', default='config.yaml', help='Path to base configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create agent command
    create_parser = subparsers.add_parser('create', help='Create a new agent')
    create_parser.add_argument('agent_id', help='Unique identifier for the agent')
    create_parser.add_argument('--agent-config', help='Path to agent-specific configuration')
    create_parser.add_argument('--template', choices=list(get_templates().keys()), 
                             help='Create agent from template')
    create_parser.add_argument('--llm-model', help='LLM model to use (e.g., gpt-4o, gpt-3.5-turbo)')
    
    # List templates command
    templates_parser = subparsers.add_parser('templates', help='List available agent templates')
    
    # Add sources command
    add_sources_parser = subparsers.add_parser('add-sources', help='Add source files to agent')
    add_sources_parser.add_argument('agent_id', help='Agent identifier')
    add_sources_parser.add_argument('sources', nargs='+', help='Source files to add')
    
    # Process sources command
    process_parser = subparsers.add_parser('process', help='Process agent sources')
    process_parser.add_argument('agent_id', help='Agent identifier')
    process_parser.add_argument('--recursive', action='store_true', help='Process subdirectories')
    process_parser.add_argument('--metadata', help='JSON string or file path for custom metadata')
    process_parser.add_argument('--force', action='store_true', help='Process even if no sources found')
    
    # List agents command
    list_parser = subparsers.add_parser('list', help='List available agents')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search agent knowledge base')
    search_parser.add_argument('agent_id', help='Agent identifier')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    search_parser.add_argument('--filters', help='Filters in format "field:value,field:value"')
    search_parser.add_argument('--search-type', choices=['hybrid', 'semantic', 'bm25'], help='Search type')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with an agent')
    chat_parser.add_argument('agent_id', help='Agent identifier')
    chat_parser.add_argument('--search-type', choices=['hybrid', 'semantic', 'bm25'], 
                           help='Override search type for this session')
    chat_parser.add_argument('--top-k', type=int, help='Override number of results to retrieve')
    chat_parser.add_argument('--semantic-weight', type=float, 
                           help='Override semantic weight for hybrid search (0.0-1.0)')
    chat_parser.add_argument('--keyword-weight', type=float, 
                           help='Override keyword weight for hybrid search (0.0-1.0)')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_agent_command(args)
    elif args.command == 'templates':
        list_templates_command(args)
    elif args.command == 'add-sources':
        add_sources_command(args)
    elif args.command == 'process':
        process_sources_command(args)
    elif args.command == 'list':
        list_agents_command(args)
    elif args.command == 'search':
        search_command(args)
    elif args.command == 'chat':
        chat_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 