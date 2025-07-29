"""
Agent module for creating and managing different personas with their own knowledge bases.
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Import local modules
from src.document_processor import DocumentProcessor
from src.retrieval_engine import RetrievalEngine, SearchResult
from src.embedding_manager import EmbeddingManager
from src.storage_manager import StorageManager

# Define the base directory for agents
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception class for Agent-related errors."""
    pass


class ConfigError(AgentError):
    """Exception raised for configuration-related errors."""
    pass


class DocumentProcessingError(AgentError):
    """Exception raised for errors during document processing."""
    pass


class RetrievalError(AgentError):
    """Exception raised for errors during retrieval operations."""
    pass


class LLMError(AgentError):
    """Exception raised for errors related to LLM operations."""
    pass


class Agent:
    """
    Agent class representing a specific persona with its own knowledge base and embedding store.
    
    Each agent has:
    1. A persona (system prompt)
    2. Its own document collection
    3. Its own embedding store
    4. Retrieval capabilities
    5. Chat memory
    6. A greeting message
    7. A configurable LLM
    """
    
    def __init__(
        self, 
        agent_id: str, 
        config_path: Union[str, Path], 
        agent_config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize an agent with its own configuration.
        
        Args:
            agent_id: Unique identifier for the agent
            config_path: Path to the base configuration file
            agent_config_path: Path to agent-specific configuration
            
        Raises:
            ConfigError: If there's an issue with the configuration
            ValueError: If agent_id is invalid
        """
        # Validate agent_id
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a non-empty string")
        
        if any(c in agent_id for c in "\\/?*:|\"<>"):
            raise ValueError("Agent ID contains invalid characters")
            
        self.agent_id = agent_id
        
        # Check if agent directory exists
        agent_dir = BASE_DIR / f"agents/{agent_id}"
        if not agent_dir.exists():
            raise ValueError(f"Agent directory for '{agent_id}' does not exist")
        
        # Load base config
        try:
            self.config = self._load_config(config_path)
            if not self.config:
                raise ConfigError(f"Failed to load base configuration from {config_path}")
                
            # Ensure required sections exist
            for section in ["general", "embeddings", "chunking"]:
                if section not in self.config:
                    self.config[section] = {}
                    
            # Ensure general section has required fields
            if "general" in self.config and isinstance(self.config["general"], dict):
                if "persist_dir" not in self.config["general"]:
                    self.config["general"]["persist_dir"] = "storage"
                if "temp_dir" not in self.config["general"]:
                    self.config["general"]["temp_dir"] = "temp"
        except Exception as e:
            raise ConfigError(f"Error initializing agent configuration: {str(e)}") from e
        
        # Load agent-specific config if provided
        try:
            if agent_config_path:
                self.agent_config = self._load_config(agent_config_path)
            else:
                # Check for agent config in the default location
                default_agent_config_path = BASE_DIR / f"agents/{agent_id}/config.yaml"
                if default_agent_config_path.exists():
                    self.agent_config = self._load_config(default_agent_config_path)
                else:
                    self.agent_config = {}
        except Exception as e:
            logger.warning(f"Error loading agent-specific config, using defaults: {str(e)}")
            self.agent_config = {}
        
        # Merge base config with agent-specific config
        self._merge_config()
        
        # Set up agent-specific paths
        self.agent_dir = BASE_DIR / f"agents/{agent_id}"
        self.sources_dir = self.agent_dir / "sources"
        self.persona_path = self.agent_dir / "persona.md"
        self.greeting_path = self.agent_dir / "greeting.txt"
        
        # Ensure agent directories exist
        try:
            os.makedirs(self.agent_dir, exist_ok=True)
            os.makedirs(self.sources_dir, exist_ok=True)
            
            # Also create storage and temp directories
            storage_dir = self.agent_dir / "storage"
            temp_dir = self.agent_dir / "temp"
            os.makedirs(storage_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Failed to create agent directories: {str(e)}") from e
        
        # Update storage paths to be agent-specific
        self.config["general"]["persist_dir"] = str(self.agent_dir / "storage")
        self.config["general"]["temp_dir"] = str(self.agent_dir / "temp")
        
        # Initialize components
        try:
            self.document_processor = DocumentProcessor(self.config)
            self.storage_manager = StorageManager(self.config)
            self.embedding_manager = EmbeddingManager(self.config)
            self.retrieval_engine = RetrievalEngine(self.config)
        except Exception as e:
            raise ConfigError(f"Failed to initialize agent components: {str(e)}") from e
        
        # Set up chat memory and conversation history
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        self.conversation_history = []
        
        # Load persona if exists, or create default
        self.persona = self._load_persona()
        
        # Load greeting if exists, or create default
        self.greeting = self._load_greeting()
        
        logger.info(f"Agent '{agent_id}' initialized")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary with configuration values
            
        Raises:
            ConfigError: If the config file doesn't exist or is invalid
        """
        import yaml
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                raise ConfigError(f"Configuration file not found: {config_path}")
                
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            if config is None:
                logger.warning(f"Empty config file: {config_path}")
                return {}
                
            if not isinstance(config, dict):
                raise ConfigError(f"Invalid config file format (not a dictionary): {config_path}")
                
            return config
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in config file {config_path}: {e}")
            raise ConfigError(f"Invalid YAML in configuration file: {e}") from e
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise ConfigError(f"Error loading configuration file: {str(e)}") from e
    
    def _merge_config(self):
        """
        Merge base config with agent-specific config.
        
        Raises:
            ConfigError: If the merge operation fails
        """
        try:
            # Recursively update the base config with agent-specific settings
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            self.config = update_dict(self.config, self.agent_config)
        except Exception as e:
            logger.error(f"Error merging configurations: {e}")
            raise ConfigError(f"Failed to merge configurations: {str(e)}") from e
    
    def _load_persona(self) -> str:
        """
        Load the agent's persona from file or return a default.
        
        Returns:
            String containing the agent's persona
            
        Raises:
            ConfigError: If there's an error reading or writing the persona file
        """
        try:
            if self.persona_path.exists():
                with open(self.persona_path, "r", encoding="utf-8") as f:
                    persona = f.read()
                if not persona.strip():
                    # File exists but is empty, create default
                    persona = self._create_default_persona()
                return persona
            else:
                # Create a default persona file
                return self._create_default_persona()
        except Exception as e:
            logger.error(f"Error loading persona: {e}")
            raise ConfigError(f"Failed to load persona: {str(e)}") from e
            
    def _create_default_persona(self) -> str:
        """
        Create a default persona for the agent.
        
        Returns:
            Default persona string
            
        Raises:
            ConfigError: If there's an error creating the default persona
        """
        try:
            default_persona = f"""# {self.agent_id.replace('_', ' ').title()} Persona

You are an AI assistant specializing in {self.agent_id.replace('_', ' ')}.
Respond to queries based on your knowledge and the provided context.
"""
            with open(self.persona_path, "w", encoding="utf-8") as f:
                f.write(default_persona)
                
            return default_persona
        except Exception as e:
            logger.error(f"Error creating default persona: {e}")
            raise ConfigError(f"Failed to create default persona: {str(e)}") from e
    
    def _load_greeting(self) -> str:
        """
        Load the agent's greeting from file or return a default.
        
        Returns:
            String containing the agent's greeting
            
        Raises:
            ConfigError: If there's an error reading or writing the greeting file
        """
        try:
            if self.greeting_path.exists():
                with open(self.greeting_path, "r", encoding="utf-8") as f:
                    greeting = f.read()
                if not greeting.strip():
                    # File exists but is empty, create default
                    greeting = self._create_default_greeting()
                return greeting
            else:
                # Create a default greeting
                return self._create_default_greeting()
        except Exception as e:
            logger.error(f"Error loading greeting: {e}")
            raise ConfigError(f"Failed to load greeting: {str(e)}") from e
            
    def _create_default_greeting(self) -> str:
        """
        Create a default greeting for the agent.
        
        Returns:
            Default greeting string
        """
        try:
            default_greeting = f"Hello! I'm a {self.agent_id.replace('_', ' ')} assistant. How can I help you today?"
            with open(self.greeting_path, "w", encoding="utf-8") as f:
                f.write(default_greeting)
                
            return default_greeting
        except Exception as e:
            logger.error(f"Error creating default greeting: {e}")
            # Return a basic greeting even if file write fails
            return f"Hello! I'm an AI assistant. How can I help you today?"
    
    def _get_llm(self):
        """
        Get the LLM based on agent configuration.
        
        Returns:
            LLM instance
            
        Raises:
            LLMError: If there's an error initializing the LLM
        """
        try:
            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "openai")
            model = llm_config.get("model", "gpt-4o")
            
            if provider == "openai":
                return OpenAI(model=model)
            else:
                # Default to OpenAI if provider not supported
                logger.warning(f"LLM provider '{provider}' not supported, using OpenAI")
                return OpenAI(model="gpt-4o")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise LLMError(f"Failed to initialize LLM: {str(e)}") from e
    
    def process_sources(self, recursive: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process all documents in the agent's sources directory.
        
        Args:
            recursive: Whether to process subdirectories
            metadata: Optional metadata to attach to all documents
            
        Raises:
            DocumentProcessingError: If there's an error processing documents
            ValueError: If the sources directory is empty
        """
        if not self.sources_dir.exists():
            logger.warning(f"Sources directory for agent '{self.agent_id}' doesn't exist")
            os.makedirs(self.sources_dir, exist_ok=True)
            raise ValueError(f"No sources directory found for agent '{self.agent_id}'. The directory has been created, but you need to add source files.")
        
        # Check if there are any files in the sources directory
        if not any(self.sources_dir.iterdir()):
            logger.warning(f"No source files found in {self.sources_dir}")
            raise ValueError(f"No source files found in {self.sources_dir}. Please add some files before processing.")
        
        # Convert metadata to JSON if provided
        metadata_str = None
        if metadata:
            try:
                metadata_str = json.dumps(metadata)
            except (TypeError, ValueError) as e:
                logger.error(f"Error converting metadata to JSON: {e}")
                raise ValueError(f"Invalid metadata format: {str(e)}") from e
        
        # Process the agent's documents
        logger.info(f"Processing sources for agent '{self.agent_id}'")
        try:
            self.document_processor.process_directory(
                input_dir=self.sources_dir,
                recursive=recursive,
                custom_metadata=metadata_str
            )
            
            logger.info(f"Source processing complete for agent '{self.agent_id}'")
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise DocumentProcessingError(f"Failed to process documents: {str(e)}") from e
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        filters: Optional[str] = None,
        search_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search the agent's knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters
            search_type: Type of search to perform
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If the query is empty or the search parameters are invalid
            RetrievalError: If there's an error during retrieval
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
            
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
            
        try:
            return self.retrieval_engine.search(
                query=query,
                top_k=top_k,
                filters=filters,
                search_type=search_type
            )
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise RetrievalError(f"Search operation failed: {str(e)}") from e
    
    def get_context(self, query: str, top_k: int = None) -> str:
        """
        Get context for a query from the retrieval engine.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            Formatted context string
        """
        # Use top_k from config if not provided
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)
            
        # Use hybrid search by default
        search_type = self.config.get("retrieval", {}).get("search_type", "hybrid")
        
        try:
            # Get search results
            search_results = self.search(
                query=query,
                    top_k=top_k,
                    search_type=search_type
                )
            
            # Format context
            if search_results:
                context = "Here's relevant information from my knowledge base:\n\n"
                for i, result in enumerate(search_results, 1):
                    # Include metadata if available
                    metadata_str = ""
                    if hasattr(result, "metadata") and result.metadata:
                        # Filter out internal metadata fields
                        user_metadata = {k: v for k, v in result.metadata.items() 
                                       if not k.startswith("_") and k not in ["doc_id", "vector_id"]}
                        if user_metadata:
                            metadata_str = f" [Source: {user_metadata}]"
                    
                    context += f"[{i}] {result.text}{metadata_str}\n\n"
                return context
            else:
                return "I don't have specific information about this in my knowledge base."
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
            return "I don't have specific information about this in my knowledge base."
    
    def chat(self, query: str, context_str: str = None, system_prompt: str = None) -> str:
        """
        Process a chat query with optional context and system prompt.
        
        Args:
            query: User query
            context_str: Optional context to include
            system_prompt: Optional system prompt override
            
        Returns:
            Response string
        """
        # Set defaults if not provided
        if system_prompt is None:
            system_prompt = self.persona
            
        if not context_str and self.retrieval_engine:
            # Get context from retrieval engine
            context_str = self.get_context(query)
            
            # Create a chat engine with context
            try:
                # First get the retriever which is required for ContextChatEngine
                retriever = self.retrieval_engine.get_retriever()
                
                # Get an LLM instance
                llm = self._get_llm()
                
                # Initialize with retriever as the required first parameter
                chat_engine = ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,  # Explicitly pass the LLM
                    system_prompt=system_prompt,
                    memory=self.memory
                )
                
                # Get response from chat engine
                response = chat_engine.chat(query)
                
                # Convert response to string
                response_text = str(response)
                
                # Update conversation history
                # self.conversation_history.append(("user", query))
                # self.conversation_history.append(("assistant", response_text))
                
                return response_text
            except Exception as e:
                logger.error(f"Error creating chat engine: {e}")
                return "I'm sorry, but I'm having trouble with my reasoning system. Please try again later."
    
    def start_chat(self) -> Tuple[str, str]:
        """
        Get the agent's name and greeting message for the start of a chat.
        
        Returns:
            Tuple of (agent_name, greeting_message)
        """
        # Extract agent name from the persona (first heading)
        lines = self.persona.strip().split('\n')
        agent_name = lines[0].replace('# ', '') if lines and lines[0].startswith('# ') else self.agent_id.replace('_', ' ').title()
        
        return agent_name, self.greeting
    
    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """
        Get the complete conversation history.
        
        Returns:
            List of (role, message) tuples
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history and memory."""
        self.conversation_history = []
        self.memory.clear()


class AgentManager:
    """Manages multiple agents."""
    
    def __init__(self, base_config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the agent manager.
        
        Args:
            base_config_path: Path to the base configuration file
            
        Raises:
            ConfigError: If there's an issue with the configuration
        """
        self.base_config_path = base_config_path
        self.agents = {}
        self.agents_dir = Path("agents")
        
        # Ensure agents directory exists
        try:
            os.makedirs(self.agents_dir, exist_ok=True)
            
            # Validate base config exists
            if not Path(base_config_path).exists():
                logger.warning(f"Base config not found at {base_config_path}, will use defaults")
                
        except OSError as e:
            logger.error(f"Error creating agents directory: {e}")
            raise ConfigError(f"Failed to create agents directory: {str(e)}") from e
        
        logger.info("Agent manager initialized")
    
    def create_agent(self, agent_id: str, agent_config_path: Optional[Union[str, Path]] = None) -> Agent:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_config_path: Optional path to agent-specific configuration
            
        Returns:
            The created agent
            
        Raises:
            ValueError: If agent_id is invalid
            ConfigError: If there's an issue with the configuration
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a non-empty string")
            
        if any(c in agent_id for c in "\\/?*:|\"<>"):
            raise ValueError("Agent ID contains invalid characters")
            
        if agent_id in self.agents:
            logger.warning(f"Agent '{agent_id}' already exists, returning existing agent")
            return self.agents[agent_id]
        
        try:
            agent = Agent(agent_id, self.base_config_path, agent_config_path)
            self.agents[agent_id] = agent
            
            return agent
        except Exception as e:
            logger.error(f"Error creating agent '{agent_id}': {e}")
            raise ConfigError(f"Failed to create agent '{agent_id}': {str(e)}") from e
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an existing agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            The agent or None if not found
            
        Raises:
            ValueError: If agent_id is invalid
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a non-empty string")
            
        if agent_id in self.agents:
            return self.agents[agent_id]
        
        # Try to load the agent if it exists
        agent_dir = self.agents_dir / agent_id
        if agent_dir.exists():
            try:
                return self.create_agent(agent_id)
            except Exception as e:
                logger.error(f"Error loading existing agent '{agent_id}': {e}")
                return None
        
        return None
    
    def list_agents(self) -> List[str]:
        """
        List all available agents.
        
        Returns:
            List of agent identifiers
        """
        # Scan the agents directory for all available agents
        agents = []
        if self.agents_dir.exists():
            agents = [d.name for d in self.agents_dir.iterdir() if d.is_dir()]
        
        return agents 