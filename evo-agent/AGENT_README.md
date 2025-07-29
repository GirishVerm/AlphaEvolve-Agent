# Interactive Evolutionary Agent System

An agent that can build code, evolve itself, and interact with you through the terminal.

## Features

- **Task-Driven Development**: Set tasks with requirements and success criteria
- **Code Generation & Evolution**: Generate and improve code based on specifications
- **Self-Evolution**: The agent evolves its own prompts, tools, and memory
- **Interactive Interface**: Command-line interface for real-time interaction
- **Cost Management**: Built-in budget controls for LLM API usage
- **Multi-Objective Evaluation**: Evaluates code for correctness, performance, and robustness

## Quick Start

### 1. Run the Demo
```bash
python3 demo_usage.py
```
This shows a quick demo of the agent working on a markdown-to-HTML converter task.

### 2. Run Interactive Mode
```bash
python3 run_agent.py
```
This starts the interactive agent where you can:
- Set tasks and specifications
- Generate and improve code
- Evaluate code quality
- Watch the agent evolve itself

## Available Commands

When running in interactive mode, you can use these commands:

### Task Management
- `set_task <name> <description> <requirements> <success_criteria>`
  - Sets a new task for the agent
  - Example: `set_task "Calculator" "Create a calculator function" "Add,subtract,multiply" "Handles basic math,error handling"`

### Code Operations
- `analyze_task` - Analyzes the current task
- `generate_code` - Generates initial code for the task
- `improve_code [feedback]` - Improves the current code (optional feedback)
- `evaluate_code` - Evaluates the current code

### Agent Management
- `show_history` - Shows evolution history
- `show_agent_status` - Shows agent status and costs
- `evolve_agent` - Manually trigger agent evolution

### System
- `quit` - Exit the agent

## Example Usage

1. **Start the agent**:
   ```bash
   python3 run_agent.py
   ```

2. **Set a task**:
   ```
   set_task "String Reverser" "Create a function that reverses strings" "Handle basic strings,unicode,empty strings" "Works correctly,handles edge cases,fast performance"
   ```

3. **Analyze the task**:
   ```
   analyze_task
   ```

4. **Generate initial code**:
   ```
   generate_code
   ```

5. **Improve the code**:
   ```
   improve_code "Add better error handling and documentation"
   ```

6. **Evaluate the code**:
   ```
   evaluate_code
   ```

7. **Check agent status**:
   ```
   show_agent_status
   ```

## How It Works

### Dual-Loop Evolution
The agent implements a dual-loop evolutionary system:

1. **Artifact Evolution**: Improves the target code (your task)
2. **Meta-Evolution**: Evolves the agent's own components (prompts, tools, memory)

### Agent Components

The agent has evolvable components:

- **Prompts**: Code generation, improvement, task analysis, evaluation
- **Tools**: Code testing, performance analysis, quality checking
- **Memory**: Context management, knowledge retrieval, experience logging

### Evaluation System

The agent evaluates code using multiple objectives:
- **Correctness**: Does the code work correctly?
- **Performance**: Is the code efficient?
- **Robustness**: Does the code handle edge cases?

## Configuration

You can modify the agent behavior by editing `interactive_agent.py`:

```python
config = AgentConfig(
    max_cost=20.0,              # Maximum cost per session
    evolution_frequency=3,       # Evolve agent every N generations
    population_size=5,           # Population size for evolution
    max_generations=10           # Maximum generations
)
```

## Cost Management

The agent includes built-in cost management:
- Tracks API usage and costs
- Stops when budget is exceeded
- Shows cost statistics in status

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API quota exceeded**:
   - Check your Azure OpenAI quota
   - Reduce `max_cost` in configuration

3. **Agent not responding**:
   - Check internet connection
   - Verify API credentials in `llm_interface.py`

### Getting Help

- Run `python3 demo_usage.py` to see a working example
- Check the logs for detailed error messages
- Use `show_agent_status` to see current state

## Advanced Usage

### Custom Tasks

You can create any programming task:

```
set_task "Web Scraper" "Create a web scraper" "Use requests,parse HTML,extract data" "Handles different sites,error handling,respectful scraping"
```

### Custom Feedback

When improving code, provide specific feedback:

```
improve_code "Add type hints and make it more efficient for large datasets"
```

### Monitoring Evolution

Watch how the agent evolves by checking status regularly:

```
show_agent_status
show_history
```

## Architecture

The system is modular and extensible:

- `interactive_agent.py` - Main agent implementation
- `run_agent.py` - Simple launcher
- `demo_usage.py` - Example usage
- `llm_interface.py` - LLM communication
- `cost_manager.py` - Cost tracking
- `multi_objective.py` - Evaluation system

## Contributing

To extend the system:

1. Add new agent components in `InteractiveAgent.__init__()`
2. Create new evaluation methods in `evaluate_code()`
3. Add new commands in `run_interactive_session()`

The agent will automatically evolve these new components over time! 