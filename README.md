# 🚀 **AlphaEvolve Agent - Advanced Evolutionary AI System**

An advanced evolutionary agent system that implements **AlphaEvolve-grade** capabilities for self-improving AI agents. The system combines multiple cutting-edge approaches including dual evolution, multi-objective optimization, and human-in-the-loop guidance.

## 🎯 **Quick Start**

### **1. Clone & Setup**
```bash
git clone https://github.com/GirishVerm/AlphaEvolve-Agent.git
cd AlphaEvolve-Agent
python setup.py
```

### **2. Run the Guided Agent**
```bash
python run_guided.py
```

## ✨ **Key Features**

- **🧬 Dual Evolution**: Evolves both task artifacts (code) and agent components (prompts, tools, memory)
- **🎯 Multi-Objective Optimization**: Handles competing objectives with Pareto fronts
- **👥 Human-in-the-Loop**: Interactive guidance and feedback throughout evolution
- **💰 Cost Management**: Budget-aware LLM usage with rate limiting
- **📊 Experiment Tracking**: Full ML-ops integration (MLflow, W&B)
- **🌐 Web Interface**: Real-time visualization of evolution process
- **🔧 Extensible**: Support for Python, SQL, JSON, YAML, Markdown artifacts

## 🔧 **Configuration Setup**

The system requires API credentials to be configured via environment variables:

### **1. Copy Environment Template**
```bash
cp .env.example .env
```

### **2. Configure Your API Keys**
Edit `.env` file with your credentials:

```bash
# Azure OpenAI Configuration (recommended)
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# OR OpenAI Configuration (fallback)
OPENAI_API_KEY=your-openai-api-key-here
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🏗️ **Architecture Overview**

The system is built with a modular, production-ready architecture:

```
palantir-agent/
├── 🧠 Core Components
│   ├── src/                    # Base agent framework
│   │   ├── agent.py           # Core agent class
│   │   ├── document_processor.py
│   │   ├── retrieval_engine.py
│   │   └── embedding_manager.py
│   └── evo_agent/             # Evolutionary system
│       ├── evolutionary_agent.py
│       ├── guided_agent.py
│       ├── multi_objective.py
│       └── cost_manager.py
├── 🌐 Web Interface
│   ├── webapp.py
│   ├── templates/
│   └── static/
└── 📚 Documentation
    ├── README.md
    └── setup.py
```

## 🧠 **How `run_guided.py` Works**

The guided agent performs **dual evolution**: evolving both the **task artifacts** (your code) and the **agent itself** (its prompts, tools, and memory).

### **System Flow**

```mermaid
graph TD
    A[User starts run_guided.py] --> B[Task Definition]
    B --> C[Task Analysis]
    C --> D[Initial Code Generation]
    D --> E[Evolution Cycles]
    E --> F[Agent Evolution]
    F --> G[Final Execution]
    
    subgraph "Evolution Cycles"
        E --> H[User Feedback]
        H --> I[Code Improvement]
        I --> J[Evaluation]
        J --> K[Selection]
        K --> L[Agent Component Evolution]
        L --> M{Continue?}
        M -->|Yes| H
        M -->|No| G
    end
    
    subgraph "Agent Evolution"
        F --> N[Evolve Prompts]
        F --> O[Evolve Tools]
        F --> P[Evolve Memory]
    end
```

### **Dual Evolution Process**

```mermaid
graph LR
    subgraph "Task Artifact Evolution"
        A1[Initial Code] --> A2[User Feedback]
        A2 --> A3[Improved Code]
        A3 --> A4[Evaluation]
        A4 --> A5[Selection]
    end
    
    subgraph "Agent Component Evolution"
        B1[Initial Prompts] --> B2[Evolved Prompts]
        B2 --> B3[Evolved Tools]
        B3 --> B4[Evolved Memory]
    end
    
    A5 --> B1
    B4 --> A1
```

## 📋 **Step-by-Step Process**

### **Step 1: Task Definition**
```
📝 TASK SETUP
Task name: Create a Fibonacci calculator
Description: A function that calculates Fibonacci numbers efficiently
Requirements: 
- Handle positive integers
- Return the nth Fibonacci number
- Include error handling
Success criteria:
- Correct results for test cases
- Efficient performance
- Proper error handling
```

### **Step 2: Task Analysis**
The agent analyzes your task and breaks it down into implementable components.

### **Step 3: Initial Code Generation**
```python
def fibonacci(n):
    if n <= 0:
        raise ValueError("Input must be positive")
    if n == 1 or n == 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)
```

### **Step 4: Evolution Cycles (Up to 5 cycles)**

For each cycle, the system:

1. **Gets User Feedback**: You provide improvement suggestions
2. **Improves Code**: Agent generates improved version
3. **Evaluates Both**: Compares current vs. improved code
4. **Selects Better**: Keeps the better version
5. **Evolves Agent**: Updates agent components every 2 cycles

**Example Evolution Cycle:**
```
🔄 EVOLUTION CYCLE 1
💬 Provide feedback for improvement: Add memoization for better performance

📊 EVALUATION RESULTS:
Current code - Overall: 0.750
Improved code - Overall: 0.920
✅ Improved code selected!
```

### **Step 5: Agent Component Evolution**

Every 2 cycles, the agent evolves its own components:

```python
# Initial Agent Components
prompts = {
    "code_generation": "Create a Python function that meets the given requirements.",
    "code_improvement": "Improve this code by adding error handling, documentation, and optimization.",
    "task_analysis": "Analyze this task and break it down into implementable components.",
    "code_evaluation": "Evaluate this code for correctness, performance, and robustness."
}

tools = {
    "code_tester": "def test_code(code, test_cases): return {'passed': len([t for t in test_cases if eval(t)]), 'total': len(test_cases)}",
    "performance_analyzer": "def analyze_performance(code): return {'complexity': 'O(n)', 'efficiency': 0.8}",
    "code_quality_checker": "def check_quality(code): return {'readability': 0.7, 'maintainability': 0.8, 'documentation': 0.6}"
}

memory = {
    "context_manager": "def store_context(task, result): return {'task': task, 'result': result, 'timestamp': time.time()}",
    "knowledge_base": "def retrieve_knowledge(query): return 'relevant_patterns_and_solutions'",
    "experience_logger": "def log_experience(action, outcome): return {'action': action, 'outcome': outcome, 'success': outcome > 0.5}"
}
```

**Evolved components become more sophisticated** based on the task context and user feedback.

### **Step 6: Final Execution**
```
🎯 EXECUTING THE FINAL TASK...
✅ TASK EXECUTION COMPLETE!
Function: fibonacci
Success Rate: 100.0%

💰 COST STATISTICS:
Total cost: $0.0234
Total requests: 12
Generations: 2
```

## 🎛️ **Configuration**

### **Agent Configuration (guided_agent.py)**
```python
@dataclass
class AgentConfig:
    max_cost: float = 20.0              # Maximum cost per experiment
    evolution_frequency: int = 2         # Evolve agent every N generations
    population_size: int = 3             # Population size for evolution
    max_generations: int = 5             # Maximum evolution cycles
```

### **Evaluation Metrics**
The system evaluates code on three objectives:
- **Correctness** (40%): Does the code work correctly?
- **Performance** (30%): Is the code efficient?
- **Robustness** (30%): Does the code handle edge cases?

## 📊 **Example Session**

```
🚀 Starting Guided Evolutionary Agent...
==================================================
This agent will guide you through:
• Setting up a task
• Analyzing the requirements
• Generating and improving code
• Evolving the agent itself
• Executing the final result
==================================================

📝 TASK SETUP
Task name: Create a Fibonacci calculator
Description: A function that calculates Fibonacci numbers efficiently
Requirements: 
- Handle positive integers
- Return the nth Fibonacci number
- Include error handling
Success criteria:
- Correct results for test cases
- Efficient performance
- Proper error handling

🔄 EVOLUTION CYCLE 1
💬 Provide feedback for improvement: Add memoization for better performance

📊 EVALUATION RESULTS:
Current code - Overall: 0.750
Improved code - Overall: 0.920
✅ Improved code selected!

🎯 EXECUTING THE FINAL TASK...
✅ TASK EXECUTION COMPLETE!
Function: fibonacci
Success Rate: 100.0%

💰 COST STATISTICS:
Total cost: $0.0234
Total requests: 12
Generations: 2

🎉 EVOLUTION COMPLETE! The agent has evolved and created working code!
```

## 🎯 **Use Cases**

- **Code Generation**: Evolve better implementations of algorithms
- **Documentation**: Generate and improve technical documentation
- **Data Processing**: Create optimized data transformation pipelines
- **API Development**: Design and refine REST API endpoints
- **Testing**: Generate comprehensive test suites
- **Configuration**: Optimize system configurations

## 🧪 **Advanced Features**

### **Multi-Objective Optimization**
```python
objectives = [
    Objective(name="correctness", weight=0.4, minimize=False),
    Objective(name="performance", weight=0.3, minimize=False),
    Objective(name="robustness", weight=0.3, minimize=False)
]
```

### **Cost Management**
```python
cost_config = CostConfig(
    max_cost_per_experiment=50.0,
    max_requests_per_minute=60,
    token_cost_per_1k=0.03
)
```

### **Human-in-the-Loop**
```python
# Request human review
response = await agent.request_human_review(candidate, context)
if response.approved:
    print("Human approved the candidate!")
```

### **Experiment Tracking**
- **MLflow Integration**: Full experiment tracking and model registry
- **Weights & Biases**: Real-time experiment monitoring
- **Checkpointing**: Resume experiments from any point
- **Artifact Logging**: Save plots, models, and results

### **Scalable Diversity Management**
- **Incremental Clustering**: BIRCH algorithm for large populations
- **Novelty Archives**: Maintains diverse solution history
- **Caching Strategy**: Reduces clustering overhead
- **Memory Management**: Automatic cache clearing

## 🔍 **Key Features**

### **Dual Evolution**
- **Task Artifacts**: Your code evolves through user feedback
- **Agent Components**: Prompts, tools, and memory evolve automatically

### **Intelligent Recommendations**
- AI suggests whether to continue evolution
- Provides feedback suggestions based on code analysis
- Confidence scoring for recommendations

### **Cost Management**
- Budget-aware LLM usage
- Real-time cost tracking
- Automatic stopping when budget exceeded

### **Multi-Objective Evaluation**
- Evaluates code on correctness, performance, and robustness
- Pareto-optimal selection between versions
- Comprehensive fitness scoring

## 🚨 **Troubleshooting**

### **Common Issues**

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Azure API quota exceeded**
   - Check your Azure OpenAI quota
   - Reduce `max_cost` in configuration
   - Wait for quota reset

3. **Agent not responding**
   - Check internet connection
   - Test with `python test_llm.py`

### **Testing Azure Connection**
```bash
python test_llm.py
```

## 📈 **Performance**

- **Scalability**: Handles populations up to 1000+ candidates
- **Efficiency**: O(N log N) diversity selection algorithms
- **Cost-Aware**: Automatic budget management and rate limiting
- **Reproducible**: Full experiment tracking and checkpointing

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Optional
OPENAI_API_KEY=your-openai-key  # Fallback
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your-wandb-key
LOG_LEVEL=INFO
```

### **Agent Configuration**
```python
@dataclass
class AgentConfig:
    max_cost: float = 20.0              # Maximum cost per experiment
    evolution_frequency: int = 2         # Evolve agent every N generations
    population_size: int = 3             # Population size for evolution
    max_generations: int = 5             # Maximum evolution cycles
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is proprietary software. All rights reserved. See the [LICENSE](LICENSE) file for details.

**⚠️ Important**: This software is protected by copyright and may not be used, copied, modified, or distributed without explicit written permission from the copyright holder.

## 🙏 **Acknowledgments**

- Inspired by AlphaEvolve research
- Built with OpenAI and Azure OpenAI
- Uses LlamaIndex for RAG capabilities
- Integrates with MLflow and Weights & Biases

## 📈 **What Makes This Special**

1. **Self-Evolving Agent**: The agent itself evolves its capabilities
2. **Dual Optimization**: Both task artifacts and agent components improve
3. **Guided Interaction**: Step-by-step process with user feedback
4. **Cost-Aware**: Built-in budget management
5. **Multi-Objective**: Evaluates multiple aspects of code quality

The system represents a significant step toward **AlphaEvolve-grade** self-improving agents that can evolve both their outputs and their own capabilities! 🚀
