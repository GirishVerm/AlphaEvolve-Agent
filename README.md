# 🚀 **evo-agent-alpha-avolve-palantir**

An advanced evolutionary agent system with **AlphaEvolve-grade** capabilities, featuring multi-objective optimization, reproducible experiments, cost management, scalable diversity, human-in-the-loop interfaces, and extensible artifact support.

## 📋 **Table of Contents**

- [🚀 Quick Start](#-quick-start)
- [⚙️ Setup & Configuration](#️-setup--configuration)
- [🎯 How to Use](#-how-to-use)
- [🔧 Azure OpenAI Configuration](#-azure-openai-configuration)
- [📚 Examples & Demos](#-examples--demos)
- [🏗️ Architecture](#️-architecture)
- [🧪 Testing](#-testing)
- [📊 Monitoring](#-monitoring)

## 🚀 **Quick Start**

### **1. Clone the Repository**
```bash
git clone https://github.com/GirishVerm/evo-agent-alpha-avolve-palantir.git
cd evo-agent-alpha-avolve-palantir
```

### **2. Install Dependencies**
```bash
cd evo-agent
pip install -r requirements.txt
```

### **3. Configure Azure OpenAI**
Create a `.env` file in the `evo-agent` directory:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Optional: Cost Management
MAX_COST_PER_EXPERIMENT=20.0
MAX_REQUESTS_PER_MINUTE=60
```

### **4. Run the Guided Agent**
```bash
python run_guided.py
```

## ⚙️ **Setup & Configuration**

### **Prerequisites**
- Python 3.8+
- Azure OpenAI service account
- Git

### **Environment Setup**

1. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   cd evo-agent
   pip install -r requirements.txt
   ```

3. **Configure Azure OpenAI**
   - Get your Azure OpenAI API key from Azure Portal
   - Update the `.env` file with your credentials
   - Ensure your deployment is active and has sufficient quota

### **Configuration Files**

#### **Environment Variables (.env)**
```bash
# Required: Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Optional: Cost Management
MAX_COST_PER_EXPERIMENT=20.0
MAX_REQUESTS_PER_MINUTE=60
MAX_COST_PER_HOUR=10.0

# Optional: Experiment Tracking
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your_wandb_key

# Optional: Logging
LOG_LEVEL=INFO
```

#### **Agent Configuration (guided_agent.py)**
```python
@dataclass
class AgentConfig:
    max_cost: float = 20.0              # Maximum cost per experiment
    evolution_frequency: int = 2         # Evolve agent every N generations
    population_size: int = 3             # Population size for evolution
    max_generations: int = 5             # Maximum evolution cycles
```

## 🎯 **How to Use**

### **1. Guided Agent (Recommended for Beginners)**

The `run_guided.py` file provides an interactive, step-by-step experience:

```bash
cd evo-agent
python run_guided.py
```

**What it does:**
- 🎯 **Task Setup**: Define your programming task
- 🔍 **Task Analysis**: Break down requirements
- 💻 **Code Generation**: Create initial Python code
- 🔄 **Evolution Cycles**: Improve code through multiple iterations
- 🧠 **Agent Evolution**: The agent itself evolves its capabilities
- ✅ **Final Execution**: Run and test the final solution

**Example Session:**
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

### **2. Advanced Agent (For Experienced Users)**

For more control and advanced features:

```bash
python run_agent.py
```

**Features:**
- Multi-objective optimization
- Cost management
- Experiment tracking
- Human-in-the-loop review
- Scalable diversity management

### **3. Interactive Demo**

For a quick demonstration:

```bash
python interactive_demo.py
```

### **4. Quick Improved Demo**

For a streamlined experience:

```bash
python quick_improved_demo.py
```

## 🔧 **Azure OpenAI Configuration**

### **Current Configuration**

The system is configured to use Azure OpenAI with these default settings:

```python
# From llm_interface.py
@dataclass
class LLMConfig:
    model: str = "o4-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    # Azure OpenAI settings
    azure_endpoint: str = "https://vinod-m7y6fqof-eastus2.cognitiveservices.azure.com/"
    api_version: str = "2024-12-01-preview"
    deployment_name: str = "o4-mini"
```

### **Updating Azure Configuration**

#### **Method 1: Environment Variables (Recommended)**
```bash
# In your .env file
AZURE_OPENAI_API_KEY=your_new_api_key
AZURE_OPENAI_ENDPOINT=https://your-new-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

#### **Method 2: Direct Code Modification**
Edit `evo-agent/llm_interface.py`:
```python
@dataclass
class LLMConfig:
    # Update these values
    azure_endpoint: str = "https://your-resource.cognitiveservices.azure.com/"
    deployment_name: str = "your_deployment_name"
    # ... other settings
```

### **Azure OpenAI Setup Steps**

1. **Create Azure OpenAI Resource**
   - Go to Azure Portal
   - Create a new "Azure OpenAI" resource
   - Note the endpoint URL

2. **Deploy a Model**
   - In your Azure OpenAI resource
   - Go to "Model deployments"
   - Deploy a model (e.g., GPT-4, o4-mini)
   - Note the deployment name

3. **Get API Key**
   - In your Azure OpenAI resource
   - Go to "Keys and Endpoint"
   - Copy Key 1 or Key 2

4. **Update Configuration**
   - Update your `.env` file with the new values
   - Test the connection

### **Testing Azure Connection**

```bash
cd evo-agent
python test_llm.py
```

This will test your Azure OpenAI connection and show any configuration issues.

## 📚 **Examples & Demos**

### **Available Demo Files**

| File | Purpose | Best For |
|------|---------|----------|
| `run_guided.py` | Interactive guided experience | Beginners |
| `run_agent.py` | Advanced agent with full features | Experienced users |
| `interactive_demo.py` | Quick demonstration | Quick overview |
| `quick_improved_demo.py` | Streamlined experience | Fast testing |
| `advanced_example.py` | Complete feature demonstration | Learning advanced features |

### **Example Tasks You Can Try**

1. **Simple Functions**
   ```
   Task: Create a function to reverse a string
   Requirements: Handle empty strings, special characters
   ```

2. **Data Processing**
   ```
   Task: Create a CSV parser
   Requirements: Handle headers, data validation, error handling
   ```

3. **Web Scraping**
   ```
   Task: Create a web scraper
   Requirements: Use requests, parse HTML, extract data
   ```

4. **API Integration**
   ```
   Task: Create a weather API client
   Requirements: Handle API responses, error handling, data formatting
   ```

## 🏗️ **Architecture**

### **Core Components**

```
evo-agent/
├── 🧠 Core Components
│   ├── evolutionary_agent.py      # Main orchestrator
│   ├── guided_agent.py           # Interactive guided agent
│   ├── llm_interface.py          # Azure OpenAI interface
│   ├── cost_manager.py           # Budget management
│   └── evaluation_framework.py   # Code evaluation
│
├── 🔬 Advanced Features
│   ├── multi_objective.py        # Multi-objective optimization
│   ├── experiment_tracker.py     # Experiment tracking
│   ├── scalable_diversity.py    # Diversity management
│   └── human_interface.py       # Human-in-the-loop
│
├── 🛠️ Support Components
│   ├── patch_manager.py          # Code patching
│   ├── diversity_manager.py      # Diversity preservation
│   └── metrics_exporter.py      # Metrics export
│
└── 📚 Examples & Documentation
    ├── run_guided.py            # Main entry point
    ├── example_usage.py         # Usage examples
    └── README.md               # Documentation
```

### **Key Features**

- ✅ **Multi-Objective Optimization** - Pareto fronts and NSGA-II selection
- ✅ **Cost Management** - Budget-aware LLM usage with rate limiting
- ✅ **Human-in-the-Loop** - Interactive review and feedback
- ✅ **Experiment Tracking** - MLflow and W&B integration
- ✅ **Scalable Diversity** - Incremental clustering and novelty archives
- ✅ **Extensible Artifacts** - Support for Python, SQL, JSON, YAML, Markdown

## 🧪 **Testing**

### **Run All Tests**
```bash
cd evo-agent
pytest test_evolutionary_agent.py -v --cov=. --cov-report=html
```

### **Test Specific Components**
```bash
# Test LLM interface
python test_llm.py

# Test cost management
pytest test_enhanced_agent.py::TestCostManagement -v

# Test diversity management
pytest test_evolutionary_agent.py::TestDiversityManagement -v
```

### **Test Azure Connection**
```bash
python test_llm.py
```

## 📊 **Monitoring**

### **Cost Tracking**
The system automatically tracks:
- Total cost per experiment
- Requests per minute
- Token usage
- Budget alerts

### **Performance Metrics**
- Evolution generations
- Best fitness scores
- Success rates
- Human review statistics

### **Experiment Tracking**
- MLflow integration for experiment tracking
- Weights & Biases for real-time monitoring
- Comprehensive logging and error tracking

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
   - Verify Azure credentials in `.env`
   - Test with `python test_llm.py`

4. **High costs**
   - Reduce `max_cost_per_experiment`
   - Use smaller population sizes
   - Limit evolution generations

### **Getting Help**

- Run `python test_llm.py` to test Azure connection
- Check logs for detailed error messages
- Use `show_agent_status` to see current state
- Review cost statistics regularly

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**The Evolutionary Agent System** represents a significant step toward true AlphaEvolve-grade self-improving agents, with robust, reproducible, cost-aware, and extensible capabilities that can handle complex, long-running evolutionary experiments with full observability and human oversight! 🚀

For more detailed information, see the [evo-agent/README.md](evo-agent/README.md) file.
