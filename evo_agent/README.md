# 🚀 **Advanced Evolutionary Agent Environment**

A sophisticated evolutionary agent system with **AlphaEvolve-grade** capabilities, featuring multi-objective optimization, reproducible experiments, cost management, scalable diversity, human-in-the-loop interfaces, and extensible artifact support.

## 🏗️ **Architecture Overview**

The system is built with a **modular, production-ready architecture** that supports:

- ✅ **Multi-Objective Optimization** - Pareto fronts and NSGA-II selection
- ✅ **Reproducible Experiments** - Full ML-ops integration (MLflow, W&B)
- ✅ **Cost & Rate-Limit Safeguards** - Budget-aware LLM usage
- ✅ **Scalable Diversity Management** - Incremental clustering and novelty archives
- ✅ **Pluggable Human Interfaces** - CLI, web callbacks, message queues
- ✅ **Extensible Artifact Support** - Python, SQL, JSON, YAML, Markdown, etc.

## 📁 **File Structure**

```
evo-agent/
├── 🧠 Core Components
│   ├── evolutionary_agent.py      # Main orchestrator
│   ├── llm_interface.py          # Lightweight LLM interface
│   ├── diff_generator.py         # Code mutation generation
│   ├── population_manager.py     # Population operations
│   ├── prompt_manager.py         # Template management
│   └── metrics_exporter.py       # Observability
│
├── 🔬 Advanced Features
│   ├── multi_objective.py        # Pareto fronts & multi-objective optimization
│   ├── experiment_tracker.py     # Reproducible experiments & ML-ops
│   ├── cost_manager.py          # Budget & rate-limit management
│   ├── scalable_diversity.py    # Incremental clustering & novelty archives
│   ├── human_interface.py       # Pluggable human-in-the-loop
│   └── artifact_support.py      # Extensible artifact evolution
│
├── 🛠️ Support Components
│   ├── patch_manager.py          # Robust patch application
│   ├── diversity_manager.py      # Diversity preservation
│   ├── evaluation_framework.py   # Evaluation system
│   └── test_evolutionary_agent.py # Comprehensive tests
│
├── 📚 Examples & Documentation
│   ├── advanced_example.py       # Complete feature demonstration
│   ├── example_usage.py          # Basic usage examples
│   ├── main.py                  # CLI interface
│   └── README.md               # This documentation
│
└── 📦 Configuration
    ├── requirements.txt          # Dependencies
    └── config.yaml              # Configuration files
```

## 🚀 **Advanced Features**

### **1. Multi-Objective & Pareto Fronts**

**Purpose**: Handle multiple competing objectives without collapsing trade-offs early.

```python
from multi_objective import MultiObjectiveConfig, Objective, SelectionMode

# Define multiple objectives
objectives = [
    Objective(name="code_quality", weight=0.5, minimize=False),
    Objective(name="novelty", weight=0.3, minimize=False),
    Objective(name="complexity", weight=0.2, minimize=False)
]

# Configure multi-objective optimization
config = MultiObjectiveConfig(
    objectives=objectives,
    selection_mode=SelectionMode.PARETO,
    pareto_epsilon=0.01
)

# Use in evolution
evaluator = MultiObjectiveEvaluator(config)
fitness_vector = evaluator.evaluate_candidate(candidate, evaluation_functions)
```

**Key Features**:
- ✅ **Pareto Dominance** - True multi-objective selection
- ✅ **NSGA-II Algorithm** - Advanced multi-objective optimization
- ✅ **Crowding Distance** - Maintains diversity in Pareto front
- ✅ **Hypervolume Calculation** - Measures Pareto front quality
- ✅ **Visualization** - Plot Pareto fronts (2D/3D)

### **2. Reproducible Experiments & ML-ops**

**Purpose**: Full reproducibility with MLflow and Weights & Biases integration.

```python
from experiment_tracker import ExperimentConfig, ExperimentTracker

# Configure experiment tracking
config = ExperimentConfig(
    experiment_name="my_evolution_experiment",
    seed=42,
    mlflow_enabled=True,
    wandb_enabled=True
)

# Initialize tracker
tracker = ExperimentTracker(config)

# Log metrics
tracker.log_metrics({
    'generation': 5,
    'best_fitness': 0.85,
    'population_size': 50
}, step=5)

# Create checkpoints
tracker.create_checkpoint(state, generation=10)

# Log artifacts
tracker.log_artifact("pareto_front.png", "pareto_front_gen_10")
```

**Key Features**:
- ✅ **Deterministic Seeds** - Full reproducibility across runs
- ✅ **MLflow Integration** - Experiment tracking and model registry
- ✅ **Weights & Biases** - Real-time experiment monitoring
- ✅ **Checkpointing** - Resume experiments from any point
- ✅ **Artifact Logging** - Save plots, models, and results
- ✅ **Error Tracking** - Comprehensive error logging

### **3. Cost & Rate-Limit Safeguards**

**Purpose**: Prevent runaway API costs and handle rate limits gracefully.

```python
from cost_manager import CostConfig, BudgetAwareLLMInterface

# Configure cost management
cost_config = CostConfig(
    max_cost_per_experiment=50.0,  # USD
    max_requests_per_minute=60,
    max_cost_per_hour=10.0,
    token_cost_per_1k=0.03  # GPT-4 pricing
)

# Wrap LLM interface with budget awareness
budget_llm = BudgetAwareLLMInterface(base_llm, cost_config)

# Automatic budget checking
response = await budget_llm.generate(prompt)

# Get cost statistics
stats = budget_llm.get_cost_stats()
alerts = budget_llm.get_alerts()
```

**Key Features**:
- ✅ **Token Tracking** - Real-time token usage monitoring
- ✅ **Cost Limits** - Per-experiment and per-hour budgets
- ✅ **Rate Limiting** - Automatic throttling and backoff
- ✅ **Adaptive Throttling** - Adjusts based on error rates
- ✅ **Budget Alerts** - Early warnings at 80% thresholds
- ✅ **Exponential Backoff** - Handles rate limit errors

### **4. Scalable Diversity Management**

**Purpose**: Handle large populations efficiently with incremental clustering.

```python
from scalable_diversity import ScalableDiversityConfig, ScalableDiversityManager

# Configure scalable diversity
config = ScalableDiversityConfig(
    max_candidates=1000,
    novelty_threshold=0.8,
    incremental_clustering=True,
    birch_threshold=0.5
)

# Initialize diversity manager
diversity_manager = ScalableDiversityManager(config)

# Add candidates to novelty archive
diversity_manager.add_candidates(candidates)

# Select diverse candidates
diverse_selection = diversity_manager.select_diverse_candidates(
    candidates, target_size=50
)

# Get diversity metrics
metrics = diversity_manager.get_diversity_metrics()
```

**Key Features**:
- ✅ **Incremental Clustering** - BIRCH algorithm for large populations
- ✅ **Novelty Archives** - Maintains diverse solution history
- ✅ **Caching Strategy** - Reduces clustering overhead
- ✅ **Memory Management** - Automatic cache clearing
- ✅ **Diversity Metrics** - Hypervolume, spread, entropy
- ✅ **Scalable Selection** - O(N log N) instead of O(N²)

### **5. Pluggable Human-in-the-Loop**

**Purpose**: Integrate human review at any point in the evolution process.

```python
from human_interface import HumanInterfaceConfig, InterfaceType

# Configure human interface
config = HumanInterfaceConfig(
    interface_type=InterfaceType.CLI,  # or WEB_CALLBACK, MESSAGE_QUEUE
    timeout_seconds=300,
    batch_size=10
)

# Create interface
interface = HumanInterfaceFactory.create_interface(config)
review_manager = AsyncHumanReviewManager(interface)

# Request human review
response = await review_manager.request_review(candidate, context)

if response and response.approved:
    print("Human approved the candidate!")
```

**Key Features**:
- ✅ **Multiple Interfaces** - CLI, web callbacks, message queues
- ✅ **Batch Processing** - Group multiple reviews together
- ✅ **Asynchronous Operation** - Non-blocking human review
- ✅ **Timeout Handling** - Graceful timeout with fallbacks
- ✅ **Priority System** - Urgent vs. normal review requests
- ✅ **Feedback Collection** - Structured human feedback

### **6. Extensible Artifact Support**

**Purpose**: Evolve any type of text artifact, not just Python code.

```python
from artifact_support import ArtifactType, ArtifactManager

# Initialize artifact manager
manager = ArtifactManager()

# Create candidates for different artifact types
python_candidate = manager.create_candidate(
    python_code, 
    ArtifactType.PYTHON_CODE
)

sql_candidate = manager.create_candidate(
    sql_query, 
    ArtifactType.SQL_QUERY
)

markdown_candidate = manager.create_candidate(
    markdown_content, 
    ArtifactType.MARKDOWN
)

# Apply mutations
diff_blocks = [DiffBlock(0, 5, old_text, new_text, ArtifactType.PYTHON_CODE)]
mutated = manager.apply_mutation(candidate, diff_blocks)
```

**Supported Artifact Types**:
- ✅ **Python Code** - Full AST validation and parsing
- ✅ **SQL Queries** - Syntax and structure validation
- ✅ **JSON Schemas** - Schema validation and quality assessment
- ✅ **YAML Configs** - Configuration file evolution
- ✅ **Markdown** - Documentation and content evolution
- ✅ **Generic Text** - Any text-based artifact

## 🎯 **Usage Examples**

### **Basic Multi-Objective Evolution**

```python
from advanced_example import AdvancedEvolutionaryAgent, AdvancedConfig
from multi_objective import Objective

# Define objectives
objectives = [
    Objective(name="code_quality", weight=0.5),
    Objective(name="novelty", weight=0.3),
    Objective(name="complexity", weight=0.2)
]

# Configure agent
config = AdvancedConfig(
    objectives=objectives,
    experiment_name="my_experiment",
    max_cost_per_experiment=25.0,
    population_size=30
)

# Create agent
agent = AdvancedEvolutionaryAgent(config)

# Initialize with baseline
initial_code = "def hello(): return 'world'"
await agent.initialize_population(initial_code, ArtifactType.PYTHON_CODE)

# Run evolution
best_candidate = await agent.run_evolution(max_generations=50)

print(f"Best fitness: {best_candidate.fitness_score}")
print(f"Best code:\n{best_candidate.content}")
```

### **Cost-Aware Evolution with Human Review**

```python
# Configure with cost limits and human review
config = AdvancedConfig(
    max_cost_per_experiment=20.0,
    human_interface_type=InterfaceType.CLI,
    human_timeout=60
)

agent = AdvancedEvolutionaryAgent(config)

# Run with human review every 10 generations
for generation in range(50):
    await agent.evolve_population()
    
    if generation % 10 == 0:
        # Request human review of best candidate
        approved = await agent.request_human_review(agent.best_candidate)
        if not approved:
            print("Human rejected best candidate, continuing...")
    
    # Check budget
    if agent.llm_interface.is_budget_exceeded():
        print("Budget exceeded, stopping evolution")
        break
```

### **Reproducible Experiment with ML-ops**

```python
# Configure with full ML-ops integration
config = AdvancedConfig(
    experiment_name="reproducible_study",
    seed=42,
    mlflow_enabled=True,
    wandb_enabled=True
)

agent = AdvancedEvolutionaryAgent(config)

# Run experiment
best_candidate = await agent.run_evolution(max_generations=100)

# Get comprehensive summary
summary = agent.get_experiment_summary()
print(f"Experiment ID: {summary['experiment_id']}")
print(f"Total cost: ${summary['total_cost']:.2f}")
print(f"Human reviews: {summary['human_reviews']}")
```

## 📊 **Performance Optimizations**

### **Caching Strategy**
- **TF-IDF Vectorizer Caching** - Reuse vectorizer across generations
- **Feature Cache** - Cache extracted features for diversity calculation
- **Similarity Cache** - Cache similarity calculations for novelty scoring
- **Template Cache** - Cache frequently used prompt templates

### **Parallel Processing**
- **CPU-bound Tasks** - Use ProcessPoolExecutor for evaluation and clustering
- **I/O-bound Tasks** - Use ThreadPoolExecutor for LLM calls
- **Batch Operations** - Process multiple candidates simultaneously

### **Memory Management**
- **Population Pruning** - Remove low-fitness candidates automatically
- **Cache Clearing** - Periodic cache cleanup to prevent memory leaks
- **Archive Management** - Maintain novelty archive within size limits

## 🔧 **Configuration**

### **Environment Variables**
```bash
export OPENAI_API_KEY="your-api-key"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export WANDB_API_KEY="your-wandb-key"
```

### **Configuration File (config.yaml)**
```yaml
evolution:
  population_size: 50
  elite_size: 5
  max_generations: 100
  
multi_objective:
  objectives:
    - name: code_quality
      weight: 0.5
      minimize: false
    - name: novelty
      weight: 0.3
      minimize: false
    - name: complexity
      weight: 0.2
      minimize: false
  selection_mode: pareto

cost_management:
  max_cost_per_experiment: 50.0
  max_requests_per_minute: 60
  token_cost_per_1k: 0.03

experiment_tracking:
  mlflow_enabled: true
  wandb_enabled: false
  checkpoint_interval: 10

diversity:
  max_candidates: 1000
  novelty_threshold: 0.8
  incremental_clustering: true

human_interface:
  type: cli
  timeout_seconds: 300
  batch_size: 10
```

## 🧪 **Testing**

### **Run All Tests**
```bash
cd evo-agent
pytest test_evolutionary_agent.py -v --cov=. --cov-report=html
```

### **Test Specific Features**
```bash
# Test multi-objective optimization
pytest test_evolutionary_agent.py::TestMultiObjective -v

# Test cost management
pytest test_evolutionary_agent.py::TestCostManagement -v

# Test diversity management
pytest test_evolutionary_agent.py::TestDiversityManagement -v
```

## 📈 **Monitoring & Observability**

### **Metrics Dashboard**
```python
# Get comprehensive metrics
metrics = agent.metrics_exporter.get_metrics_summary()

print(f"Runtime: {metrics['runtime_hours']:.2f} hours")
print(f"Total generations: {metrics['total_generations']}")
print(f"Best fitness: {metrics['best_fitness']:.3f}")
print(f"LLM requests: {metrics['llm_requests']}")
print(f"Success rate: {metrics['llm_success_rate']:.2%}")
```

### **Prometheus Integration**
```python
# Export metrics to Prometheus
agent.metrics_exporter.export_metrics(format='prometheus')
```

## 🚀 **Production Deployment**

### **Docker Setup**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### **Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evolutionary-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: evolutionary-agent
  template:
    metadata:
      labels:
        app: evolutionary-agent
    spec:
      containers:
      - name: evolutionary-agent
        image: evolutionary-agent:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

## 🎯 **Roadmap to Full AlphaEvolve Fidelity**

### **Phase 1 (Current)** ✅
- ✅ Modular architecture with clean separation
- ✅ Multi-objective optimization with Pareto fronts
- ✅ Reproducible experiments with ML-ops integration
- ✅ Cost-aware LLM usage with rate limiting
- ✅ Scalable diversity management
- ✅ Pluggable human-in-the-loop interfaces
- ✅ Extensible artifact support

### **Phase 2 (Next)**
- 🔄 Advanced mutation strategies with adaptive scheduling
- 🔄 Enhanced ML-ops integration with custom metrics
- 🔄 Advanced budget optimization algorithms
- 🔄 Real-time human review dashboards
- 🔄 Advanced artifact type support (Docker, Kubernetes, etc.)

### **Phase 3 (Future)**
- 🔄 Novelty archives with advanced similarity metrics
- 🔄 Non-blocking human review with web UI
- 🔄 Generalized evolution for arbitrary text artifacts
- 🔄 Advanced convergence detection and optimization
- 🔄 Distributed evolution across multiple agents

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**The Advanced Evolutionary Agent Environment** represents a significant step toward true AlphaEvolve-grade self-improving agents, with robust, reproducible, cost-aware, and extensible capabilities that can handle complex, long-running evolutionary experiments with full observability and human oversight! 🚀 