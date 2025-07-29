#!/usr/bin/env python3
"""
Advanced Example: Demonstrating All New Features.
"""
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# Import all the new advanced modules
from multi_objective import (
    MultiObjectiveConfig, MultiObjectiveEvaluator, 
    Objective, SelectionMode, AdaptiveMutationScheduler
)
from experiment_tracker import (
    ExperimentConfig, ExperimentTracker, 
    ReproducibilityManager, ExperimentRegistry
)
from cost_manager import CostConfig, BudgetAwareLLMInterface
from scalable_diversity import ScalableDiversityConfig, ScalableDiversityManager
from human_interface import (
    HumanInterfaceConfig, InterfaceType, 
    HumanInterfaceFactory, AsyncHumanReviewManager
)
from artifact_support import (
    ArtifactType, ArtifactManager, 
    ArtifactCandidate, DiffBlock
)
from llm_interface import LLMInterface, LLMConfig

# Import existing modules
from evolutionary_agent import EvolutionaryAgent
from diff_generator import DiffGenerator
from population_manager import PopulationManager, PopulationConfig
from prompt_manager import PromptManager
from metrics_exporter import MetricsExporter

logger = logging.getLogger(__name__)


@dataclass
class AdvancedConfig:
    """Configuration for advanced evolutionary agent."""
    # Multi-objective configuration
    objectives: List[Objective]
    selection_mode: SelectionMode = SelectionMode.PARETO
    
    # Experiment tracking
    experiment_name: str = "advanced_evolution"
    seed: int = 42
    mlflow_enabled: bool = False
    wandb_enabled: bool = False
    
    # Cost management
    max_cost_per_experiment: float = 50.0
    max_requests_per_minute: int = 60
    
    # Scalable diversity
    max_candidates: int = 1000
    novelty_threshold: float = 0.8
    
    # Human interface
    human_interface_type: InterfaceType = InterfaceType.CLI
    human_timeout: int = 300
    
    # Population
    population_size: int = 50
    elite_size: int = 5


class AdvancedEvolutionaryAgent:
    """
    Advanced evolutionary agent with all new features.
    """
    
    def __init__(self, config: AdvancedConfig):
        """
        Initialize advanced evolutionary agent.
        
        Args:
            config: Advanced configuration
        """
        self.config = config
        
        # Initialize reproducibility
        self.reproducibility_manager = ReproducibilityManager(base_seed=config.seed)
        self.reproducibility_manager.set_random_seeds(config.seed)
        
        # Initialize experiment tracking
        experiment_config = ExperimentConfig(
            experiment_name=config.experiment_name,
            seed=config.seed,
            mlflow_enabled=config.mlflow_enabled,
            wandb_enabled=config.wandb_enabled
        )
        self.experiment_tracker = ExperimentTracker(experiment_config)
        
        # Initialize multi-objective evaluation
        multi_obj_config = MultiObjectiveConfig(
            objectives=config.objectives,
            selection_mode=config.selection_mode
        )
        self.multi_objective_evaluator = MultiObjectiveEvaluator(multi_obj_config)
        
        # Initialize cost management
        cost_config = CostConfig(
            max_cost_per_experiment=config.max_cost_per_experiment,
            max_requests_per_minute=config.max_requests_per_minute
        )
        
        # Initialize LLM interface with budget awareness
        llm_config = LLMConfig()
        base_llm_interface = LLMInterface(llm_config)
        self.llm_interface = BudgetAwareLLMInterface(base_llm_interface, cost_config)
        
        # Initialize scalable diversity
        diversity_config = ScalableDiversityConfig(
            max_candidates=config.max_candidates,
            novelty_threshold=config.novelty_threshold
        )
        self.diversity_manager = ScalableDiversityManager(diversity_config)
        
        # Initialize human interface
        human_config = HumanInterfaceConfig(
            interface_type=config.human_interface_type,
            timeout_seconds=config.human_timeout
        )
        self.human_interface = HumanInterfaceFactory.create_interface(human_config)
        self.human_review_manager = AsyncHumanReviewManager(self.human_interface)
        
        # Initialize artifact manager
        self.artifact_manager = ArtifactManager()
        
        # Initialize adaptive mutation scheduler
        self.mutation_scheduler = AdaptiveMutationScheduler()
        
        # Initialize existing components
        self.diff_generator = DiffGenerator()
        self.population_manager = PopulationManager(PopulationConfig(
            population_size=config.population_size,
            elite_size=config.elite_size
        ))
        self.prompt_manager = PromptManager()
        self.metrics_exporter = MetricsExporter()
        
        # State
        self.current_generation = 0
        self.best_candidate = None
        self.population: List[ArtifactCandidate] = []
    
    async def initialize_population(self, initial_content: str, artifact_type: ArtifactType) -> None:
        """
        Initialize population with baseline candidate.
        
        Args:
            initial_content: Initial artifact content
            artifact_type: Type of artifact
        """
        # Create baseline candidate
        baseline = self.artifact_manager.create_candidate(
            initial_content, 
            artifact_type,
            "baseline"
        )
        
        # Initialize population
        self.population = [baseline]
        self.best_candidate = baseline
        
        # Log initial state
        self.experiment_tracker.log_metrics({
            'generation': 0,
            'population_size': len(self.population),
            'best_fitness': baseline.fitness_score,
            'artifact_type': artifact_type.value
        }, step=0)
        
        logger.info(f"Initialized population with baseline candidate (fitness: {baseline.fitness_score:.3f})")
    
    async def evaluate_candidate_multi_objective(self, candidate: ArtifactCandidate) -> List[float]:
        """
        Evaluate candidate using multiple objectives.
        
        Args:
            candidate: Candidate to evaluate
            
        Returns:
            List of objective values
        """
        # Define evaluation functions for each objective
        evaluation_functions = []
        
        for objective in self.config.objectives:
            if objective.name == "code_quality":
                evaluation_functions.append(lambda c: c.fitness_score)
            elif objective.name == "novelty":
                evaluation_functions.append(lambda c: self.diversity_manager.calculate_novelty_score(c, self.population))
            elif objective.name == "complexity":
                evaluation_functions.append(lambda c: min(1.0, len(c.content.splitlines()) / 100.0))
            else:
                # Default to fitness score
                evaluation_functions.append(lambda c: c.fitness_score)
        
        # Evaluate candidate
        fitness_vector = self.multi_objective_evaluator.evaluate_candidate(
            candidate, 
            evaluation_functions
        )
        
        return fitness_vector.tolist()
    
    async def generate_mutations(self, parent: ArtifactCandidate) -> List[ArtifactCandidate]:
        """
        Generate mutations for a parent candidate.
        
        Args:
            parent: Parent candidate
            
        Returns:
            List of mutated candidates
        """
        mutations = []
        
        # Get adaptive mutation rate
        mutation_rate = self.mutation_scheduler.update_rate(parent.fitness_score)
        patch_size_factor = self.mutation_scheduler.get_patch_size_factor()
        
        # Generate mutations using LLM
        try:
            # Create prompt for mutation
            prompt = f"""
            Generate a mutation for the following {parent.artifact_type.value}:
            
            Original:
            {parent.content}
            
            Task: Improve the code quality, maintainability, and functionality.
            Focus on small, surgical changes that enhance the code.
            """
            
            # Generate mutation
            response = await self.llm_interface.generate(prompt)
            
            # Parse diff blocks
            diff_blocks = self.diff_generator.parse_llm_response(response)
            
            # Apply mutation
            if diff_blocks:
                mutated_candidate = self.artifact_manager.apply_mutation(parent, diff_blocks)
                if mutated_candidate.id != parent.id:  # Mutation was successful
                    mutated_candidate.generation = self.current_generation + 1
                    mutations.append(mutated_candidate)
            
        except Exception as e:
            logger.error(f"Error generating mutation: {e}")
        
        return mutations
    
    async def select_parents(self) -> List[ArtifactCandidate]:
        """
        Select parents using multi-objective selection.
        
        Returns:
            List of selected parents
        """
        if len(self.population) < 2:
            return self.population
        
        # Use multi-objective selection
        selected = self.multi_objective_evaluator.select_candidates(
            self.population,
            num_select=min(5, len(self.population))
        )
        
        return selected
    
    async def evolve_population(self) -> None:
        """Evolve the population for one generation."""
        self.current_generation += 1
        
        # Select parents
        parents = await self.select_parents()
        
        # Generate offspring
        offspring = []
        for parent in parents:
            mutations = await self.generate_mutations(parent)
            offspring.extend(mutations)
        
        # Add to diversity management
        self.diversity_manager.add_candidates(offspring)
        
        # Select diverse candidates for next generation
        next_population = self.diversity_manager.select_diverse_candidates(
            offspring + self.population,
            target_size=self.config.population_size
        )
        
        # Evaluate new candidates
        for candidate in next_population:
            if candidate.generation == self.current_generation:
                # Evaluate using multi-objective criteria
                fitness_vector = await self.evaluate_candidate_multi_objective(candidate)
                # Use first objective as primary fitness
                candidate.fitness_score = fitness_vector[0] if fitness_vector else 0.0
        
        # Update population
        self.population = next_population
        
        # Update best candidate
        best_in_gen = max(self.population, key=lambda x: x.fitness_score)
        if self.best_candidate is None or best_in_gen.fitness_score > self.best_candidate.fitness_score:
            self.best_candidate = best_in_gen
        
        # Log metrics
        self._log_generation_metrics()
        
        # Create checkpoint
        if self.current_generation % 10 == 0:
            self._create_checkpoint()
    
    def _log_generation_metrics(self) -> None:
        """Log metrics for current generation."""
        # Population metrics
        fitness_scores = [c.fitness_score for c in self.population]
        diversity_metrics = self.diversity_manager.get_diversity_metrics()
        cost_stats = self.llm_interface.get_cost_stats()
        
        metrics = {
            'generation': self.current_generation,
            'population_size': len(self.population),
            'best_fitness': max(fitness_scores),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'min_fitness': min(fitness_scores),
            'mutation_rate': self.mutation_scheduler.current_rate,
            **diversity_metrics,
            **cost_stats
        }
        
        self.experiment_tracker.log_population_stats(metrics, self.current_generation)
        self.metrics_exporter.update_population_metrics(metrics)
        
        # Log alerts
        alerts = self.llm_interface.get_alerts()
        if alerts:
            for alert in alerts:
                logger.warning(alert)
    
    def _create_checkpoint(self) -> None:
        """Create experiment checkpoint."""
        state = {
            'generation': self.current_generation,
            'population': [c.id for c in self.population],
            'best_candidate': self.best_candidate.id if self.best_candidate else None,
            'mutation_rate': self.mutation_scheduler.current_rate,
            'diversity_metrics': self.diversity_manager.get_diversity_metrics(),
            'cost_stats': self.llm_interface.get_cost_stats()
        }
        
        self.experiment_tracker.create_checkpoint(state, self.current_generation)
    
    async def request_human_review(self, candidate: ArtifactCandidate) -> bool:
        """
        Request human review for a candidate.
        
        Args:
            candidate: Candidate to review
            
        Returns:
            True if approved
        """
        context = {
            'generation': candidate.generation,
            'artifact_type': candidate.artifact_type.value,
            'fitness_score': candidate.fitness_score
        }
        
        response = await self.human_review_manager.request_review(candidate, context)
        
        if response and response.approved:
            logger.info(f"Human approved candidate {candidate.id}")
            return True
        else:
            logger.info(f"Human rejected candidate {candidate.id}")
            return False
    
    async def run_evolution(self, max_generations: int = 100) -> ArtifactCandidate:
        """
        Run evolutionary optimization.
        
        Args:
            max_generations: Maximum number of generations
            
        Returns:
            Best candidate found
        """
        logger.info(f"Starting evolution for {max_generations} generations")
        
        try:
            for generation in range(max_generations):
                logger.info(f"Generation {self.current_generation + 1}/{max_generations}")
                
                # Check budget
                if self.llm_interface.is_budget_exceeded():
                    logger.warning("Budget exceeded, stopping evolution")
                    break
                
                # Evolve population
                await self.evolve_population()
                
                # Optional human review for best candidate
                if generation % 20 == 0 and self.best_candidate:
                    approved = await self.request_human_review(self.best_candidate)
                    if not approved:
                        logger.info("Human rejected best candidate, continuing evolution")
                
                # Log Pareto front
                if generation % 10 == 0:
                    self.experiment_tracker.log_pareto_front(
                        self.multi_objective_evaluator.pareto_front,
                        generation
                    )
                
                # Check convergence
                if self._check_convergence():
                    logger.info("Convergence detected, stopping evolution")
                    break
            
            # Finish experiment
            self.experiment_tracker.finish()
            
            return self.best_candidate
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            self.experiment_tracker.log_error(e, "evolution_loop")
            raise
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.population) < 2:
            return False
        
        # Check if fitness improvement is minimal
        recent_fitness = [c.fitness_score for c in self.population[-5:]]
        if len(recent_fitness) >= 3:
            improvement = max(recent_fitness) - min(recent_fitness)
            if improvement < 0.01:
                return True
        
        return False
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Experiment summary
        """
        return {
            'experiment_id': self.experiment_tracker.experiment_id,
            'generations': self.current_generation,
            'best_fitness': self.best_candidate.fitness_score if self.best_candidate else 0.0,
            'final_population_size': len(self.population),
            'total_cost': self.llm_interface.get_cost_stats().get('total_cost', 0.0),
            'human_reviews': self.human_review_manager.get_review_stats(),
            'diversity_metrics': self.diversity_manager.get_diversity_metrics(),
            'experiment_summary': self.experiment_tracker.get_experiment_summary()
        }


async def main():
    """Main function demonstrating advanced features."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Define objectives for multi-objective optimization
    objectives = [
        Objective(name="code_quality", weight=0.5, minimize=False),
        Objective(name="novelty", weight=0.3, minimize=False),
        Objective(name="complexity", weight=0.2, minimize=False)
    ]
    
    # Create advanced configuration
    config = AdvancedConfig(
        objectives=objectives,
        experiment_name="advanced_evolution_demo",
        seed=42,
        max_cost_per_experiment=25.0,
        population_size=20,
        human_interface_type=InterfaceType.CLI
    )
    
    # Create advanced evolutionary agent
    agent = AdvancedEvolutionaryAgent(config)
    
    # Initialize with a sample Python function
    initial_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
    
    await agent.initialize_population(initial_code, ArtifactType.PYTHON_CODE)
    
    # Run evolution
    best_candidate = await agent.run_evolution(max_generations=30)
    
    # Print results
    print("\n" + "="*80)
    print("EVOLUTION COMPLETED")
    print("="*80)
    
    summary = agent.get_experiment_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nBest candidate fitness: {best_candidate.fitness_score:.3f}")
    print(f"Best candidate content:\n{best_candidate.content}")


if __name__ == "__main__":
    asyncio.run(main()) 