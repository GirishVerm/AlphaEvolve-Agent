#!/usr/bin/env python3
"""
Simple Evolutionary Example - Demonstrates the system working with proper initial content.
"""
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

from llm_interface import LLMInterface, LLMConfig
from artifact_support import ArtifactType, ArtifactCandidate
from multi_objective import Objective, MultiObjectiveConfig, MultiObjectiveEvaluator
from cost_manager import CostConfig, BudgetAwareLLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleEvolutionConfig:
    """Simple evolution configuration."""
    population_size: int = 5
    generations: int = 3
    max_cost: float = 5.0

class SimpleEvolutionaryAgent:
    """Simple evolutionary agent for demonstration."""
    
    def __init__(self, config: SimpleEvolutionConfig):
        """Initialize the agent."""
        self.config = config
        self.population = []
        self.generation = 0
        
        # Initialize LLM
        llm_config = LLMConfig()
        base_llm = LLMInterface(llm_config)
        
        # Wrap with cost management
        cost_config = CostConfig(max_cost_per_experiment=config.max_cost)
        self.llm = BudgetAwareLLMInterface(base_llm, cost_config)
        
        # Initialize evaluation
        objectives = [
            Objective(name="code_quality", weight=0.6, minimize=False),
            Objective(name="functionality", weight=0.4, minimize=False)
        ]
        eval_config = MultiObjectiveConfig(objectives=objectives)
        self.evaluator = MultiObjectiveEvaluator(eval_config)
    
    async def initialize_population(self, initial_code: str):
        """Initialize population with baseline."""
        logger.info("Initializing population...")
        
        # Create baseline candidate
        baseline = ArtifactCandidate(
            id="baseline",
            content=initial_code,
            artifact_type=ArtifactType.PYTHON_CODE,
            metadata={"generation": 0, "parent": "none"}
        )
        
        # Evaluate baseline
        fitness = await self.evaluate_candidate(baseline)
        baseline.fitness_scores = fitness
        baseline.fitness_score = sum(fitness) / len(fitness)
        
        self.population = [baseline]
        logger.info(f"Baseline fitness: {baseline.fitness_score:.3f}")
    
    async def evaluate_candidate(self, candidate: ArtifactCandidate) -> List[float]:
        """Evaluate a candidate."""
        # Simple evaluation based on code characteristics
        code = candidate.content
        
        # Code quality score (0-1)
        quality_score = 0.5  # Baseline
        if "def" in code and "return" in code:
            quality_score = 0.7
        if "docstring" in code or '"""' in code:
            quality_score = 0.8
        if "try" in code and "except" in code:
            quality_score = 0.9
        
        # Functionality score (0-1)
        functionality_score = 0.5  # Baseline
        if "add" in code.lower() or "sum" in code.lower():
            functionality_score = 0.8
        if "test" in code.lower() or "assert" in code.lower():
            functionality_score = 0.9
        
        return [quality_score, functionality_score]
    
    async def generate_mutation(self, parent: ArtifactCandidate) -> ArtifactCandidate:
        """Generate a mutation of the parent."""
        prompt = f"""
        Improve this Python function. Make it more robust, add error handling, and improve the code quality.
        
        Original function:
        {parent.content}
        
        Provide only the improved function code:
        """
        
        try:
            response = await self.llm.generate(prompt)
            
            # Create new candidate
            child = ArtifactCandidate(
                id=f"gen_{self.generation}_{len(self.population)}",
                content=response,
                artifact_type=ArtifactType.PYTHON_CODE,
                metadata={"generation": self.generation, "parent": parent.id}
            )
            
            return child
            
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            # Return parent as fallback
            return parent
    
    async def evolve_generation(self):
        """Evolve one generation."""
        logger.info(f"Evolving generation {self.generation + 1}...")
        
        new_candidates = []
        
        # Generate mutations for each parent
        for parent in self.population:
            child = await self.generate_mutation(parent)
            
            # Evaluate child
            fitness = await self.evaluate_candidate(child)
            child.fitness_scores = fitness
            child.fitness_score = sum(fitness) / len(fitness)
            
            new_candidates.append(child)
            logger.info(f"Child fitness: {child.fitness_score:.3f}")
        
        # Select best candidates for next generation
        all_candidates = self.population + new_candidates
        all_candidates.sort(key=lambda x: x.fitness_score, reverse=True)
        
        self.population = all_candidates[:self.config.population_size]
        self.generation += 1
        
        # Log best candidate
        best = self.population[0]
        logger.info(f"Best fitness: {best.fitness_score:.3f}")
        logger.info(f"Best code:\n{best.content}")
    
    async def run_evolution(self):
        """Run the complete evolution."""
        logger.info("Starting evolution...")
        
        for gen in range(self.config.generations):
            await self.evolve_generation()
            
            # Check budget
            if self.llm.is_budget_exceeded():
                logger.warning("Budget exceeded, stopping evolution")
                break
        
        # Return best candidate
        best = max(self.population, key=lambda x: x.fitness_score)
        logger.info(f"Evolution complete! Best fitness: {best.fitness_score:.3f}")
        return best

async def main():
    """Run the simple evolution example."""
    logger.info("=== Simple Evolutionary Agent Demo ===")
    
    # Initialize agent
    config = SimpleEvolutionConfig()
    agent = SimpleEvolutionaryAgent(config)
    
    # Initial code
    initial_code = """
def add_numbers(a, b):
    return a + b
"""
    
    # Initialize population
    await agent.initialize_population(initial_code)
    
    # Run evolution
    best_candidate = await agent.run_evolution()
    
    # Show results
    logger.info("\n=== Final Results ===")
    logger.info(f"Best candidate ID: {best_candidate.id}")
    logger.info(f"Best fitness: {best_candidate.fitness_score:.3f}")
    logger.info(f"Best code:\n{best_candidate.content}")
    
    # Show cost stats
    cost_stats = agent.llm.get_cost_stats()
    logger.info(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
    logger.info(f"Total requests: {cost_stats.get('total_requests', 0)}")

if __name__ == "__main__":
    asyncio.run(main()) 