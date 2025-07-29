#!/usr/bin/env python3
"""
Main entry point for the Evolutionary Agent system.
"""
import os
import sys
import asyncio
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evolutionary_agent import EvolutionaryAgent, EvolutionConfig
from evaluation_framework import create_evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def run_evolutionary_task(
    agent_id: str,
    task_description: str,
    task_type: str = "default",
    config_path: str = "config.yaml",
    evolution_config: Optional[EvolutionConfig] = None
) -> Dict[str, Any]:
    """
    Run a complete evolutionary task.
    
    Args:
        agent_id: ID of the agent to use
        task_description: Description of the task
        task_type: Type of task for evaluation
        config_path: Path to configuration file
        evolution_config: Evolutionary algorithm configuration
        
    Returns:
        Results of the evolutionary process
    """
    logger.info(f"Starting evolutionary task: {task_description}")
    
    # Initialize evolutionary agent
    evo_agent = EvolutionaryAgent(agent_id, config_path, evolution_config)
    
    # Set the task
    evo_agent.set_task(task_description)
    
    # Run interactive planning (steps 1-4)
    logger.info("Phase 1: Interactive Planning")
    planning_results = await evo_agent.interactive_planning()
    
    # Set up evaluation function
    evaluator = create_evaluator(task_type)
    evo_agent.set_evaluation_function(evaluator)
    
    # Run evolutionary optimization (step 5)
    logger.info("Phase 2: Evolutionary Optimization")
    best_candidate = await evo_agent.evolutionary_optimization()
    
    # Get final statistics
    stats = evo_agent.get_evolution_stats()
    
    return {
        "planning_results": planning_results,
        "best_candidate": best_candidate,
        "evolution_stats": stats
    }


async def run_markdown_to_html_example():
    """Run the Markdown to HTML conversion example."""
    task_description = """
    Create a function that converts Markdown text to HTML.
    
    Requirements:
    - Function should be named 'convert'
    - Should handle basic Markdown syntax: headings, paragraphs, lists, links, bold text
    - Should be efficient (target: <50ms for medium-sized documents)
    - Should handle edge cases gracefully
    - Should pass all test cases
    
    The function should take a string input and return the HTML string.
    """
    
    evolution_config = EvolutionConfig(
        population_size=20,
        generations=30,
        mutation_rate=0.4,
        crossover_rate=0.6,
        elite_size=3,
        fitness_threshold=0.9
    )
    
    results = await run_evolutionary_task(
        agent_id="evolutionary_agent",
        task_description=task_description,
        task_type="markdown_to_html",
        evolution_config=evolution_config
    )
    
    return results


async def run_custom_task(
    task_description: str,
    task_type: str = "default",
    agent_id: str = "evolutionary_agent",
    config_path: str = "config.yaml",
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.7,
    elite_size: int = 5,
    fitness_threshold: float = 0.95
):
    """Run a custom evolutionary task."""
    
    evolution_config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_size=elite_size,
        fitness_threshold=fitness_threshold
    )
    
    results = await run_evolutionary_task(
        agent_id=agent_id,
        task_description=task_description,
        task_type=task_type,
        config_path=config_path,
        evolution_config=evolution_config
    )
    
    return results


def print_results(results: Dict[str, Any]):
    """Print the results of the evolutionary process."""
    print("\n" + "="*60)
    print("EVOLUTIONARY AGENT RESULTS")
    print("="*60)
    
    # Print planning results
    print("\n📋 PLANNING PHASE RESULTS:")
    planning = results.get("planning_results", {})
    for key, value in planning.items():
        print(f"  {key}: {len(str(value))} chars")
    
    # Print best candidate
    best_candidate = results.get("best_candidate")
    if best_candidate:
        print(f"\n🏆 BEST CANDIDATE:")
        print(f"  ID: {best_candidate.id}")
        print(f"  Generation: {best_candidate.generation}")
        print(f"  Fitness Score: {best_candidate.fitness_score:.4f}")
        print(f"  Parent: {best_candidate.parent_id}")
        print(f"  Mutation Type: {best_candidate.mutation_type}")
        
        print(f"\n  📝 CODE ({len(best_candidate.code)} chars):")
        print("  " + "-"*40)
        for line in best_candidate.code.split('\n')[:10]:
            print(f"  {line}")
        if len(best_candidate.code.split('\n')) > 10:
            print(f"  ... ({len(best_candidate.code.split('\n')) - 10} more lines)")
    
    # Print evolution statistics
    stats = results.get("evolution_stats", {})
    if stats:
        print(f"\n📊 EVOLUTION STATISTICS:")
        print(f"  Generation: {stats.get('generation', 'N/A')}")
        print(f"  Population Size: {stats.get('population_size', 'N/A')}")
        print(f"  Best Fitness: {stats.get('best_fitness', 'N/A'):.4f}")
        print(f"  Average Fitness: {stats.get('avg_fitness', 'N/A'):.4f}")
        print(f"  Fitness Std Dev: {stats.get('std_fitness', 'N/A'):.4f}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evolutionary Agent System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Example command
    example_parser = subparsers.add_parser("example", help="Run Markdown to HTML example")
    
    # Custom task command
    custom_parser = subparsers.add_parser("custom", help="Run custom evolutionary task")
    custom_parser.add_argument("--task", required=True, help="Task description")
    custom_parser.add_argument("--task-type", default="default", help="Task type for evaluation")
    custom_parser.add_argument("--agent-id", default="evolutionary_agent", help="Agent ID")
    custom_parser.add_argument("--config", default="config.yaml", help="Config file path")
    custom_parser.add_argument("--population", type=int, default=50, help="Population size")
    custom_parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    custom_parser.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation rate")
    custom_parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    custom_parser.add_argument("--elite-size", type=int, default=5, help="Elite size")
    custom_parser.add_argument("--fitness-threshold", type=float, default=0.95, help="Fitness threshold")
    
    args = parser.parse_args()
    
    if args.command == "example":
        logger.info("Running Markdown to HTML example")
        results = await run_markdown_to_html_example()
        print_results(results)
        
    elif args.command == "custom":
        logger.info(f"Running custom task: {args.task}")
        results = await run_custom_task(
            task_description=args.task,
            task_type=args.task_type,
            agent_id=args.agent_id,
            config_path=args.config,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_size=args.elite_size,
            fitness_threshold=args.fitness_threshold
        )
        print_results(results)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 