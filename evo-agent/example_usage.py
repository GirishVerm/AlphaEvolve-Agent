#!/usr/bin/env python3
"""
Example usage of the Evolutionary Agent with all improvements.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from evolutionary_agent import EvolutionaryAgent, EvolutionConfig
from evaluation_framework import create_evaluator
from patch_manager import PatchManager, RobustLLMParser
from diversity_manager import DiversityManager, HumanInteractionManager, MetaEvolutionLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_markdown_converter():
    """Example: Evolve a Markdown to HTML converter."""
    
    # Initialize evolutionary agent
    config_path = Path("config.yaml")
    evolution_config = EvolutionConfig(
        population_size=20,
        generations=50,
        mutation_rate=0.4,
        elite_size=3
    )
    
    agent = EvolutionaryAgent("markdown_converter", config_path, evolution_config)
    
    # Set the task
    agent.set_task(
        task_description="Create a robust Markdown to HTML converter that handles edge cases",
        initial_tools={
            "parse_markdown": "def parse_markdown(text): pass",
            "render_html": "def render_html(ast): pass"
        },
        initial_spec={
            "test_cases": ["basic formatting", "links", "lists"],
            "performance_target": "50ms latency"
        },
        initial_prompt="Write a function convert(text) that converts Markdown to HTML"
    )
    
    # Step 1-4: Interactive planning
    planning_result = await agent.interactive_planning()
    logger.info("Planning completed")
    
    # Set evaluation function
    evaluator = create_evaluator("markdown_converter")
    agent.set_evaluation_function(evaluator.evaluate)
    
    # Step 5: Evolutionary optimization
    best_candidate = await agent.evolutionary_optimization()
    
    logger.info(f"Best candidate: {best_candidate.id}")
    logger.info(f"Fitness score: {best_candidate.fitness_score}")
    
    return best_candidate


async def demonstrate_robust_parsing():
    """Demonstrate robust LLM response parsing."""
    
    parser = RobustLLMParser()
    
    # Example 1: AlphaEvolve format parsing
    alpha_evolve_response = """
    Here are the improvements:
    
    <<<<<<< SEARCH
    def convert(text):
        return text
    =======
    def convert(text):
        if not text:
            return ""
        return process_markdown(text)
    >>>>>>> REPLACE
    """
    
    diff_blocks = parser.parse_mutation_response(alpha_evolve_response)
    logger.info(f"Parsed {len(diff_blocks)} diff blocks")
    
    # Example 2: Code block parsing
    code_block_response = """
    Here's the improved function:
    
    ```python
    def convert(text):
        return process_markdown(text)
    ```
    """
    
    diff_blocks = parser.parse_mutation_response(code_block_response)
    logger.info(f"Parsed {len(diff_blocks)} code blocks")
    
    # Example 3: List parsing
    list_response = """
    Edge cases to consider:
    1. Unicode characters
    2. Nested lists
    3. Mixed content
    """
    
    items = parser.parse_list_response(list_response)
    logger.info(f"Parsed {len(items)} list items")


async def demonstrate_diversity_management():
    """Demonstrate diversity management features."""
    
    from diversity_manager import DiversityConfig
    
    config = DiversityConfig(
        min_cluster_size=2,
        max_clusters=5,
        diversity_weight=0.2
    )
    
    diversity_manager = DiversityManager(config)
    
    # Simulate candidates with different characteristics
    candidates = []
    for i in range(10):
        candidate = type('Candidate', (), {
            'code': f"def approach_{i}(): pass",
            'prompt': f"prompt_{i}",
            'fitness_score': 0.5 + (i * 0.1)
        })()
        candidates.append(candidate)
    
    # Cluster candidates
    clusters = diversity_manager.cluster_candidates(candidates)
    logger.info(f"Created {len(clusters)} clusters")
    
    # Select diverse candidates
    diverse_selection = diversity_manager.select_diverse_candidates(candidates, 5)
    logger.info(f"Selected {len(diverse_selection)} diverse candidates")
    
    # Calculate diversity metrics
    metrics = diversity_manager.get_diversity_metrics()
    logger.info(f"Diversity metrics: {metrics}")


async def demonstrate_human_interaction():
    """Demonstrate human-in-the-loop interactions."""
    
    human_manager = HumanInteractionManager()
    
    # Example edge case suggestions
    edge_cases = [
        "Empty input handling",
        "Unicode character processing",
        "Nested structure handling"
    ]
    
    # Get human input (simulated)
    approved_cases = await human_manager.get_human_input("edge_cases", edge_cases)
    logger.info(f"Human approved {len(approved_cases)} edge cases")
    
    # Example tool recommendations
    tool_recommendations = {
        'tools': [
            {'name': 'validate_input', 'description': 'Validate user input'},
            {'name': 'sanitize_output', 'description': 'Sanitize HTML output'}
        ]
    }
    
    approved_tools = await human_manager.get_human_input("tools", tool_recommendations)
    logger.info(f"Human approved tools: {approved_tools}")
    
    # Get interaction summary
    summary = human_manager.get_interaction_summary()
    logger.info(f"Interaction summary: {summary}")


async def demonstrate_meta_evolution():
    """Demonstrate meta-evolution logging."""
    
    meta_logger = MetaEvolutionLogger()
    
    # Log prompt performance
    meta_logger.log_prompt_performance(
        "You are an expert programmer. Improve this code:",
        0.8,  # 80% success rate
        0.75  # Average fitness
    )
    
    # Log mutation success
    meta_logger.log_mutation_success(
        "code_optimization",
        True,  # Success
        0.15   # 15% fitness gain
    )
    
    # Get insights
    best_prompts = meta_logger.get_best_prompts()
    logger.info(f"Best prompts: {best_prompts}")
    
    mutation_insights = meta_logger.get_mutation_insights()
    logger.info(f"Mutation insights: {mutation_insights}")


async def demonstrate_patch_management():
    """Demonstrate robust patch application."""
    
    patch_manager = PatchManager()
    
    # Original code
    original_code = """
def convert(text):
    return text
"""
    
    # LLM response with diff
    llm_response = """
    Here's the improved version:
    
    <<<<<<< SEARCH
    def convert(text):
        return text
    =======
    def convert(text):
        if not text:
            return ""
        return process_markdown(text)
    >>>>>>> REPLACE
    """
    
    # Parse and apply patch
    diff_blocks = patch_manager.parse_llm_diff(llm_response)
    patched_code = patch_manager.apply_patch(original_code, diff_blocks)
    
    logger.info("Original code:")
    logger.info(original_code)
    logger.info("Patched code:")
    logger.info(patched_code)
    
    # Validate patch
    is_valid = patch_manager.validate_patch(original_code, patched_code)
    logger.info(f"Patch is valid: {is_valid}")
    
    # Create diff summary
    summary = patch_manager.create_diff_summary(original_code, patched_code)
    logger.info(f"Patch summary: {summary}")


async def main():
    """Run all demonstrations."""
    
    logger.info("=== Evolutionary Agent Improvements Demo ===")
    
    # Demonstrate robust parsing
    logger.info("\n1. Robust LLM Response Parsing")
    await demonstrate_robust_parsing()
    
    # Demonstrate diversity management
    logger.info("\n2. Diversity Management")
    await demonstrate_diversity_management()
    
    # Demonstrate human interaction
    logger.info("\n3. Human-in-the-Loop Interaction")
    await demonstrate_human_interaction()
    
    # Demonstrate meta-evolution
    logger.info("\n4. Meta-Evolution Logging")
    await demonstrate_meta_evolution()
    
    # Demonstrate patch management
    logger.info("\n5. Robust Patch Management")
    await demonstrate_patch_management()
    
    # Run full evolutionary example
    logger.info("\n6. Full Evolutionary Example")
    try:
        best_candidate = await example_markdown_converter()
        logger.info("Evolutionary optimization completed successfully!")
    except Exception as e:
        logger.error(f"Evolutionary optimization failed: {e}")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main()) 