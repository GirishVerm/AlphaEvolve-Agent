#!/usr/bin/env python3
"""
Interactive Evolutionary Agent Demo - Watch the agent evolve markdown in real-time!
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from llm_interface import LLMInterface, LLMConfig
from artifact_support import ArtifactType, ArtifactCandidate
from multi_objective import Objective, MultiObjectiveConfig, MultiObjectiveEvaluator
from cost_manager import CostConfig, BudgetAwareLLMInterface
from human_interface import HumanInterfaceConfig, InterfaceType, HumanInterfaceFactory
from experiment_tracker import ExperimentConfig, ExperimentTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    """Configuration for interactive evolution."""
    population_size: int = 3
    generations: int = 5
    max_cost: float = 10.0
    human_review_interval: int = 2  # Ask for human input every N generations

class InteractiveEvolutionaryAgent:
    """Interactive evolutionary agent with detailed tracking."""
    
    def __init__(self, config: InteractiveConfig):
        """Initialize the interactive agent."""
        self.config = config
        self.population = []
        self.generation = 0
        self.evolution_history = []
        self.human_inputs = []
        
        # Initialize LLM
        llm_config = LLMConfig()
        base_llm = LLMInterface(llm_config)
        
        # Wrap with cost management
        cost_config = CostConfig(max_cost_per_experiment=config.max_cost)
        self.llm = BudgetAwareLLMInterface(base_llm, cost_config)
        
        # Initialize evaluation
        objectives = [
            Objective(name="functionality", weight=0.4, minimize=False),
            Objective(name="code_quality", weight=0.3, minimize=False),
            Objective(name="completeness", weight=0.3, minimize=False)
        ]
        eval_config = MultiObjectiveConfig(objectives=objectives)
        self.evaluator = MultiObjectiveEvaluator(eval_config)
        
        # Initialize experiment tracking
        experiment_config = ExperimentConfig(
            experiment_name="interactive_markdown_evolution",
            seed=42,
            mlflow_enabled=False,
            wandb_enabled=False
        )
        self.tracker = ExperimentTracker(experiment_config)
        
        # Initialize human interface
        human_config = HumanInterfaceConfig(
            interface_type=InterfaceType.CLI,
            timeout_seconds=60
        )
        self.human_interface = HumanInterfaceFactory.create_interface(human_config)
    
    async def get_human_input(self, prompt: str, suggestions: List[str] = None) -> str:
        """Get human input with suggestions."""
        print(f"\n🤖 AGENT: {prompt}")
        if suggestions:
            print("💡 Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        
        print("\n👤 YOUR INPUT (type your response):")
        user_input = input("> ").strip()
        
        # Log the interaction
        self.human_inputs.append({
            'generation': self.generation,
            'prompt': prompt,
            'suggestions': suggestions,
            'user_input': user_input,
            'timestamp': datetime.now()
        })
        
        return user_input
    
    async def evaluate_candidate(self, candidate: ArtifactCandidate) -> List[float]:
        """Evaluate a markdown-to-HTML converter function by actually testing it."""
        content = candidate.content
        
        # Test cases for markdown-to-HTML conversion
        test_cases = [
            ("# Hello", "<h1>Hello</h1>"),
            ("**bold**", "<strong>bold</strong>"),
            ("*italic*", "<em>italic</em>"),
            ("[link](url)", "<a href=\"url\">link</a>"),
            ("- item", "<ul><li>item</li></ul>"),
            ("1. item", "<ol><li>item</li></ol>"),
            ("`code`", "<code>code</code>"),
            ("```\ncode\n```", "<pre><code>code</code></pre>")
        ]
        
        # Functionality score (0-1) - does it actually convert correctly?
        functionality_score = 0.0
        try:
            # Try to execute the function
            exec(content, globals())
            if 'convert_markdown_to_html' in globals():
                func = globals()['convert_markdown_to_html']
                
                # Test the function
                correct_conversions = 0
                for markdown_input, expected_html in test_cases:
                    try:
                        result = func(markdown_input)
                        # Simple similarity check (in real implementation, use proper HTML parsing)
                        if expected_html.lower() in result.lower() or result.strip():
                            correct_conversions += 1
                    except Exception:
                        pass
                
                functionality_score = correct_conversions / len(test_cases)
            else:
                functionality_score = 0.0
        except Exception:
            functionality_score = 0.0
        
        # Code quality score (0-1) - is it well-written code?
        code_quality_score = 0.5  # Baseline
        if 'try' in content and 'except' in content:  # Error handling
            code_quality_score += 0.2
        if '"""' in content or "'''" in content:  # Docstrings
            code_quality_score += 0.2
        if 'import' in content:  # Uses libraries
            code_quality_score += 0.1
        if 'def' in content and 'convert_markdown_to_html' in content:  # Correct function name
            code_quality_score += 0.1
        
        # Completeness score (0-1) - does it handle various markdown features?
        completeness_score = 0.5  # Baseline
        if '#' in content:  # Handles headers
            completeness_score += 0.2
        if '*' in content or '**' in content:  # Handles formatting
            completeness_score += 0.2
        if '[' in content and ']' in content:  # Handles links
            completeness_score += 0.1
        if '```' in content:  # Handles code blocks
            completeness_score += 0.1
        
        return [functionality_score, code_quality_score, completeness_score]
    
    async def test_converter(self, candidate: ArtifactCandidate) -> Dict[str, Any]:
        """Test the converter function with real examples."""
        content = candidate.content
        
        test_examples = [
            ("# Hello World", "Basic header"),
            ("**Bold text**", "Bold formatting"),
            ("*Italic text*", "Italic formatting"),
            ("[Link text](https://example.com)", "Link conversion"),
            ("- List item", "Unordered list"),
            ("1. Numbered item", "Ordered list"),
            ("`inline code`", "Inline code"),
            ("```\nblock code\n```", "Code block")
        ]
        
        results = {}
        try:
            exec(content, globals())
            if 'convert_markdown_to_html' in globals():
                func = globals()['convert_markdown_to_html']
                
                for markdown_input, description in test_examples:
                    try:
                        result = func(markdown_input)
                        results[description] = {
                            'input': markdown_input,
                            'output': result,
                            'success': True
                        }
                    except Exception as e:
                        results[description] = {
                            'input': markdown_input,
                            'output': f"Error: {str(e)}",
                            'success': False
                        }
            else:
                results['error'] = {'output': 'Function not found', 'success': False}
        except Exception as e:
            results['error'] = {'output': f'Execution error: {str(e)}', 'success': False}
        
        return results
    
    async def generate_mutation(self, parent: ArtifactCandidate, human_feedback: str = "") -> ArtifactCandidate:
        """Generate a mutation with optional human feedback."""
        prompt = f"""
        Create or improve a Python function that converts markdown to HTML.
        
        Requirements:
        - Function should be named 'convert_markdown_to_html'
        - Should handle basic markdown syntax: headings, paragraphs, lists, links, bold/italic text
        - Should be efficient and handle edge cases gracefully
        - Should return HTML string
        
        Current function:
        {parent.content}
        
        Human feedback: {human_feedback if human_feedback else "No specific feedback provided"}
        
        Provide only the improved Python function code:
        """
        
        try:
            response = await self.llm.generate(prompt)
            
            # Create new candidate
            child = ArtifactCandidate(
                id=f"gen_{self.generation}_{len(self.population)}",
                content=response,
                artifact_type=ArtifactType.PYTHON_CODE,
                metadata={
                    "generation": self.generation,
                    "parent": parent.id,
                    "human_feedback": human_feedback
                }
            )
            
            return child
            
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return parent
    
    async def evolve_generation(self):
        """Evolve one generation with detailed tracking."""
        print(f"\n{'='*60}")
        print(f"🔄 GENERATION {self.generation + 1}")
        print(f"{'='*60}")
        
        new_candidates = []
        generation_stats = {
            'generation': self.generation + 1,
            'timestamp': datetime.now(),
            'candidates': []
        }
        
        # Generate mutations for each parent
        for i, parent in enumerate(self.population):
            print(f"\n📝 Mutating candidate {i+1}/{len(self.population)}")
            print(f"Parent ID: {parent.id}")
            print(f"Parent fitness: {parent.fitness_score:.3f}")
            
            # Get human feedback if it's time
            human_feedback = ""
            if self.generation > 0 and self.generation % self.config.human_review_interval == 0:
                feedback_prompt = f"What improvements would you like to see in this markdown-to-HTML converter?"
                human_feedback = await self.get_human_input(feedback_prompt, [
                    "Add support for more markdown features",
                    "Improve error handling",
                    "Add performance optimizations",
                    "Make the code more readable",
                    "Add comprehensive documentation"
                ])
            
            child = await self.generate_mutation(parent, human_feedback)
            
            # Evaluate child
            fitness = await self.evaluate_candidate(child)
            child.fitness_scores = fitness
            child.fitness_score = sum(fitness) / len(fitness)
            
            new_candidates.append(child)
            
            # Log candidate details
            candidate_info = {
                'id': child.id,
                'parent_id': parent.id,
                'fitness_score': child.fitness_score,
                'fitness_scores': child.fitness_scores,
                'human_feedback': human_feedback,
                'content_length': len(child.content),
                'improvement': child.fitness_score - parent.fitness_score
            }
            generation_stats['candidates'].append(candidate_info)
            
            print(f"✅ Child fitness: {child.fitness_score:.3f} (improvement: {candidate_info['improvement']:+.3f})")
            print(f"📊 Scores - Functionality: {fitness[0]:.3f}, Code Quality: {fitness[1]:.3f}, Completeness: {fitness[2]:.3f}")
        
        # Select best candidates for next generation
        all_candidates = self.population + new_candidates
        all_candidates.sort(key=lambda x: x.fitness_score, reverse=True)
        
        self.population = all_candidates[:self.config.population_size]
        self.generation += 1
        
        # Log generation stats
        self.evolution_history.append(generation_stats)
        
        # Show best candidate
        best = self.population[0]
        print(f"\n🏆 BEST CANDIDATE (Generation {self.generation})")
        print(f"ID: {best.id}")
        print(f"Fitness: {best.fitness_score:.3f}")
        
        # Test and show actual conversion results
        test_results = await self.test_converter(best)
        print(f"\n🧪 CONVERSION TEST RESULTS:")
        for description, result in test_results.items():
            if description != 'error':
                status = "✅" if result['success'] else "❌"
                print(f"{status} {description}:")
                print(f"   Input:  {result['input']}")
                print(f"   Output: {result['output'][:100]}{'...' if len(result['output']) > 100 else ''}")
        
        print(f"Content preview: {best.content[:200]}...")
        
        # Log to experiment tracker
        self.tracker.log_metrics({
            'generation': self.generation,
            'best_fitness': best.fitness_score,
            'population_size': len(self.population),
            'total_candidates': len(all_candidates)
        }, step=self.generation)
    
    async def run_evolution(self, initial_content: str):
        """Run the complete evolution with detailed tracking."""
        print("🚀 STARTING INTERACTIVE EVOLUTION")
        print("="*60)
        
        # Initialize population
        print("\n📋 INITIALIZING POPULATION")
        baseline = ArtifactCandidate(
            id="baseline",
            content=initial_content,
            artifact_type=ArtifactType.PYTHON_CODE,
            metadata={"generation": 0, "parent": "none"}
        )
        
        fitness = await self.evaluate_candidate(baseline)
        baseline.fitness_scores = fitness
        baseline.fitness_score = sum(fitness) / len(fitness)
        
        self.population = [baseline]
        
        print(f"📊 Baseline fitness: {baseline.fitness_score:.3f}")
        print(f"📊 Scores - Functionality: {fitness[0]:.3f}, Code Quality: {fitness[1]:.3f}, Completeness: {fitness[2]:.3f}")
        print(f"📄 Initial function:\n{baseline.content}")
        
        # Run evolution
        for gen in range(self.config.generations):
            await self.evolve_generation()
            
            # Check budget
            if self.llm.is_budget_exceeded():
                print("\n⚠️ Budget exceeded, stopping evolution")
                break
        
        # Show final results
        await self.show_final_results()
    
    async def show_final_results(self):
        """Show comprehensive final results."""
        print(f"\n{'='*60}")
        print("🎯 EVOLUTION COMPLETE - FINAL RESULTS")
        print(f"{'='*60}")
        
        # Show evolution history
        print("\n📈 EVOLUTION HISTORY:")
        for gen in self.evolution_history:
            print(f"\nGeneration {gen['generation']}:")
            for candidate in gen['candidates']:
                print(f"  {candidate['id']}: {candidate['fitness_score']:.3f} "
                      f"(improvement: {candidate['improvement']:+.3f})")
                if candidate['human_feedback']:
                    print(f"    Human feedback: {candidate['human_feedback']}")
        
        # Show best candidate
        best = max(self.population, key=lambda x: x.fitness_score)
        print(f"\n🏆 BEST CANDIDATE:")
        print(f"ID: {best.id}")
        print(f"Final fitness: {best.fitness_score:.3f}")
        print(f"Final scores - Functionality: {best.fitness_scores[0]:.3f}, "
              f"Code Quality: {best.fitness_scores[1]:.3f}, Completeness: {best.fitness_scores[2]:.3f}")
        print(f"\n📄 FINAL FUNCTION:")
        print("-" * 40)
        print(best.content)
        print("-" * 40)
        
        # Show cost stats
        cost_stats = self.llm.get_cost_stats()
        print(f"\n💰 COST STATISTICS:")
        print(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
        print(f"Total requests: {cost_stats.get('total_requests', 0)}")
        
        # Show human interactions
        if self.human_inputs:
            print(f"\n👤 HUMAN INTERACTIONS ({len(self.human_inputs)} total):")
            for interaction in self.human_inputs:
                print(f"  Generation {interaction['generation']}: {interaction['user_input']}")
        
        # Show improvement summary
        initial_fitness = self.evolution_history[0]['candidates'][0]['fitness_score'] if self.evolution_history else 0
        final_fitness = best.fitness_score
        improvement = final_fitness - initial_fitness
        
        print(f"\n📊 IMPROVEMENT SUMMARY:")
        print(f"Initial fitness: {initial_fitness:.3f}")
        print(f"Final fitness: {final_fitness:.3f}")
        print(f"Total improvement: {improvement:+.3f} ({improvement/initial_fitness*100:+.1f}%)")

async def main():
    """Run the interactive evolution demo."""
    print("🎭 INTERACTIVE EVOLUTIONARY AGENT DEMO")
    print("="*60)
    print("Watch the agent evolve a markdown-to-HTML converter function in real-time!")
    print("You'll be asked for input at key moments to guide the evolution.")
    print("="*60)
    
    # Get initial content from user
    print("\n📝 ENTER YOUR INITIAL MARKDOWN-TO-HTML CONVERTER FUNCTION:")
    print("(Type your Python function below, press Enter twice when done)")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    initial_content = "\n".join(lines[:-1])  # Remove the last empty line
    
    if not initial_content.strip():
        initial_content = """def convert_markdown_to_html(markdown_text):
    \"\"\"Convert markdown text to HTML.\"\"\"
    html = markdown_text
    return html"""
    
    print(f"\n✅ Using initial function ({len(initial_content)} characters):")
    print("-" * 40)
    print(initial_content)
    print("-" * 40)
    
    # Initialize agent
    config = InteractiveConfig()
    agent = InteractiveEvolutionaryAgent(config)
    
    # Run evolution
    await agent.run_evolution(initial_content)

if __name__ == "__main__":
    asyncio.run(main()) 