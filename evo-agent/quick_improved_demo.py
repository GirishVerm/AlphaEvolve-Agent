#!/usr/bin/env python3
"""
Quick Improved Dual-Loop Demo - Small environment with better JSON parsing.
"""
import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from llm_interface import LLMInterface, LLMConfig
from artifact_support import ArtifactType, ArtifactCandidate
from multi_objective import Objective, MultiObjectiveConfig, MultiObjectiveEvaluator
from cost_manager import CostConfig, BudgetAwareLLMInterface
from patch_manager import PatchManager
from diff_generator import DiffGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskDefinition:
    """Defines the task for dual-loop evolution."""
    outcome: str
    stubs: List[str]
    initial_prompts: List[str]
    performance_target: float

@dataclass
class HardEvalSuite:
    """Comprehensive evaluation suite for artifacts."""
    unit_tests: List[Dict[str, Any]]
    performance_benchmarks: List[Dict[str, Any]]
    robustness_tests: List[Dict[str, Any]]
    edge_cases: List[str]

@dataclass
class EvolutionConfig:
    """Configuration for quick improved dual-loop evolution."""
    artifact_population_size: int = 2  # μ (parent population) - REDUCED
    artifact_children_per_parent: int = 2  # λ (children per parent) - REDUCED
    scaffold_population_size: int = 2  # REDUCED
    generations: int = 2  # REDUCED
    max_cost: float = 20.0  # REDUCED
    checkpoint_interval: int = 1

class QuickImprovedDualLoopSystem:
    """Quick improved dual-loop evolution with better JSON parsing."""
    
    def __init__(self, task: TaskDefinition, config: EvolutionConfig):
        """Initialize the quick improved dual-loop evolution system."""
        self.task = task
        self.config = config
        self.generation = 0
        
        # Initialize LLM
        llm_config = LLMConfig()
        base_llm = LLMInterface(llm_config)
        cost_config = CostConfig(max_cost_per_experiment=config.max_cost)
        self.llm = BudgetAwareLLMInterface(base_llm, cost_config)
        
        # Initialize components
        self.patch_manager = PatchManager()
        self.diff_generator = DiffGenerator(llm_config)
        
        # Evolution pools
        self.artifact_population = []  # μ parents
        self.scaffold_population = {
            'prompts': [],
            'tools': [],
            'memory': [],
            'evaluation_methods': []
        }
        
        # Evaluation suite
        self.eval_suite = None
        
        # Evolution history
        self.artifact_history = []
        self.scaffold_history = []
        
        # Initialize evaluation
        objectives = [
            Objective(name="correctness", weight=0.4, minimize=False),
            Objective(name="performance", weight=0.3, minimize=True),
            Objective(name="robustness", weight=0.3, minimize=False)
        ]
        eval_config = MultiObjectiveConfig(objectives=objectives)
        self.evaluator = MultiObjectiveEvaluator(eval_config)
    
    async def generate_hard_eval_suite(self) -> HardEvalSuite:
        """Generate comprehensive evaluation suite with IMPROVED JSON parsing."""
        prompt = f"""
        Create a comprehensive evaluation suite for markdown-to-HTML conversion.
        
        Performance target: {self.task.performance_target}ms
        
        Return ONLY a valid JSON object with this EXACT structure (no extra text, no explanations):
        {{
            "unit_tests": [
                {{"input": "# Hello", "expected": "<h1>Hello</h1>", "name": "basic_heading"}},
                {{"input": "**bold**", "expected": "<strong>bold</strong>", "name": "basic_bold"}},
                {{"input": "*italic*", "expected": "<em>italic</em>", "name": "basic_italic"}},
                {{"input": "- item", "expected": "<ul><li>item</li></ul>", "name": "basic_list"}}
            ],
            "performance_benchmarks": [
                {{"name": "small_text", "input": "# Test", "max_time": {self.task.performance_target}}},
                {{"name": "large_text", "input": "# " + "x" * 100, "max_time": {self.task.performance_target * 5}}}
            ],
            "robustness_tests": [
                {{"name": "empty_input", "input": "", "should_handle": true}},
                {{"name": "null_input", "input": null, "should_handle": true}}
            ],
            "edge_cases": [
                "Unclosed tags: **bold without closing",
                "Mixed line endings: \\r\\n and \\n",
                "HTML passthrough: <script>alert('xss')</script>",
                "RTL text: مرحبا بالعالم"
            ]
        }}
        """
        
        try:
            response = await self.llm.generate(prompt)
            print(f"🔍 LLM Response for eval suite: {response[:200]}...")
            
            # Try multiple JSON extraction methods
            json_str = self._extract_json_from_response(response)
            if json_str:
                print(f"✅ Extracted JSON: {json_str[:100]}...")
                eval_data = json.loads(json_str)
                logger.info("✅ Successfully generated comprehensive eval suite")
                return HardEvalSuite(**eval_data)
            else:
                logger.warning("❌ Failed to extract JSON, using fallback")
                return self._create_basic_eval_suite()
                
        except Exception as e:
            logger.warning(f"Failed to generate eval suite: {e}")
            return self._create_basic_eval_suite()
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from LLM response using multiple methods."""
        # Method 1: Look for JSON in code blocks
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # Method 2: Look for JSON between braces
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            return response[json_start:json_end]
        
        # Method 3: Look for JSON in any code block
        code_block_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                json.loads(code_block_match.group(1).strip())
                return code_block_match.group(1).strip()
            except:
                pass
        
        # Method 4: Try to find JSON-like structure
        lines = response.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            if '{' in line:
                in_json = True
                brace_count += line.count('{')
            if in_json:
                json_lines.append(line)
                brace_count -= line.count('}')
                if brace_count <= 0:
                    break
        
        if json_lines:
            try:
                json_str = '\n'.join(json_lines)
                json.loads(json_str)
                return json_str
            except:
                pass
        
        return None
    
    def _create_basic_eval_suite(self) -> HardEvalSuite:
        """Create a basic evaluation suite as fallback."""
        return HardEvalSuite(
            unit_tests=[
                {"input": "# Hello", "expected": "<h1>Hello</h1>", "name": "basic_heading"},
                {"input": "**bold**", "expected": "<strong>bold</strong>", "name": "basic_bold"},
                {"input": "*italic*", "expected": "<em>italic</em>", "name": "basic_italic"},
                {"input": "- item", "expected": "<ul><li>item</li></ul>", "name": "basic_list"}
            ],
            performance_benchmarks=[
                {"name": "small_text", "input": "# Test", "max_time": self.task.performance_target},
                {"name": "large_text", "input": "# " + "x" * 100, "max_time": self.task.performance_target * 5}
            ],
            robustness_tests=[
                {"name": "empty_input", "input": "", "should_handle": True},
                {"name": "null_input", "input": None, "should_handle": True}
            ],
            edge_cases=[
                "Unclosed tags: **bold without closing",
                "Mixed line endings: \r\n and \n",
                "HTML passthrough: <script>alert('xss')</script>",
                "RTL text: مرحبا بالعالم"
            ]
        )
    
    async def initialize_populations(self):
        """Initialize both artifact and scaffold populations."""
        print("🔧 INITIALIZING QUICK IMPROVED DUAL-LOOP POPULATIONS")
        
        # Initialize artifact population (μ parents)
        baseline_converter = self._create_baseline_converter()
        self.artifact_population = [baseline_converter] * self.config.artifact_population_size
        
        # Initialize scaffold population
        self.scaffold_population = {
            'prompts': [
                "Write convert(text) using parse_markdown/render_html.",
                "Optimize this converter for speed."
            ],
            'tools': [
                "def parse_markdown(text) -> AST: ...",
                "def render_html(ast) -> str: ..."
            ],
            'memory': [
                "def store_context(task, result): ...",
                "def retrieve_knowledge(query): ..."
            ],
            'evaluation_methods': [
                "def test_correctness(code, test_cases): ...",
                "def benchmark_performance(code, benchmarks): ..."
            ]
        }
        
        print(f"✅ Artifact population (μ): {len(self.artifact_population)}")
        print(f"✅ Children per parent (λ): {self.config.artifact_children_per_parent}")
        print(f"✅ Scaffold components: {sum(len(v) for v in self.scaffold_population.values())}")
    
    def _create_baseline_converter(self) -> ArtifactCandidate:
        """Create baseline markdown-to-HTML converter."""
        baseline_code = '''
def convert_markdown_to_html(text):
    """Basic markdown to HTML converter."""
    if not text:
        return ""
    
    # Simple conversions
    html = text
    html = html.replace("# ", "<h1>").replace("\\n", "</h1>\\n")
    html = html.replace("**", "<strong>").replace("**", "</strong>")
    html = html.replace("*", "<em>").replace("*", "</em>")
    
    return html
'''
        return ArtifactCandidate(
            id=f"baseline_{int(time.time())}",
            content=baseline_code,
            artifact_type=ArtifactType.PYTHON_CODE,
            metadata={"generation": 0, "parent": "baseline"}
        )
    
    async def artifact_evolution_loop(self) -> List[ArtifactCandidate]:
        """Evolve artifact population using μ→λ selection."""
        print(f"\n🔄 ARTIFACT EVOLUTION LOOP - GENERATION {self.generation + 1}")
        print(f"μ={len(self.artifact_population)} parents → λ={self.config.artifact_children_per_parent} children each")
        
        all_children = []
        
        # Generate children for each parent (μ→λ)
        for i, parent in enumerate(self.artifact_population):
            print(f"Evolving parent {i+1}/{len(self.artifact_population)}")
            
            parent_children = []
            for j in range(self.config.artifact_children_per_parent):
                # Generate mutation using EVOLVE-BLOCK
                mutation_prompt = f"""
                Improve this markdown-to-HTML converter within the EVOLVE-BLOCK:
                
                EVOLVE-BLOCK START
                {parent.content}
                EVOLVE-BLOCK END
                
                Requirements:
                - Improve correctness, performance, and robustness
                - Handle edge cases better
                - Optimize for speed (target: {self.task.performance_target}ms)
                - Add error handling and input validation
                - Support more markdown features (lists, links, etc.)
                
                Provide only the improved code within EVOLVE-BLOCK:
                """
                
                try:
                    response = await self.llm.generate(mutation_prompt)
                    
                    # Extract code from EVOLVE-BLOCK
                    evolve_pattern = r'EVOLVE-BLOCK START\s*(.*?)\s*EVOLVE-BLOCK END'
                    match = re.search(evolve_pattern, response, re.DOTALL)
                    
                    if match:
                        evolved_code = match.group(1).strip()
                        print(f"  ✅ Extracted {len(evolved_code)} chars from EVOLVE-BLOCK")
                    else:
                        evolved_code = response.strip()
                        print(f"  ⚠️ No EVOLVE-BLOCK found, using full response")
                    
                    # Debug: show first 100 chars of evolved code
                    print(f"  📝 Evolved code preview: {evolved_code[:100]}...")
                    
                    # Create evolved candidate
                    evolved_candidate = ArtifactCandidate(
                        id=f"artifact_{self.generation}_{i}_{j}_{int(time.time())}",
                        content=evolved_code,
                        artifact_type=ArtifactType.PYTHON_CODE,
                        metadata={
                            "generation": self.generation + 1,
                            "parent": parent.id,
                            "mutation_type": "llm_guided",
                            "child_index": j
                        }
                    )
                    
                    # Evaluate evolved candidate with namespace hygiene
                    fitness = await self.evaluate_artifact_with_hygiene(evolved_candidate)
                    evolved_candidate.fitness_score = sum(fitness) / len(fitness)
                    
                    parent_children.append(evolved_candidate)
                    print(f"  Child {j+1} fitness: {evolved_candidate.fitness_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Artifact evolution failed for parent {i}, child {j}: {e}")
                    # Keep parent as fallback
                    parent_children.append(parent)
            
            all_children.extend(parent_children)
        
        # Select top μ from all children (μ→λ→μ selection)
        all_children.sort(key=lambda x: x.fitness_score, reverse=True)
        selected_population = all_children[:self.config.artifact_population_size]
        
        print(f"✅ Selected top {len(selected_population)} from {len(all_children)} children")
        return selected_population
    
    async def evaluate_artifact_with_hygiene(self, candidate: ArtifactCandidate) -> List[float]:
        """Evaluate artifact using hard eval suite with namespace hygiene."""
        if not self.eval_suite:
            self.eval_suite = await self.generate_hard_eval_suite()
        
        correctness_score = 0.0
        performance_score = 0.0
        robustness_score = 0.0
        
        try:
            # Create fresh namespace for each evaluation
            fresh_globals = {}
            exec(candidate.content, fresh_globals)
            
            if 'convert_markdown_to_html' not in fresh_globals:
                return [0.0, 1.0, 0.0]  # No function found
            
            func = fresh_globals['convert_markdown_to_html']
            
            # Test correctness
            correct_tests = 0
            for test in self.eval_suite.unit_tests:
                try:
                    result = func(test['input'])
                    if test['expected'].lower() in result.lower():
                        correct_tests += 1
                except Exception:
                    pass
            correctness_score = correct_tests / len(self.eval_suite.unit_tests)
            
            # Test performance
            try:
                start_time = time.time()
                func("# " + "x" * 100)  # Performance test
                elapsed = (time.time() - start_time) * 1000  # Convert to ms
                performance_score = max(0, 1 - (elapsed / self.task.performance_target))
            except Exception:
                performance_score = 0.0
            
            # Test robustness
            robust_tests = 0
            for test in self.eval_suite.robustness_tests:
                try:
                    func(test['input'])
                    robust_tests += 1
                except Exception:
                    pass
            robustness_score = robust_tests / len(self.eval_suite.robustness_tests)
            
        except Exception as e:
            logger.warning(f"Artifact evaluation failed: {e}")
            return [0.0, 0.0, 0.0]
        
        return [correctness_score, performance_score, robustness_score]
    
    async def meta_evolution_loop(self) -> Dict[str, List[str]]:
        """Evolve scaffold population with improved proxy evaluation."""
        print(f"\n🧠 META-EVOLUTION LOOP - GENERATION {self.generation + 1}")
        
        new_scaffold = {}
        
        for component_type, components in self.scaffold_population.items():
            print(f"Evolving {component_type}...")
            new_components = []
            
            for i, component in enumerate(components):
                # Generate mutation for scaffold component
                mutation_prompt = f"""
                Improve this {component_type} component:
                
                Current {component_type}:
                {component}
                
                Requirements:
                - Make it more effective for markdown-to-HTML conversion
                - Improve efficiency and performance
                - Make it more adaptable to different scenarios
                - Ensure it's syntactically correct
                
                Provide only the improved {component_type}:
                """
                
                try:
                    response = await self.llm.generate(mutation_prompt)
                    
                    # Test the mutated component with improved proxy evaluation
                    proxy_fitness = await self.improved_proxy_evaluate_scaffold_component(
                        component_type, response, self.artifact_population[0]
                    )
                    
                    if proxy_fitness > 0.5:  # Keep if it improves performance
                        new_components.append(response)
                        print(f"✅ {component_type} {i+1} improved (proxy fitness: {proxy_fitness:.3f})")
                    else:
                        new_components.append(component)  # Keep original
                        print(f"⚠️ {component_type} {i+1} kept original (proxy fitness: {proxy_fitness:.3f})")
                        
                except Exception as e:
                    logger.warning(f"Scaffold evolution failed for {component_type}: {e}")
                    new_components.append(component)
            
            new_scaffold[component_type] = new_components
        
        return new_scaffold
    
    async def improved_proxy_evaluate_scaffold_component(self, component_type: str, new_component: str, test_artifact: ArtifactCandidate) -> float:
        """Improved proxy evaluation with better error handling."""
        try:
            # Temporarily replace scaffold component
            original_component = self.scaffold_population[component_type][0]
            self.scaffold_population[component_type][0] = new_component
            
            # Test the new component by generating a single mutation
            test_prompt = f"""
            Improve this markdown-to-HTML converter using the new {component_type}:
            
            Current code:
            {test_artifact.content}
            
            New {component_type}:
            {new_component}
            
            Provide only the improved code:
            """
            
            try:
                response = await self.llm.generate(test_prompt)
                
                # Create a test candidate with the improved code
                test_candidate = ArtifactCandidate(
                    id=f"proxy_test_{int(time.time())}",
                    content=response,
                    artifact_type=ArtifactType.PYTHON_CODE,
                    metadata={"generation": self.generation, "proxy_test": True}
                )
                
                # Evaluate the test candidate with namespace hygiene
                fitness = await self.evaluate_artifact_with_hygiene(test_candidate)
                proxy_score = sum(fitness) / len(fitness)
                
            except Exception as e:
                logger.warning(f"Proxy test failed: {e}")
                proxy_score = 0.0
            
            # Restore original component
            self.scaffold_population[component_type][0] = original_component
            
            return proxy_score
            
        except Exception as e:
            logger.warning(f"Proxy evaluation failed: {e}")
            return 0.0
    
    async def run_quick_improved_dual_loop_evolution(self):
        """Run the quick improved dual-loop evolution."""
        print("🚀 STARTING QUICK IMPROVED DUAL-LOOP EVOLUTION")
        print("="*60)
        print("μ→λ Evolution + Meta-Evolution (Small Environment)")
        print("="*60)
        
        # Initialize populations
        await self.initialize_populations()
        
        # Generate hard eval suite
        print("\n📋 GENERATING HARD EVALUATION SUITE")
        self.eval_suite = await self.generate_hard_eval_suite()
        print(f"✅ Created {len(self.eval_suite.unit_tests)} unit tests")
        print(f"✅ Created {len(self.eval_suite.performance_benchmarks)} performance benchmarks")
        print(f"✅ Created {len(self.eval_suite.robustness_tests)} robustness tests")
        print(f"✅ Created {len(self.eval_suite.edge_cases)} edge cases")
        
        # Run evolution loops
        for gen in range(self.config.generations):
            print(f"\n{'='*60}")
            print(f"🔄 QUICK IMPROVED DUAL-LOOP GENERATION {gen + 1}/{self.config.generations}")
            print(f"{'='*60}")
            
            # Artifact evolution loop (μ→λ→μ)
            self.artifact_population = await self.artifact_evolution_loop()
            
            # Meta-evolution loop
            self.scaffold_population = await self.meta_evolution_loop()
            
            # Log evolution
            self.generation = gen + 1
            self._log_generation()
            
            # Check budget
            if self.llm.is_budget_exceeded():
                print("\n⚠️ Budget exceeded, stopping evolution")
                break
        
        # Show final results
        await self.show_quick_improved_final_results()
    
    def _log_generation(self):
        """Log generation statistics."""
        artifact_fitnesses = [a.fitness_score for a in self.artifact_population]
        avg_artifact_fitness = sum(artifact_fitnesses) / len(artifact_fitnesses)
        
        print(f"\n📊 GENERATION {self.generation} SUMMARY:")
        print(f"Artifact population fitness: {avg_artifact_fitness:.3f}")
        print(f"Best artifact fitness: {max(artifact_fitnesses):.3f}")
        print(f"Fitness range: {min(artifact_fitnesses):.3f} - {max(artifact_fitnesses):.3f}")
        
        # Store history
        self.artifact_history.append({
            'generation': self.generation,
            'avg_fitness': avg_artifact_fitness,
            'best_fitness': max(artifact_fitnesses),
            'min_fitness': min(artifact_fitnesses),
            'max_fitness': max(artifact_fitnesses),
            'fitnesses': artifact_fitnesses
        })
    
    async def show_quick_improved_final_results(self):
        """Show comprehensive final results with proper code display."""
        print(f"\n{'='*60}")
        print("🎯 QUICK IMPROVED DUAL-LOOP EVOLUTION COMPLETE - FINAL RESULTS")
        print(f"{'='*60}")
        
        # Show evolution history
        print("\n📈 EVOLUTION HISTORY:")
        for gen in self.artifact_history:
            print(f"Generation {gen['generation']}: avg={gen['avg_fitness']:.3f}, best={gen['best_fitness']:.3f}, range={gen['min_fitness']:.3f}-{gen['max_fitness']:.3f}")
        
        # Show best artifact with proper code display
        best_artifact = max(self.artifact_population, key=lambda x: x.fitness_score)
        print(f"\n🏆 BEST EVOLVED ARTIFACT:")
        print(f"Fitness: {best_artifact.fitness_score:.3f}")
        print(f"Generation: {best_artifact.metadata.get('generation', 'unknown')}")
        print(f"ID: {best_artifact.id}")
        print(f"\n📝 EVOLVED CODE:")
        print("="*60)
        print(best_artifact.content)
        print("="*60)
        
        # Test the evolved code
        print(f"\n🧪 TESTING THE EVOLVED CONVERTER:")
        try:
            # Use fresh namespace for testing
            fresh_globals = {}
            exec(best_artifact.content, fresh_globals)
            
            if 'convert_markdown_to_html' in fresh_globals:
                func = fresh_globals['convert_markdown_to_html']
                
                # Test some basic cases
                test_cases = [
                    ("# Hello", "Hello"),
                    ("**bold**", "bold"),
                    ("*italic*", "italic"),
                    ("- item", "item")
                ]
                
                print("Test Results:")
                for input_text, expected in test_cases:
                    try:
                        result = func(input_text)
                        print(f"  Input: {repr(input_text)}")
                        print(f"  Output: {repr(result)}")
                        print(f"  Expected: {repr(expected)}")
                        print()
                    except Exception as e:
                        print(f"  Error testing {repr(input_text)}: {e}")
            else:
                print("❌ Function 'convert_markdown_to_html' not found in evolved code")
        except Exception as e:
            print(f"❌ Error testing evolved code: {e}")
        
        # Show evolved scaffold
        print(f"\n🧠 EVOLVED SCAFFOLD COMPONENTS:")
        for component_type, components in self.scaffold_population.items():
            print(f"\n{component_type.upper()}:")
            for i, component in enumerate(components):
                print(f"  {i+1}. {component}")
        
        # Show cost stats
        cost_stats = self.llm.get_cost_stats()
        print(f"\n💰 COST STATISTICS:")
        print(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
        print(f"Total requests: {cost_stats.get('total_requests', 0)}")
        
        # Show improvement summary
        initial_fitness = 0.5  # Baseline
        final_fitness = best_artifact.fitness_score
        improvement = final_fitness - initial_fitness
        print(f"\n📊 IMPROVEMENT SUMMARY:")
        print(f"Initial fitness: {initial_fitness:.3f}")
        print(f"Final fitness: {final_fitness:.3f}")
        print(f"Total improvement: {improvement:+.3f} ({improvement/initial_fitness*100:+.1f}%)")

async def main():
    """Run the quick improved dual-loop evolution demo."""
    print("🎭 QUICK IMPROVED DUAL-LOOP EVOLUTION DEMO")
    print("="*60)
    print("μ→λ Evolution + Meta-Evolution (Small Environment)")
    print("="*60)
    
    # Define the task
    task = TaskDefinition(
        outcome="Convert Markdown to HTML, passing all spec tests in ≤50ms",
        stubs=[
            "def parse_markdown(text) -> AST: ...",
            "def render_html(ast) -> str: ..."
        ],
        initial_prompts=[
            "Write convert(text) using parse_markdown/render_html.",
            "Optimize this converter for speed."
        ],
        performance_target=50.0
    )
    
    # Initialize system with small config
    config = EvolutionConfig(
        generations=2,  # REDUCED
        artifact_population_size=2,  # REDUCED
        artifact_children_per_parent=2,  # REDUCED
        max_cost=20.0  # REDUCED
    )
    system = QuickImprovedDualLoopSystem(task, config)
    
    # Run quick improved dual-loop evolution
    await system.run_quick_improved_dual_loop_evolution()

if __name__ == "__main__":
    asyncio.run(main()) 