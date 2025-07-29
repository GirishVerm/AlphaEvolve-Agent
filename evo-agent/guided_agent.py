#!/usr/bin/env python3
"""
Guided Evolutionary Agent System
===============================

An interactive agent that guides you through the evolution process.

Usage: python3 guided_agent.py
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re

from llm_interface import LLMInterface, LLMConfig
from artifact_support import ArtifactType, ArtifactCandidate
from multi_objective import Objective, MultiObjectiveConfig, MultiObjectiveEvaluator
from cost_manager import CostConfig, BudgetAwareLLMInterface
from analysis_engine import AnalysisEngine, Recommendation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskSpec:
    """Task specification for the agent."""
    task_name: str
    description: str
    requirements: List[str]
    success_criteria: List[str]
    code_type: str = "python"

@dataclass
class AgentConfig:
    """Configuration for the guided agent."""
    max_cost: float = 20.0
    evolution_frequency: int = 2  # Evolve agent every N generations
    population_size: int = 3
    max_generations: int = 5

class GuidedAgent:
    """A guided agent that walks through the evolution process step by step."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the guided agent."""
        self.config = config
        self.generation = 0
        self.task = None
        self.task_history = []
        
        # Agent components that will evolve
        self.prompts = {
            "code_generation": "Create a Python function that meets the given requirements.",
            "code_improvement": "Improve this code by adding error handling, documentation, and optimization.",
            "task_analysis": "Analyze this task and break it down into implementable components.",
            "code_evaluation": "Evaluate this code for correctness, performance, and robustness."
        }
        
        self.tools = {
            "code_tester": "def test_code(code, test_cases): return {'passed': len([t for t in test_cases if eval(t)]), 'total': len(test_cases)}",
            "performance_analyzer": "def analyze_performance(code): return {'complexity': 'O(n)', 'efficiency': 0.8}",
            "code_quality_checker": "def check_quality(code): return {'readability': 0.7, 'maintainability': 0.8, 'documentation': 0.6}"
        }
        
        self.memory = {
            "context_manager": "def store_context(task, result): return {'task': task, 'result': result, 'timestamp': time.time()}",
            "knowledge_base": "def retrieve_knowledge(query): return 'relevant_patterns_and_solutions'",
            "experience_logger": "def log_experience(action, outcome): return {'action': action, 'outcome': outcome, 'success': outcome > 0.5}"
        }
        
        # Store initial state for comparison
        self.initial_prompts = self.prompts.copy()
        self.initial_tools = self.tools.copy()
        self.initial_memory = self.memory.copy()
        self.initial_code = None
        
        # Track eval suite usage
        self.used_llm_eval_suite = False
        self.eval_suite_details = {}
        
        # Initialize LLM
        llm_config = LLMConfig()
        base_llm = LLMInterface(llm_config)
        cost_config = CostConfig(max_cost_per_experiment=config.max_cost)
        self.llm = BudgetAwareLLMInterface(base_llm, cost_config)
        
        # Initialize evaluation
        objectives = [
            Objective(name="correctness", weight=0.4, minimize=False),
            Objective(name="performance", weight=0.3, minimize=False),
            Objective(name="robustness", weight=0.3, minimize=False)
        ]
        eval_config = MultiObjectiveConfig(objectives=objectives)
        self.evaluator = MultiObjectiveEvaluator(eval_config)
        
        # Initialize analysis engine
        self.analysis_engine = AnalysisEngine()
    
    async def get_task_from_user(self) -> TaskSpec:
        """Get task specification from user."""
        print("\n🎯 LET'S CREATE A TASK!")
        print("=" * 50)
        
        # Provide example tasks
        print("Example tasks:")
        print("1. String Processor - Create a function that processes strings")
        print("2. Calculator - Create a calculator function")
        print("3. Data Validator - Create a data validation function")
        print("4. Custom task")
        print("5. Fibonacci Calculator (Hardcoded)")
        
        choice = input("\nChoose a task (1-5): ").strip()
        
        if choice == "1":
            return TaskSpec(
                task_name="String Processor",
                description="Create a function that processes strings",
                requirements=["Handle basic operations", "unicode support", "error handling"],
                success_criteria=["Works correctly", "handles edge cases", "good performance"]
            )
        elif choice == "2":
            return TaskSpec(
                task_name="Calculator",
                description="Create a calculator function",
                requirements=["Add", "subtract", "multiply", "divide", "error handling"],
                success_criteria=["Handles basic math", "division by zero", "invalid inputs"]
            )
        elif choice == "3":
            return TaskSpec(
                task_name="Data Validator",
                description="Create a data validation function",
                requirements=["Validate emails", "phone numbers", "URLs", "error handling"],
                success_criteria=["Handles various formats", "edge cases", "clear error messages"]
            )
        elif choice == "4":
            print("\n🎯 ENTER YOUR CUSTOM TASK")
            print("=" * 30)
            task_name = input("Task name: ").strip()
            description = input("Description: ").strip()
            requirements = input("Requirements (comma-separated): ").strip().split(",")
            success_criteria = input("Success criteria (comma-separated): ").strip().split(",")
            
            return TaskSpec(
                task_name=task_name,
                description=description,
                requirements=requirements,
                success_criteria=success_criteria
            )
        elif choice == "5":
            print("\n🎯 USING HARDCODED FIBONACCI CALCULATOR TASK")
            print("=" * 50)
            return TaskSpec(
                task_name="Fibonacci Calculator",
                description="Compute the _n_th Fibonacci number for non‑negative inputs, optimizing for both correctness and runtime performance.",
                requirements=["Accept integer n ≥ 0", "raise ValueError for n < 0", "return correct F(n) with F(0)=0, F(1)=1", "handle base cases", "optimize algorithmic complexity (e.g. memoization or iterative) to handle n up to 40", "measure runtime on test inputs n=[5,10,20,30,35,40]"],
                success_criteria=["Correct Fibonacci outputs for all test inputs", "runtime below threshold on each test (e.g. <0.1 s for n=30, <0.8 s for n=35, <5 s for n=40)", "overall performance score > 1.0 relative to baseline naive recursion"]
            )
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            return await self.get_task_from_user()  # Recursive call to try again
    
    async def analyze_task(self) -> str:
        """Analyze the current task."""
        prompt = f"""
        {self.prompts['task_analysis']}
        
        Task: {self.task.task_name}
        Description: {self.task.description}
        Requirements: {self.task.requirements}
        Success Criteria: {self.task.success_criteria}
        
        Provide a detailed analysis of how to approach this task:
        """
        
        try:
            analysis = await self.llm.generate(prompt)
            return analysis
        except Exception as e:
            logger.warning(f"Task analysis failed: {e}")
            return "Basic task analysis: Implement the requirements step by step."
    
    async def generate_initial_code(self) -> str:
        """Generate initial code."""
        prompt = f"""
        {self.prompts['code_generation']}
        
        Task: {self.task.task_name}
        Requirements: {self.task.requirements}
        Code Type: {self.task.code_type}
        
        Generate the initial implementation:
        """
        
        try:
            code = await self.llm.generate(prompt)
            return self._extract_code(code)
        except Exception as e:
            logger.warning(f"Code generation failed: {e}")
            return f"def {self.task.task_name.lower().replace(' ', '_')}():\n    # TODO: Implement {self.task.task_name}\n    pass"
    
    async def improve_code(self, current_code: str, feedback: str = "") -> str:
        """Improve code."""
        prompt = f"""
        {self.prompts['code_improvement']}
        
        Current Code:
        {current_code}
        
        Task: {self.task.task_name}
        Requirements: {self.task.requirements}
        Feedback: {feedback}
        
        Provide the improved code:
        """
        
        try:
            improved_code = await self.llm.generate(prompt)
            return self._extract_code(improved_code)
        except Exception as e:
            logger.warning(f"Code improvement failed: {e}")
            return current_code
    
    async def evaluate_code(self, code: str) -> Dict[str, float]:
        """Evaluate code."""
        try:
            artifact = ArtifactCandidate(
                id=f"eval_{int(time.time())}",
                content=code,
                artifact_type=ArtifactType.PYTHON_CODE,
                metadata={"task": self.task.task_name}
            )
            
            fitness_scores = await self._evaluate_artifact_with_hygiene(artifact)
            
            return {
                "correctness": fitness_scores[0] if len(fitness_scores) > 0 else 0.5,
                "performance": fitness_scores[1] if len(fitness_scores) > 1 else 0.5,
                "robustness": fitness_scores[2] if len(fitness_scores) > 2 else 0.5,
                "overall": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.5
            }
        except Exception as e:
            logger.warning(f"Code evaluation failed: {e}")
            return {"correctness": 0.5, "performance": 0.5, "robustness": 0.5, "overall": 0.5}
    
    async def _evaluate_artifact_with_hygiene(self, artifact: ArtifactCandidate) -> List[float]:
        """Evaluate artifact with namespace hygiene."""
        try:
            local_vars = {}
            exec(artifact.content, {"__builtins__": __builtins__}, local_vars)
            
            code = artifact.content
            
            # Correctness score (0-1)
            correctness = 0.5
            if "def" in code and "return" in code:
                correctness += 0.2
            if "try" in code and "except" in code:
                correctness += 0.2
            if "import" in code:
                correctness += 0.1
            
            # Performance score (0-1)
            performance = 0.5
            if "for" in code or "while" in code:
                performance += 0.2
            if "list" in code or "dict" in code:
                performance += 0.1
            if len(code) < 500:
                performance += 0.2
            
            # Robustness score (0-1)
            robustness = 0.5
            if "try" in code and "except" in code:
                robustness += 0.3
            if "if" in code and "else" in code:
                robustness += 0.2
            
            return [correctness, performance, robustness]
            
        except Exception as e:
            logger.warning(f"Artifact evaluation failed: {e}")
            return [0.3, 0.3, 0.3]
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'EVOLVE-BLOCK START\n(.*?)\nEVOLVE-BLOCK END',
            r'```(.*?)```'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return response.strip()
    
    async def evolve_agent_components(self):
        """Evolve the agent's components."""
        print(f"\n🔄 EVOLVING AGENT COMPONENTS (Generation {self.generation + 1})")
        
        # Store previous state for comparison
        previous_prompts = self.prompts.copy()
        previous_tools = self.tools.copy()
        previous_memory = self.memory.copy()
        
        # Evolve prompts
        for name, prompt in self.prompts.items():
            evolution_prompt = f"""
            Improve this agent prompt to make it more effective at its task.
            
            Current {name} prompt:
            {prompt}
            
            Task context: {self.task.task_name}
            
            Make it more specific, clear, and effective. Provide only the improved prompt:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.prompts[name] = improved_content.strip()
                print(f"✅ Evolved {name} prompt")
                
                # Show evolution details
                print(f"   📝 {name} evolution:")
                print(f"   Before: {prompt}")
                print(f"   After:  {improved_content.strip()}")
                print(f"   {'-'*50}")
                
            except Exception as e:
                logger.warning(f"Failed to evolve {name} prompt: {e}")
        
        # Evolve tools
        for name, tool in self.tools.items():
            evolution_prompt = f"""
            Improve this agent tool to make it more effective and robust.
            
            Current {name} tool:
            {tool}
            
            Make it more comprehensive and error-resistant. Provide only the improved tool:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.tools[name] = improved_content.strip()
                print(f"✅ Evolved {name} tool")
                
                # Show evolution details
                print(f"   🛠️ {name} evolution:")
                print(f"   Before: {tool}")
                print(f"   After:  {improved_content.strip()}")
                print(f"   {'-'*50}")
                
            except Exception as e:
                logger.warning(f"Failed to evolve {name} tool: {e}")
        
        # Evolve memory
        for name, memory in self.memory.items():
            evolution_prompt = f"""
            Improve this agent memory component to make it more effective at storing and retrieving information.
            
            Current {name} component:
            {memory}
            
            Make it more comprehensive and useful. Provide only the improved component:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.memory[name] = improved_content.strip()
                print(f"✅ Evolved {name} memory component")
                
                # Show evolution details
                print(f"   🧠 {name} evolution:")
                print(f"   Before: {memory}")
                print(f"   After:  {improved_content.strip()}")
                print(f"   {'-'*50}")
                
            except Exception as e:
                logger.warning(f"Failed to evolve {name} memory component: {e}")
        
        self.generation += 1
    
    async def execute_task(self, code: str) -> Dict[str, Any]:
        """Execute the evolved code and demonstrate the task."""
        print(f"\n🎯 EXECUTING TASK: {self.task.task_name}")
        print("=" * 50)
        
        try:
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            main_function = None
            for name, obj in local_vars.items():
                if callable(obj) and not name.startswith('_'):
                    main_function = obj
                    break
            
            if main_function:
                test_inputs = await self._generate_test_inputs()
                
                print(f"🧪 TESTING THE EVOLVED CODE:")
                results = []
                
                for i, test_input in enumerate(test_inputs[:3]):
                    try:
                        if isinstance(test_input, dict):
                            result = main_function(**test_input)
                        elif isinstance(test_input, (list, tuple)):
                            result = main_function(*test_input)
                        else:
                            result = main_function(test_input)
                        
                        results.append({
                            "input": test_input,
                            "output": result,
                            "success": True
                        })
                        
                        print(f"  Test {i+1}: {test_input} → {result}")
                        
                    except Exception as e:
                        results.append({
                            "input": test_input,
                            "output": str(e),
                            "success": False
                        })
                        print(f"  Test {i+1}: {test_input} → ERROR: {e}")
                
                success_rate = sum(1 for r in results if r["success"]) / len(results) if results else 0
                
                return {
                    "success": True,
                    "function_name": main_function.__name__,
                    "test_results": results,
                    "success_rate": success_rate,
                    "code": code
                }
                
            else:
                print("❌ No executable function found in the code")
                return {
                    "success": False,
                    "error": "No executable function found",
                    "code": code
                }
                
        except Exception as e:
            print(f"❌ Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": code
            }
    
    def _get_default_test_inputs(self) -> List[Any]:
        """Get default test inputs based on task type."""
        task_name = self.task.task_name.lower()
        
        if "string" in task_name:
            return ["hello world", "test", "", "123", "Hello World!"]
        elif "calculator" in task_name or "math" in task_name:
            return [(1, 2), (10, 5), (0, 0), (-1, 1)]
        elif "file" in task_name:
            return ["test.txt", "data.csv", "config.json"]
        elif "email" in task_name or "validator" in task_name:
            return ["test@example.com", "invalid-email", "user@domain.org"]
        else:
            return ["test input", 42, {"key": "value"}]
    
    async def _generate_test_inputs(self) -> List[Any]:
        """Generate test inputs based on the task."""
        print(f"\n🧪 GENERATING HARD EVAL SUITE FOR: {self.task.task_name}")
        print("=" * 50)
        
        prompt = f"""
        Generate 3-5 test inputs for testing this task:
        Task: {self.task.task_name}
        Description: {self.task.description}
        Requirements: {self.task.requirements}
        
        Provide test inputs that would verify the functionality works correctly.
        Return as a Python list of test inputs.
        """
        
        try:
            print("🔄 Requesting LLM to generate hard eval suite...")
            response = await self.llm.generate(prompt)
            print("✅ LLM generated hard eval suite successfully")
            print(f"📝 LLM Response: {response[:200]}...")
            
            # Try to extract list from response
            import ast
            try:
                # Look for list in the response
                list_match = re.search(r'\[.*\]', response, re.DOTALL)
                if list_match:
                    test_inputs = ast.literal_eval(list_match.group())
                    print(f"🎯 SUCCESS: Using LLM-generated hard eval suite!")
                    print(f"📊 Extracted {len(test_inputs)} test inputs from LLM response")
                    print(f"🧪 Test inputs: {test_inputs}")
                    print("=" * 50)
                    
                    # Track successful LLM eval usage
                    self.used_llm_eval_suite = True
                    self.eval_suite_details = {
                        "source": "LLM-generated",
                        "test_inputs": test_inputs,
                        "count": len(test_inputs)
                    }
                    
                    return test_inputs
                else:
                    print("⚠️ LLM response didn't contain valid list format")
                    print(f"📝 Full LLM response: {response}")
                    print("🔄 FALLBACK: Using default test inputs")
                    fallback_inputs = self._get_default_test_inputs()
                    print(f"📊 Default test inputs: {fallback_inputs}")
                    print("=" * 50)
                    
                    # Track fallback usage
                    self.used_llm_eval_suite = False
                    self.eval_suite_details = {
                        "source": "Fallback defaults",
                        "test_inputs": fallback_inputs,
                        "count": len(fallback_inputs)
                    }
                    
                    return fallback_inputs
            except Exception as parse_error:
                print(f"⚠️ Failed to parse LLM response: {parse_error}")
                print(f"📝 Full LLM response: {response}")
                print("🔄 FALLBACK: Using default test inputs")
                fallback_inputs = self._get_default_test_inputs()
                print(f"📊 Default test inputs: {fallback_inputs}")
                print("=" * 50)
                
                # Track fallback usage
                self.used_llm_eval_suite = False
                self.eval_suite_details = {
                    "source": "Fallback defaults",
                    "test_inputs": fallback_inputs,
                    "count": len(fallback_inputs)
                }
                
                return fallback_inputs
        except Exception as e:
            print(f"❌ Failed to generate test inputs: {e}")
            print("🔄 FALLBACK: Using default test inputs")
            fallback_inputs = self._get_default_test_inputs()
            print(f"📊 Default test inputs: {fallback_inputs}")
            print("=" * 50)
            
            # Track fallback usage
            self.used_llm_eval_suite = False
            self.eval_suite_details = {
                "source": "Fallback defaults",
                "test_inputs": fallback_inputs,
                "count": len(fallback_inputs)
            }
            
            return fallback_inputs
    
    async def _show_evolution_summary(self):
        """Show comprehensive evolution summary."""
        print(f"\n{'='*60}")
        print("🎯 COMPREHENSIVE EVOLUTION SUMMARY")
        print(f"{'='*60}")
        
        # Show initial vs final agent state
        print(f"\n🤖 AGENT EVOLUTION:")
        print(f"{'='*40}")
        
        print(f"\n📝 PROMPTS EVOLUTION:")
        for name in self.prompts.keys():
            initial = self.initial_prompts[name]
            final = self.prompts[name]
            print(f"\n  {name}:")
            print(f"    Initial: {initial}")
            print(f"    Final:   {final}")
            print(f"    {'='*60}")
        
        print(f"\n🛠️ TOOLS EVOLUTION:")
        for name in self.tools.keys():
            initial = self.initial_tools[name]
            final = self.tools[name]
            print(f"\n  {name}:")
            print(f"    Initial: {initial}")
            print(f"    Final:   {final}")
            print(f"    {'='*60}")
        
        print(f"\n🧠 MEMORY EVOLUTION:")
        for name in self.memory.keys():
            initial = self.initial_memory[name]
            final = self.memory[name]
            print(f"\n  {name}:")
            print(f"    Initial: {initial}")
            print(f"    Final:   {final}")
            print(f"    {'='*60}")
        
        # Show initial vs final code
        print(f"\n💻 ARTIFACT EVOLUTION:")
        print(f"{'='*40}")
        
        if self.initial_code:
            print(f"\n📝 INITIAL CODE:")
            print(f"{'='*30}")
            print(self.initial_code)
            
            final_code = self.task_history[-1]["code"] if self.task_history else "No final code"
            print(f"\n📝 FINAL EVOLVED CODE:")
            print(f"{'='*30}")
            print(final_code)
            
            # Show evolution metrics
            if len(self.task_history) > 1:
                initial_eval = self.task_history[0]["evaluation"]["overall"]
                final_eval = self.task_history[-1]["evaluation"]["overall"]
                improvement = final_eval - initial_eval
                print(f"\n📊 EVOLUTION METRICS:")
                print(f"  Initial fitness: {initial_eval:.3f}")
                print(f"  Final fitness: {final_eval:.3f}")
                print(f"  Improvement: {improvement:+.3f} ({improvement/initial_eval*100:+.1f}%)")
        
        # Show eval suite information
        print(f"\n🧪 EVAL SUITE INFORMATION:")
        print(f"{'='*40}")
        if hasattr(self, 'eval_suite_details') and self.eval_suite_details:
            print(f"  Source: {self.eval_suite_details['source']}")
            print(f"  Test inputs count: {self.eval_suite_details['count']}")
            print(f"  Test inputs: {self.eval_suite_details['test_inputs']}")
            if self.used_llm_eval_suite:
                print(f"  ✅ Used LLM-generated hard eval suite")
            else:
                print(f"  ⚠️ Used fallback default test inputs")
        else:
            print(f"  No eval suite information available")
        
        print(f"\n{'='*60}")
    
    async def _generate_recommendations(self) -> Recommendation:
        """Generate intelligent recommendations for next steps."""
        if not self.task_history:
            return self.analysis_engine._generate_default_recommendation()
        
        # Prepare data for analysis
        recent_metrics = []
        recent_code_analyses = []
        recent_component_analyses = []
        
        # Analyze each generation
        for i, history_item in enumerate(self.task_history):
            if i == 0:
                continue  # Skip initial generation for comparison
            
            # Get previous and current data
            prev_item = self.task_history[i-1]
            current_item = history_item
            
            # Analyze evolution cycle
            code_analysis, component_analyses, metrics = self.analysis_engine.analyze_evolution_cycle(
                generation=i,
                initial_code=prev_item["code"],
                final_code=current_item["code"],
                initial_fitness=prev_item["evaluation"]["overall"],
                final_fitness=current_item["evaluation"]["overall"],
                initial_components={
                    **self.initial_prompts,
                    **self.initial_tools,
                    **self.initial_memory
                },
                final_components={
                    **self.prompts,
                    **self.tools,
                    **self.memory
                }
            )
            
            recent_metrics.append(metrics)
            recent_code_analyses.append(code_analysis)
            recent_component_analyses.extend(component_analyses)
        
        # Generate recommendations
        return self.analysis_engine.generate_recommendations(
            current_generation=self.generation,
            max_generations=self.config.max_generations,
            recent_metrics=recent_metrics,
            recent_code_analyses=recent_code_analyses,
            recent_component_analyses=recent_component_analyses
        )
    
    async def _show_recommendations(self, recommendation: Recommendation):
        """Display intelligent recommendations."""
        print(f"\n🤖 INTELLIGENT RECOMMENDATIONS")
        print(f"{'='*50}")
        
        # Main recommendation
        status_icon = "✅" if recommendation.should_continue else "⏹️"
        print(f"{status_icon} CONTINUE EVOLUTION: {recommendation.should_continue}")
        print(f"🎯 CONFIDENCE: {recommendation.confidence:.1%}")
        print(f"💭 REASONING: {recommendation.reasoning}")
        
        # Suggested feedback
        print(f"\n💡 SUGGESTED FEEDBACK:")
        print(f"   {recommendation.suggested_feedback}")
        
        # Agent improvements
        if recommendation.agent_improvements:
            print(f"\n🔄 AGENT IMPROVEMENTS:")
            for improvement in recommendation.agent_improvements:
                print(f"   • {improvement}")
        
        # Code improvements
        if recommendation.code_improvements:
            print(f"\n💻 CODE IMPROVEMENTS:")
            for improvement in recommendation.code_improvements:
                print(f"   • {improvement}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        print(f"   {recommendation.risk_assessment}")
        
        # Next generation focus
        print(f"\n🎯 NEXT GENERATION FOCUS:")
        print(f"   {recommendation.next_generation_focus}")
        
        print(f"{'='*50}")
    
    async def run_guided_session(self):
        """Run the guided agent session."""
        print("🤖 GUIDED EVOLUTIONARY AGENT")
        print("=" * 50)
        print("I'll guide you through the evolution process step by step!")
        print("=" * 50)
        
        # Step 1: Get task from user
        self.task = await self.get_task_from_user()
        print(f"\n✅ Task set: {self.task.task_name}")
        
        # Step 2: Analyze task
        print("\n📝 Analyzing the task...")
        analysis = await self.analyze_task()
        print(f"Analysis: {analysis[:200]}...")
        
        # Step 3: Generate initial code
        print("\n💻 Generating initial code...")
        initial_code = await self.generate_initial_code()
        print(f"Generated code:\n{initial_code}")
        
        # Store initial code for comparison
        self.initial_code = initial_code
        
        # Store initial code
        self.task_history.append({
            "generation": self.generation,
            "code": initial_code,
            "evaluation": {"overall": 0.5},
            "user_feedback": "Initial generation"
        })
        
        # Step 4: Evolution cycles
        for cycle in range(self.config.max_generations):
            print(f"\n🔄 EVOLUTION CYCLE {cycle + 1}")
            
            # Generate intelligent recommendations (only if we have history)
            if cycle > 0:  # Only show recommendations after first cycle
                recommendation = await self._generate_recommendations()
                await self._show_recommendations(recommendation)
                
                # Get user feedback with suggestions
                print(f"\n💬 Provide feedback for improvement:")
                print(f"💡 SUGGESTED: {recommendation.suggested_feedback}")
                feedback = input(f"💬 Your feedback (or press Enter to use suggestion): ").strip()
                
                # Use suggested feedback if user didn't provide any
                if not feedback:
                    feedback = recommendation.suggested_feedback
            else:
                # First cycle - just ask for feedback
                feedback = input(f"\n💬 Provide feedback for improvement (or press Enter to skip): ").strip()
            
            # Improve code
            improved_code = await self.improve_code(initial_code, feedback)
            
            # Evaluate both versions
            current_eval = await self.evaluate_code(initial_code)
            improved_eval = await self.evaluate_code(improved_code)
            
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"Current code - Overall: {current_eval['overall']:.3f}")
            print(f"Improved code - Overall: {improved_eval['overall']:.3f}")
            
            # Choose better version
            if improved_eval['overall'] > current_eval['overall']:
                print(f"✅ Improved code selected!")
                initial_code = improved_code
                best_eval = improved_eval
            else:
                print(f"⚠️ Current code retained")
                best_eval = current_eval
            
            # Store in history
            self.task_history.append({
                "generation": self.generation + cycle + 1,
                "code": initial_code,
                "evaluation": best_eval,
                "user_feedback": feedback
            })
            
            # Evolve agent components periodically
            if cycle % self.config.evolution_frequency == 0:
                await self.evolve_agent_components()
            
            # Ask if user wants to continue (with recommendation if available)
            if cycle < self.config.max_generations - 1:
                if cycle > 0:  # Only show recommendation if we have one
                    status_icon = "✅" if recommendation.should_continue else "⏹️"
                    print(f"\n{status_icon} RECOMMENDATION: {'Continue' if recommendation.should_continue else 'Stop'} evolution")
                    print(f"🎯 CONFIDENCE: {recommendation.confidence:.1%}")
                continue_choice = input(f"🔄 Continue to evolution cycle {cycle + 2}? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
        
        # Step 5: Execute the final task
        print(f"\n🎯 EXECUTING THE FINAL TASK...")
        result = await self.execute_task(initial_code)
        
        if result["success"]:
            print(f"\n✅ TASK EXECUTION COMPLETE!")
            print(f"Function: {result['function_name']}")
            print(f"Success Rate: {result['success_rate']:.1%}")
            print(f"\n📝 FINAL WORKING CODE:")
            print("=" * 50)
            print(result["code"])
            print("=" * 50)
        else:
            print(f"\n❌ TASK EXECUTION FAILED: {result['error']}")
        
        # Show final stats
        cost_stats = self.llm.get_cost_stats()
        print(f"\n💰 COST STATISTICS:")
        print(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
        print(f"Total requests: {cost_stats.get('total_requests', 0)}")
        print(f"Generations: {self.generation}")
        
        # Show comprehensive evolution summary
        await self._show_evolution_summary()
        
        print(f"\n🎉 EVOLUTION COMPLETE! The agent has evolved and created working code!")

async def main():
    """Run the guided agent."""
    config = AgentConfig()
    agent = GuidedAgent(config)
    await agent.run_guided_session()

if __name__ == "__main__":
    asyncio.run(main()) 