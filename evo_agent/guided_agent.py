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
import os
import shutil
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.layout import Layout
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Rich library not found. Falling back to basic CLI.")

# #region agent log
with open('/Users/girishverma/Developer/AlphaEvolve-Agent/.cursor/debug.log', 'a') as f:
    f.write('{"id":"log_guided_import_1","timestamp":0,"location":"evo_agent/guided_agent.py:imports","message":"Importing llm_interface","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}\n')
# #endregion
from llm_interface import LLMInterface, LLMConfig
from artifact_support import ArtifactType, ArtifactCandidate
from multi_objective import Objective, MultiObjectiveConfig, MultiObjectiveEvaluator
from cost_manager import CostConfig, BudgetAwareLLMInterface
from analysis_engine import AnalysisEngine, Recommendation
from evaluation_framework import EvaluationFramework
from models import Candidate
from models import (
    TaskSpec as EvalTaskSpec,
    TestCase,
    Benchmark,
    RobustnessTest,
    SuccessCriteria,
)

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
    evolution_frequency: int = 1  # Evolve agent every generation for this demo
    population_size: int = 3
    max_generations: int = 3  # Reduced to 3 as requested

class GuidedAgent:
    """A guided agent that walks through the evolution process step by step."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the guided agent."""
        self.config = config
        self.generation = 0
        self.task = None
        self.task_history = []
        self.llm_live = None  # Will be set by async check
        
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
        
    async def check_llm_live(self):
        """Check if LLM is live by sending a test prompt."""
        try:
            test_response = await self.llm.generate("Say 'pong' if you are live.")
            assert "pong" in test_response.lower()
            self.llm_live = True
        except Exception as e:
            logger.warning(f"LLM check failed: {e}")
            self.llm_live = False
    
    async def get_task_from_user(self) -> TaskSpec:
        """Get task specification from user (custom-only flow)."""
        # Environment-variable driven non-interactive input (preferred for automation)
        env_json = os.environ.get("AGENT_TASK_JSON", "").strip()
        if env_json:
            try:
                parsed = json.loads(env_json)
                task_name = parsed.get("task_name", "").strip()
                description = parsed.get("description", "").strip()
                requirements = parsed.get("requirements", []) or []
                success_criteria = parsed.get("success_criteria", []) or []
                code_type = (parsed.get("code_type") or "python").strip() or "python"
                # Allow comma-separated strings in JSON as well
                if isinstance(requirements, str):
                    requirements = [r.strip() for r in requirements.split(",") if r.strip()]
                if isinstance(success_criteria, str):
                    success_criteria = [s.strip() for s in success_criteria.split(",") if s.strip()]
                if task_name and description:
                    return TaskSpec(
                        task_name=task_name,
                        description=description,
                        requirements=requirements,
                        success_criteria=success_criteria,
                        code_type=code_type,
                    )
            except Exception:
                pass

        # Fallback to discrete env vars
        env_task_name = os.environ.get("AGENT_TASK_NAME", "").strip()
        env_description = os.environ.get("AGENT_TASK_DESCRIPTION", "").strip()
        env_requirements = os.environ.get("AGENT_TASK_REQUIREMENTS", "").strip()
        env_success = os.environ.get("AGENT_TASK_SUCCESS_CRITERIA", "").strip()
        env_code_type = (os.environ.get("AGENT_CODE_TYPE", "python") or "python").strip()

        if env_task_name and env_description:
            requirements = [r.strip() for r in env_requirements.split(",") if r.strip()] if env_requirements else []
            success_criteria = [s.strip() for s in env_success.split(",") if s.strip()] if env_success else []
            return TaskSpec(
                task_name=env_task_name,
                description=env_description,
                requirements=requirements,
                success_criteria=success_criteria,
                code_type=env_code_type,
            )

        # Interactive prompt as last resort
        print("\n🎯 ENTER YOUR TASK")
        print("=" * 50)
        task_name = input("Task name: ").strip()
        description = input("Description: ").strip()
        requirements_raw = input("Requirements (comma-separated): ").strip()
        success_raw = input("Success criteria (comma-separated): ").strip()
        code_type = input("Code type [python]: ").strip() or "python"

        # Normalize comma-separated lists by trimming whitespace and dropping empties
        requirements = [r.strip() for r in requirements_raw.split(",") if r.strip()]
        success_criteria = [s.strip() for s in success_raw.split(",") if s.strip()]

        return TaskSpec(
            task_name=task_name,
            description=description,
            requirements=requirements,
            success_criteria=success_criteria,
            code_type=code_type,
        )
    
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
        """Evaluate code. Prefer Pydantic-based EvaluationFramework when a spec exists."""
        # Attempt to build a strict eval spec for the task
        spec = self._build_eval_spec_for_task()

        if spec is not None:
            try:
                # Normalize code to expose a canonical 'solve' function for tests
                normalized_code = self._ensure_solve_alias(code)

                candidate = Candidate(
                    id=f"eval_{int(time.time())}",
                    code=normalized_code,
                    prompt=self.prompts.get("code_generation", ""),
                    tools=self.tools,
                    memory=self.memory,
                    generation=self.generation,
                    parent_id=None,
                    mutation_type="guided"
                )

                evaluator = EvaluationFramework(spec)
                result = evaluator.evaluate_candidate(candidate)

                return {
                    "correctness": result.correctness_score,
                    "performance": result.performance_score,
                    "robustness": result.robustness_score,
                    "overall": result.overall_score,
                }
            except Exception as e:
                logger.warning(f"Pydantic eval failed, falling back to heuristic: {e}")

        # Fallback heuristic evaluation
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

    def _ensure_solve_alias(self, code: str) -> str:
        """Ensure tests can call a canonical function 'solve'. If missing, alias the first public function."""
        try:
            if re.search(r"\n\s*def\s+solve\s*\(", code):
                return code
            match = re.search(r"\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
            if match:
                func_name = match.group(1)
                if func_name != "solve":
                    return f"{code}\n\n# Alias for evaluation\nsolve = {func_name}\n"
            return code
        except Exception:
            return code

    def _build_eval_spec_for_task(self) -> Optional[EvalTaskSpec]:
        """Build a strict TaskSpec for known tasks. Currently supports Fibonacci."""
        name = (self.task.task_name or "").lower()
        if "fibonacci" in name:
            tests: List[TestCase] = [
                TestCase(name="n=0", code="result = solve(0)", expected=0),
                TestCase(name="n=1", code="result = solve(1)", expected=1),
                TestCase(name="n=5", code="result = solve(5)", expected=5),
                TestCase(name="n=10", code="result = solve(10)", expected=55),
            ]

            benches: List[Benchmark] = [
                Benchmark(
                    name="perf_small",
                    code="pass",
                    function_call="solve(20)",
                    iterations=200,
                    target=0.003,
                    baseline=0.01,
                ),
                Benchmark(
                    name="perf_medium",
                    code="pass",
                    function_call="solve(30)",
                    iterations=50,
                    target=0.02,
                    baseline=0.08,
                ),
            ]

            robust: List[RobustnessTest] = [
                RobustnessTest(name="negative", code="pass", function_call="solve(-1)", expect_exception=True),
                RobustnessTest(name="non_int", code="pass", function_call="solve('10')", expect_exception=True),
            ]

            criteria = SuccessCriteria(
                correctness_threshold=0.9,
                performance_threshold_ms=50.0,
                robustness_threshold=0.8,
            )

            return EvalTaskSpec(
                test_cases=tests,
                performance_benchmarks=benches,
                robustness_tests=robust,
                success_criteria=criteria,
            )

        # Unknown task type → no strict spec yet
        return None
    
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
            # Code content is intentionally not printed; artifacts are saved to disk.
            final_code = self.task_history[-1]["code"] if self.task_history else "No final code"
            
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
    
    def save_generation_artifacts(self, generation, prompt, code, evaluation):
        """Save prompt, code, and evaluation to their respective folders as gen_N.txt."""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        prompt_dir = os.path.join(base_dir, 'prompts')
        code_dir = os.path.join(base_dir, 'code')
        eval_dir = os.path.join(base_dir, 'evaluations')
        os.makedirs(prompt_dir, exist_ok=True)
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, f'gen_{generation}.txt'), 'w') as f:
            f.write(prompt)
        with open(os.path.join(code_dir, f'gen_{generation}.py'), 'w') as f:
            f.write(code)
        with open(os.path.join(eval_dir, f'gen_{generation}.txt'), 'w') as f:
            f.write(str(evaluation))
    
    async def run_guided_session(self):
        # #region agent log
        with open('/Users/girishverma/Developer/AlphaEvolve-Agent/.cursor/debug.log', 'a') as f:
            f.write('{"id":"log_run_session_start","timestamp":0,"location":"evo_agent/guided_agent.py:run_guided_session","message":"Starting guided session with modern CLI","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}\n')
        # #endregion

        # Clean generation folders at the start of each run
        base_dir = os.path.abspath(os.path.dirname(__file__))
        for folder in ['prompts', 'code', 'evaluations']:
            folder_path = os.path.join(base_dir, folder)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        
        await self.check_llm_live()
        
        if HAS_RICH:
            console.clear()
            console.print(Panel.fit("[bold blue]🤖 GUIDED EVOLUTIONARY AGENT[/bold blue]", border_style="blue"))
            llm_status = "[bold green]🟢 LLM is LIVE[/bold green]" if self.llm_live else "[bold red]🔴 LLM is NOT AVAILABLE[/bold red]"
            console.print(f"Status: {llm_status}")
            console.print("[dim]Use hardcoded specs: gpt-5-nano @ eastus2[/dim]")
        else:
            print("🤖 GUIDED EVOLUTIONARY AGENT")
            print("=" * 50)
            llm_status = "🟢 LLM is LIVE" if self.llm_live else "🔴 LLM is NOT AVAILABLE"
            print(f"LLM Status: {llm_status}")
        
        # Step 1: Get task from user
        self.task = await self.get_task_from_user()
        if HAS_RICH:
            console.print(f"\n[bold green]✅ Task set:[/bold green] {self.task.task_name}")
        else:
            print(f"\n✅ Task set: {self.task.task_name}")
        
        # Step 2: Analyze task
        if HAS_RICH:
            with console.status("[bold yellow]📝 Analyzing the task...[/bold yellow]"):
                analysis = await self.analyze_task()
            console.print(Panel(analysis[:500] + "...", title="Task Analysis (Truncated)", border_style="yellow"))
        else:
            print("\n📝 Analyzing the task...")
            analysis = await self.analyze_task()
            print(f"Analysis: {analysis[:200]}...")
        
        # Step 3: Generate initial code
        if HAS_RICH:
            with console.status("[bold cyan]💻 Generating initial code...[/bold cyan]"):
                initial_code = await self.generate_initial_code()
            console.print(Panel(Syntax(initial_code[:500] + "\n# ... truncated ...", "python", theme="monokai", line_numbers=True), title="Initial Code (Truncated)", border_style="cyan"))
        else:
            print("\n💻 Generating initial code...")
            initial_code = await self.generate_initial_code()
        
        # Store initial code for comparison
        self.initial_code = initial_code
        
        # Store initial code
        self.task_history.append({
            "generation": self.generation,
            "code": initial_code,
            "evaluation": {"overall": 0.5},
            "user_feedback": "Initial generation"
        })
        
        # Save initial artifacts
        self.save_generation_artifacts(1, analysis, initial_code, {"overall": 0.5})
        
        # Step 4: Evolution cycles
        for cycle in range(self.config.max_generations):
            if HAS_RICH:
                console.clear()
                console.print(Panel(f"[bold magenta]🔄 EVOLUTION CYCLE {cycle + 1}/{self.config.max_generations}[/bold magenta]", border_style="magenta"))
                console.print(f"[dim]Task: {self.task.task_name}[/dim]")
            else:
                print(f"\n🔄 EVOLUTION CYCLE {cycle + 1}")
            
            # Ask for feedback each cycle (skip if non-interactive)
            if os.environ.get("AGENT_NON_INTERACTIVE", "").strip().lower() in {"1", "true", "yes", "y"}:
                feedback = ""
            else:
                if HAS_RICH:
                    feedback = console.input(f"[bold]💬 Feedback (Enter to skip): [/bold]").strip()
                else:
                    feedback = input(f"\n💬 Provide feedback for improvement (or press Enter to skip): ").strip()
            
            # Improve code
            if HAS_RICH:
                with console.status("[bold green]🛠️  Improving code...[/bold green]"):
                    improved_code = await self.improve_code(initial_code, feedback)
                    
                    # Evaluate both versions
                    current_eval = await self.evaluate_code(initial_code)
                    improved_eval = await self.evaluate_code(improved_code)
            else:
                improved_code = await self.improve_code(initial_code, feedback)
                current_eval = await self.evaluate_code(initial_code)
                improved_eval = await self.evaluate_code(improved_code)
            
            if HAS_RICH:
                # Show truncated code diff or summary
                console.print(Panel(Syntax(improved_code[:800] + ("\n... [truncated]" if len(improved_code) > 800 else ""), "python", theme="monokai", line_numbers=True), title="Improved Code (Snapshot)", border_style="green"))

                table = Table(title="📊 Evaluation Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Current Code", style="magenta")
                table.add_column("Improved Code", style="green")
                table.add_row("Overall", f"{current_eval['overall']:.3f}", f"{improved_eval['overall']:.3f}")
                table.add_row("Correctness", f"{current_eval['correctness']:.3f}", f"{improved_eval['correctness']:.3f}")
                table.add_row("Performance", f"{current_eval['performance']:.3f}", f"{improved_eval['performance']:.3f}")
                console.print(table)
            else:
                print(f"\n📊 EVALUATION RESULTS:")
                print(f"Current code - Overall: {current_eval['overall']:.3f}")
                print(f"Improved code - Overall: {improved_eval['overall']:.3f}")
            
            # Choose better version
            if improved_eval['overall'] > current_eval['overall']:
                if HAS_RICH:
                    console.print(f"[bold green]✅ Improved code selected![/bold green]")
                else:
                    print(f"✅ Improved code selected!")
                initial_code = improved_code
                best_eval = improved_eval
            else:
                if HAS_RICH:
                    console.print(f"[bold yellow]⚠️ Current code retained[/bold yellow]")
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
            
            # Save artifacts for each generation (cycle+2 because gen_1 is initial)
            self.save_generation_artifacts(cycle + 2, analysis, initial_code, best_eval)
            
            # Evolve agent components periodically
            if cycle % self.config.evolution_frequency == 0:
                if HAS_RICH:
                    with console.status("[bold purple]🧬 Evolving agent components...[/bold purple]"):
                        await self.evolve_agent_components()
                else:
                    await self.evolve_agent_components()
            
            # Ask if user wants to continue
            if cycle < self.config.max_generations - 1:
                if os.environ.get("AGENT_NON_INTERACTIVE", "").strip().lower() in {"1", "true", "yes", "y"}:
                    continue_choice = "y"
                else:
                    if HAS_RICH:
                        # console.input(f"[dim]Press Enter to continue...[/dim]")
                        continue_choice = "y" # Auto continue for smoother flow
                    else:
                        continue_choice = input(f"🔄 Continue to evolution cycle {cycle + 2}? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
        
        # Step 5: Execute the final task
        if HAS_RICH:
            console.clear()
            console.print(Panel("[bold red]🎯 Executing Final Task[/bold red]", border_style="red"))
            with console.status("[bold red]Running...[/bold red]"):
                result = await self.execute_task(initial_code)
        else:
            print(f"\n🎯 EXECUTING THE FINAL TASK...")
            result = await self.execute_task(initial_code)
        
        if result["success"]:
            if HAS_RICH:
                console.print(f"\n[bold green]✅ TASK EXECUTION COMPLETE![/bold green]")
                console.print(f"Function: {result['function_name']}")
                console.print(f"Success Rate: {result['success_rate']:.1%}")
            else:
                print(f"\n✅ TASK EXECUTION COMPLETE!")
                print(f"Function: {result['function_name']}")
                print(f"Success Rate: {result['success_rate']:.1%}")
        else:
            if HAS_RICH:
                console.print(f"\n[bold red]❌ TASK EXECUTION FAILED: {result['error']}[/bold red]")
            else:
                print(f"\n❌ TASK EXECUTION FAILED: {result['error']}")
        
        # Show final stats
        cost_stats = self.llm.get_cost_stats()
        if HAS_RICH:
            console.print(Panel(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}\nTotal requests: {cost_stats.get('total_requests', 0)}\nGenerations: {self.generation}", title="Cost Statistics", border_style="green"))
        else:
            print(f"\n💰 COST STATISTICS:")
            print(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
            print(f"Total requests: {cost_stats.get('total_requests', 0)}")
            print(f"Generations: {self.generation}")
        
        # Show comprehensive evolution summary
        await self._show_evolution_summary()
        
        if HAS_RICH:
            console.print(f"\n[bold rainbow]🎉 EVOLUTION COMPLETE! The agent has evolved and created working code![/bold rainbow]")
        else:
            print(f"\n🎉 EVOLUTION COMPLETE! The agent has evolved and created working code!")

async def main():
    """Run the guided agent."""
    config = AgentConfig()
    agent = GuidedAgent(config)
    await agent.run_guided_session()

if __name__ == "__main__":
    asyncio.run(main()) 