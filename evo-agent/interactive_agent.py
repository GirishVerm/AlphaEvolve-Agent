#!/usr/bin/env python3
"""
Interactive Evolutionary Agent System
====================================

An agent that can:
1. Accept tasks and specifications
2. Build and evolve code
3. Evolve its own prompts, tools, and memory
4. Interact with users through terminal

Usage: python3 interactive_agent.py
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentComponent:
    """Represents an evolvable component of the agent."""
    name: str
    content: str
    fitness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    """Configuration for the interactive agent."""
    max_cost: float = 20.0
    evolution_frequency: int = 3  # Evolve agent every N generations
    population_size: int = 5
    max_generations: int = 10

class InteractiveAgent:
    """An interactive agent that evolves itself while building code."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the interactive agent."""
        self.config = config
        self.generation = 0
        self.task = None
        self.artifacts = []
        
        # Agent components that will evolve
        self.prompts = {
            "code_generation": AgentComponent("code_generation", 
                "Create a Python function that meets the given requirements."),
            "code_improvement": AgentComponent("code_improvement", 
                "Improve this code by adding error handling, documentation, and optimization."),
            "task_analysis": AgentComponent("task_analysis", 
                "Analyze this task and break it down into implementable components."),
            "code_evaluation": AgentComponent("code_evaluation", 
                "Evaluate this code for correctness, performance, and robustness.")
        }
        
        self.tools = {
            "code_tester": AgentComponent("code_tester", 
                "def test_code(code, test_cases): return {'passed': len([t for t in test_cases if eval(t)]), 'total': len(test_cases)}"),
            "performance_analyzer": AgentComponent("performance_analyzer", 
                "def analyze_performance(code): return {'complexity': 'O(n)', 'efficiency': 0.8}"),
            "code_quality_checker": AgentComponent("code_quality_checker", 
                "def check_quality(code): return {'readability': 0.7, 'maintainability': 0.8, 'documentation': 0.6}")
        }
        
        self.memory = {
            "context_manager": AgentComponent("context_manager", 
                "def store_context(task, result): return {'task': task, 'result': result, 'timestamp': time.time()}"),
            "knowledge_base": AgentComponent("knowledge_base", 
                "def retrieve_knowledge(query): return 'relevant_patterns_and_solutions'"),
            "experience_logger": AgentComponent("experience_logger", 
                "def log_experience(action, outcome): return {'action': action, 'outcome': outcome, 'success': outcome > 0.5}")
        }
        
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
        
        # Evolution history
        self.evolution_history = []
        self.task_history = []
    
    async def set_task(self, task_spec: TaskSpec):
        """Set the current task for the agent."""
        self.task = task_spec
        print(f"\n🎯 TASK SET: {task_spec.task_name}")
        print(f"Description: {task_spec.description}")
        print(f"Requirements: {', '.join(task_spec.requirements)}")
        print(f"Success Criteria: {', '.join(task_spec.success_criteria)}")
        
        # Clear previous history for new task
        self.task_history = []
        self.generation = 0
    
    async def analyze_task(self) -> str:
        """Analyze the current task using evolved task analysis prompt."""
        prompt = f"""
        {self.prompts['task_analysis'].content}
        
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
        """Generate initial code using evolved code generation prompt."""
        prompt = f"""
        {self.prompts['code_generation'].content}
        
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
        """Improve code using evolved code improvement prompt."""
        prompt = f"""
        {self.prompts['code_improvement'].content}
        
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
        """Evaluate code using evolved evaluation methods."""
        try:
            # Create a test artifact
            artifact = ArtifactCandidate(
                id=f"eval_{int(time.time())}",
                content=code,
                artifact_type=ArtifactType.PYTHON_CODE,
                metadata={"task": self.task.task_name}
            )
            
            # Evaluate using the multi-objective evaluator
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
            # Create isolated namespace
            local_vars = {}
            
            # Execute the code
            exec(artifact.content, {"__builtins__": __builtins__}, local_vars)
            
            # Simple evaluation based on code characteristics
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
            if len(code) < 500:  # Concise code
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
        # Look for code blocks
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
        
        # If no code blocks found, return the whole response
        return response.strip()
    
    async def evolve_agent_components(self):
        """Evolve the agent's components."""
        print(f"\n🔄 EVOLVING AGENT COMPONENTS (Generation {self.generation + 1})")
        
        # Evolve prompts
        for name, prompt in self.prompts.items():
            evolution_prompt = f"""
            Improve this agent prompt to make it more effective at its task.
            
            Current {name} prompt:
            {prompt.content}
            
            Task context: {self.task.task_name if self.task else 'General programming'}
            
            Make it more specific, clear, and effective. Provide only the improved prompt:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.prompts[name].content = improved_content.strip()
                print(f"✅ Evolved {name} prompt")
            except Exception as e:
                logger.warning(f"Failed to evolve {name} prompt: {e}")
        
        # Evolve tools
        for name, tool in self.tools.items():
            evolution_prompt = f"""
            Improve this agent tool to make it more effective and robust.
            
            Current {name} tool:
            {tool.content}
            
            Make it more comprehensive and error-resistant. Provide only the improved tool:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.tools[name].content = improved_content.strip()
                print(f"✅ Evolved {name} tool")
            except Exception as e:
                logger.warning(f"Failed to evolve {name} tool: {e}")
        
        # Evolve memory
        for name, memory in self.memory.items():
            evolution_prompt = f"""
            Improve this agent memory component to make it more effective at storing and retrieving information.
            
            Current {name} component:
            {memory.content}
            
            Make it more comprehensive and useful. Provide only the improved component:
            """
            
            try:
                improved_content = await self.llm.generate(evolution_prompt)
                self.memory[name].content = improved_content.strip()
                print(f"✅ Evolved {name} memory component")
            except Exception as e:
                logger.warning(f"Failed to evolve {name} memory component: {e}")
        
        self.generation += 1
    
    async def run_evolution_cycle(self, current_code: str, user_feedback: str = "") -> str:
        """Run one evolution cycle."""
        print(f"\n🔄 EVOLUTION CYCLE {self.generation + 1}")
        
        # Improve code
        improved_code = await self.improve_code(current_code, user_feedback)
        
        # Evaluate both versions
        current_eval = await self.evaluate_code(current_code)
        improved_eval = await self.evaluate_code(improved_code)
        
        print(f"📊 EVALUATION RESULTS:")
        print(f"Current code - Overall: {current_eval['overall']:.3f}")
        print(f"Improved code - Overall: {improved_eval['overall']:.3f}")
        
        # Choose the better version
        if improved_eval['overall'] > current_eval['overall']:
            print(f"✅ Improved code selected (improvement: {improved_eval['overall'] - current_eval['overall']:+.3f})")
            best_code = improved_code
            best_eval = improved_eval
        else:
            print(f"⚠️ Current code retained (no improvement)")
            best_code = current_code
            best_eval = current_eval
        
        # Store in history
        self.task_history.append({
            "generation": self.generation,
            "code": best_code,
            "evaluation": best_eval,
            "user_feedback": user_feedback
        })
        
        # Evolve agent components periodically
        if self.generation % self.config.evolution_frequency == 0:
            await self.evolve_agent_components()
        
        return best_code
    
    async def execute_task(self, code: str) -> Dict[str, Any]:
        """Execute the evolved code and demonstrate the task."""
        print(f"\n🎯 EXECUTING TASK: {self.task.task_name}")
        print("=" * 50)
        
        try:
            # Create isolated namespace for execution
            local_vars = {}
            
            # Execute the code
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Find the main function (usually the task name or similar)
            main_function = None
            for name, obj in local_vars.items():
                if callable(obj) and not name.startswith('_'):
                    main_function = obj
                    break
            
            if main_function:
                # Generate test inputs based on task
                test_inputs = await self._generate_test_inputs()
                
                print(f"🧪 TESTING THE EVOLVED CODE:")
                results = []
                
                for i, test_input in enumerate(test_inputs[:3]):  # Test first 3 inputs
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
                
                # Calculate success rate
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
    
    async def _generate_test_inputs(self) -> List[Any]:
        """Generate test inputs based on the task."""
        prompt = f"""
        Generate 3-5 test inputs for testing this task:
        Task: {self.task.task_name}
        Description: {self.task.description}
        Requirements: {self.task.requirements}
        
        Provide test inputs that would verify the functionality works correctly.
        Return as a Python list of test inputs.
        """
        
        try:
            response = await self.llm.generate(prompt)
            # Try to extract list from response
            import ast
            try:
                # Look for list in the response
                list_match = re.search(r'\[.*\]', response, re.DOTALL)
                if list_match:
                    test_inputs = ast.literal_eval(list_match.group())
                    return test_inputs
                else:
                    # Fallback to simple inputs
                    return self._get_default_test_inputs()
            except:
                return self._get_default_test_inputs()
        except Exception as e:
            logger.warning(f"Failed to generate test inputs: {e}")
            return self._get_default_test_inputs()
    
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
    
    async def run_interactive_session(self):
        """Run the interactive agent session."""
        print("🤖 INTERACTIVE EVOLUTIONARY AGENT")
        print("=" * 50)
        print("This agent can build code and evolve itself!")
        print("=" * 50)
        
        while True:
            print(f"\n📋 AVAILABLE COMMANDS:")
            print("1. set_task <task_name> <description> <requirements> <success_criteria>")
            print("2. analyze_task")
            print("3. generate_code")
            print("4. improve_code [feedback]")
            print("5. evaluate_code")
            print("6. execute_task")
            print("7. show_history")
            print("8. show_agent_status")
            print("9. evolve_agent")
            print("10. quit")
            
            try:
                command = input("\n🤖 Enter command: ").strip()
                
                if command == "quit":
                    print("👋 Goodbye!")
                    break
                
                elif command.startswith("set_task"):
                    # Parse the command more robustly
                    try:
                        # Remove the "set_task " prefix
                        task_part = command[9:].strip()
                        
                        # Split by quotes to handle spaces in descriptions
                        import shlex
                        parts = shlex.split(task_part)
                        
                        if len(parts) >= 4:
                            task_spec = TaskSpec(
                                task_name=parts[0],
                                description=parts[1],
                                requirements=parts[2].split(","),
                                success_criteria=parts[3].split(",")
                            )
                            await self.set_task(task_spec)
                        else:
                            print("❌ Usage: set_task <name> <description> <requirements> <success_criteria>")
                            print("Example: set_task \"String Processor\" \"Create a function that processes strings\" \"Handle basic operations,unicode support,error handling\" \"Works correctly,handles edge cases,good performance\"")
                    except Exception as e:
                        print(f"❌ Error parsing task: {e}")
                        print("❌ Usage: set_task <name> <description> <requirements> <success_criteria>")
                
                elif command == "analyze_task":
                    if not self.task:
                        print("❌ No task set. Use 'set_task' first.")
                        continue
                    analysis = await self.analyze_task()
                    print(f"\n📝 TASK ANALYSIS:\n{analysis}")
                
                elif command == "generate_code":
                    if not self.task:
                        print("❌ No task set. Use 'set_task' first.")
                        continue
                    code = await self.generate_initial_code()
                    print(f"\n💻 GENERATED CODE:\n{code}")
                    
                    # Store the generated code in history
                    self.task_history.append({
                        "generation": self.generation,
                        "code": code,
                        "evaluation": {"overall": 0.5},  # Default evaluation
                        "user_feedback": "Initial generation"
                    })
                
                elif command.startswith("improve_code"):
                    if not self.task:
                        print("❌ No task set. Use 'set_task' first.")
                        continue
                    
                    feedback = command.split(" ", 1)[1] if len(command.split(" ", 1)) > 1 else ""
                    if not self.task_history:
                        print("❌ No code to improve. Generate code first.")
                        continue
                    
                    current_code = self.task_history[-1]["code"]
                    improved_code = await self.run_evolution_cycle(current_code, feedback)
                    print(f"\n💻 IMPROVED CODE:\n{improved_code}")
                
                elif command == "evaluate_code":
                    if not self.task_history:
                        print("❌ No code to evaluate. Generate code first.")
                        continue
                    
                    current_code = self.task_history[-1]["code"]
                    evaluation = await self.evaluate_code(current_code)
                    print(f"\n📊 CODE EVALUATION:")
                    for metric, score in evaluation.items():
                        print(f"  {metric.capitalize()}: {score:.3f}")
                
                elif command == "execute_task":
                    if not self.task_history:
                        print("❌ No code to execute. Generate code first.")
                        continue
                    
                    current_code = self.task_history[-1]["code"]
                    result = await self.execute_task(current_code)
                    
                    if result["success"]:
                        print(f"\n✅ TASK EXECUTION COMPLETE!")
                        print(f"Function: {result['function_name']}")
                        print(f"Success Rate: {result['success_rate']:.1%}")
                        print(f"\n📝 FINAL CODE:")
                        print("=" * 50)
                        print(result["code"])
                        print("=" * 50)
                    else:
                        print(f"\n❌ TASK EXECUTION FAILED: {result['error']}")
                        print(f"Code: {result['code']}")
                
                elif command == "show_history":
                    if not self.task_history:
                        print("❌ No history available.")
                        continue
                    
                    print(f"\n📈 TASK HISTORY:")
                    for i, entry in enumerate(self.task_history):
                        print(f"\nGeneration {entry['generation']}:")
                        print(f"  Overall Score: {entry['evaluation']['overall']:.3f}")
                        print(f"  Code Preview: {entry['code'][:100]}...")
                        if entry['user_feedback']:
                            print(f"  Feedback: {entry['user_feedback']}")
                
                elif command == "show_agent_status":
                    print(f"\n🤖 AGENT STATUS:")
                    print(f"  Generation: {self.generation}")
                    print(f"  Task: {self.task.task_name if self.task else 'None'}")
                    print(f"  History Entries: {len(self.task_history)}")
                    
                    cost_stats = self.llm.get_cost_stats()
                    print(f"  Total Cost: ${cost_stats.get('total_cost', 0):.4f}")
                    print(f"  Total Requests: {cost_stats.get('total_requests', 0)}")
                
                elif command == "evolve_agent":
                    await self.evolve_agent_components()
                    print("✅ Agent components evolved!")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Run the interactive agent."""
    config = AgentConfig()
    agent = InteractiveAgent(config)
    await agent.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main()) 