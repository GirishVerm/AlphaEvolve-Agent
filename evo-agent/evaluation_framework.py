#!/usr/bin/env python3
"""
Evaluation Framework for Evolutionary Agent Candidates.
"""
import os
import sys
import json
import time
import ast
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import numpy as np
import subprocess
import tempfile
import traceback

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evolutionary_agent import Candidate
from models import TaskSpec, TestCase, Benchmark, RobustnessTest, EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationFramework:
    """
    Framework for evaluating candidate solutions.
    """
    
    def __init__(self, task_spec: Union[TaskSpec, Dict[str, Any]]):
        """
        Initialize the evaluation framework.
        
        Args:
            task_spec: Task specification including evaluation criteria
        """
        # Normalize input to Pydantic TaskSpec (backward-compatible)
        self.task_spec: TaskSpec = task_spec if isinstance(task_spec, TaskSpec) else TaskSpec(**task_spec)
        self.test_cases: List[TestCase] = self.task_spec.test_cases
        self.performance_benchmarks: List[Benchmark] = self.task_spec.performance_benchmarks
        self.robustness_tests: List[RobustnessTest] = self.task_spec.robustness_tests
        self.success_criteria = self.task_spec.success_criteria
        
        # Evaluation weights
        self.weights = {
            "correctness": 0.4,
            "performance": 0.3,
            "robustness": 0.3
        }
    
    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        """
        Evaluate a candidate solution.
        
        Args:
            candidate: Candidate to evaluate
            
        Returns:
            EvaluationResult with scores and metrics
        """
        start_time = time.time()
        
        try:
            # Parse and validate candidate code
            parsed_code = self._parse_candidate_code(candidate)
            
            # Run correctness tests
            correctness_result = self._evaluate_correctness(candidate, parsed_code)
            
            # Run performance benchmarks
            performance_result = self._evaluate_performance(candidate, parsed_code)
            
            # Run robustness tests
            robustness_result = self._evaluate_robustness(candidate, parsed_code)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                correctness_result["score"],
                performance_result["score"],
                robustness_result["score"]
            )
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                candidate_id=candidate.id,
                overall_score=overall_score,
                correctness_score=correctness_result["score"],
                performance_score=performance_result["score"],
                robustness_score=robustness_result["score"],
                metrics={
                    "correctness": correctness_result["metrics"],
                    "performance": performance_result["metrics"],
                    "robustness": robustness_result["metrics"]
                },
                errors=correctness_result["errors"] + performance_result["errors"] + robustness_result["errors"],
                warnings=correctness_result["warnings"] + performance_result["warnings"] + robustness_result["warnings"],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for candidate {candidate.id}: {e}")
            return EvaluationResult(
                candidate_id=candidate.id,
                overall_score=0.0,
                correctness_score=0.0,
                performance_score=0.0,
                robustness_score=0.0,
                metrics={},
                errors=[str(e)],
                warnings=[],
                execution_time=time.time() - start_time
            )
    
    def _parse_candidate_code(self, candidate: Candidate) -> Dict[str, Any]:
        """Parse and validate candidate code."""
        errors = []
        warnings = []
        
        # Check if code is valid Python
        try:
            ast.parse(candidate.code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for common issues
        if not candidate.code.strip():
            warnings.append("Empty code")
        
        if len(candidate.code) > 10000:
            warnings.append("Very long code")
        
        return {
            "valid": True,
            "code": candidate.code,
            "errors": errors,
            "warnings": warnings
        }
    
    def _evaluate_correctness(self, candidate: Candidate, parsed_code: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate correctness using test cases."""
        if not parsed_code["valid"]:
            return {
                "score": 0.0,
                "metrics": {"tests_passed": 0, "total_tests": len(self.test_cases)},
                "errors": parsed_code["errors"],
                "warnings": parsed_code["warnings"]
            }
        
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        warnings = []
        
        # Create temporary environment for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_candidate.py"
            
            # Write candidate code to file
            test_file.write_text(parsed_code["code"])
            
            # Run test cases
            for i, test_case in enumerate(self.test_cases):
                try:
                    result = self._run_test_case(test_file, test_case)
                    if result["passed"]:
                        passed_tests += 1
                    else:
                        errors.append(f"Test {i} failed: {result['error']}")
                except Exception as e:
                    errors.append(f"Test {i} error: {e}")
        
        score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            "score": score,
            "metrics": {
                "tests_passed": passed_tests,
                "total_tests": total_tests,
                "pass_rate": score
            },
            "errors": errors,
            "warnings": warnings
        }
    
    def _evaluate_performance(self, candidate: Candidate, parsed_code: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance using benchmarks."""
        if not parsed_code["valid"]:
            return {
                "score": 0.0,
                "metrics": {},
                "errors": parsed_code["errors"],
                "warnings": parsed_code["warnings"]
            }
        
        errors = []
        warnings = []
        metrics = {}
        
        # Run performance benchmarks
        for benchmark in self.performance_benchmarks:
            try:
                result = self._run_performance_benchmark(parsed_code["code"], benchmark)
                metrics[benchmark.name] = result
                
                # Check if performance meets requirements
                if result["time"] > benchmark.target:
                    warnings.append(f"Performance target not met for {benchmark.name}")
                
            except Exception as e:
                errors.append(f"Benchmark {benchmark.name} error: {e}")
        
        # Calculate performance score
        if metrics:
            # Normalize performance scores
            performance_scores = []
            for benchmark in self.performance_benchmarks:
                if benchmark.name in metrics:
                    score = self._normalize_performance_score(
                        metrics[benchmark.name]["time"],
                        getattr(benchmark, "target", float('inf')),
                        getattr(benchmark, "baseline", 1.0) or 1.0,
                    )
                    performance_scores.append(score)
            
            score = np.mean(performance_scores) if performance_scores else 0.0
        else:
            score = 0.0
        
        return {
            "score": score,
            "metrics": metrics,
            "errors": errors,
            "warnings": warnings
        }
    
    def _evaluate_robustness(self, candidate: Candidate, parsed_code: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate robustness using edge cases and error conditions."""
        if not parsed_code["valid"]:
            return {
                "score": 0.0,
                "metrics": {},
                "errors": parsed_code["errors"],
                "warnings": parsed_code["warnings"]
            }
        
        passed_tests = 0
        total_tests = len(self.robustness_tests)
        errors = []
        warnings = []
        
        # Run robustness tests
        for i, test in enumerate(self.robustness_tests):
            try:
                result = self._run_robustness_test(parsed_code["code"], test)
                if result["passed"]:
                    passed_tests += 1
                else:
                    errors.append(f"Robustness test {i} failed: {result['error']}")
            except Exception as e:
                errors.append(f"Robustness test {i} error: {e}")
        
        score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            "score": score,
            "metrics": {
                "robustness_tests_passed": passed_tests,
                "total_robustness_tests": total_tests,
                "robustness_rate": score
            },
            "errors": errors,
            "warnings": warnings
        }
    
    def _run_test_case(self, test_file: Path, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        # Create test script
        test_script = f"""
import sys
sys.path.append('{test_file.parent}')

# Import the candidate function
from test_candidate import *

# Test case
{test_case.code}

# Expected result
expected = {test_case.expected}
actual = result

# Check result
if actual == expected:
    print("PASS")
else:
    print(f"FAIL: expected {{expected}}, got {{actual}}")
"""
        
        # Run test
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "PASS" in result.stdout:
                return {"passed": True, "error": None}
            else:
                return {"passed": False, "error": result.stderr or result.stdout}
                
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _run_performance_benchmark(self, code: str, benchmark: Benchmark) -> Dict[str, Any]:
        """Run a performance benchmark."""
        # Create benchmark script
        benchmark_script = f"""
import time
import sys

# Import the candidate function
{code}

# Benchmark
{benchmark.code}

# Measure execution time
start_time = time.time()
for _ in range({benchmark.iterations}):
    result = {benchmark.function_call}
end_time = time.time()

execution_time = (end_time - start_time) / {benchmark.iterations}
print(f"{{execution_time}}")
"""
        
        # Run benchmark
        try:
            result = subprocess.run(
                [sys.executable, "-c", benchmark_script],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                execution_time = float(result.stdout.strip())
                return {"time": execution_time, "success": True}
            else:
                return {"time": float('inf'), "success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"time": float('inf'), "success": False, "error": "Benchmark timed out"}
        except Exception as e:
            return {"time": float('inf'), "success": False, "error": str(e)}
    
    def _run_robustness_test(self, code: str, test: RobustnessTest) -> Dict[str, Any]:
        """Run a robustness test."""
        # Create test script
        test_script = f"""
import sys

# Import the candidate function
{code}

# Robustness test
{test.code}

# Check if function handles edge case gracefully
try:
    result = {test.function_call}
    print("PASS")
except Exception as e:
    if {test.expect_exception}:
        print("PASS")
    else:
        print(f"FAIL: {{e}}")
"""
        
        # Run test
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "PASS" in result.stdout:
                return {"passed": True, "error": None}
            else:
                return {"passed": False, "error": result.stderr or result.stdout}
                
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _normalize_performance_score(self, actual_time: float, target_time: float, baseline_time: float) -> float:
        """Normalize performance score."""
        if actual_time <= target_time:
            return 1.0
        elif actual_time <= baseline_time:
            return 0.5
        else:
            return max(0.0, 1.0 - (actual_time - target_time) / target_time)
    
    def _calculate_overall_score(self, correctness: float, performance: float, robustness: float) -> float:
        """Calculate overall fitness score."""
        return (
            self.weights["correctness"] * correctness +
            self.weights["performance"] * performance +
            self.weights["robustness"] * robustness
        )


class MarkdownToHTMLEvaluator:
    """
    Specific evaluator for Markdown to HTML conversion task.
    """
    
    def __init__(self):
        """Initialize the Markdown to HTML evaluator."""
        self.test_cases = self._create_test_cases()
        self.performance_benchmarks = self._create_performance_benchmarks()
        self.robustness_tests = self._create_robustness_tests()
        
        self.task_spec = {
            "test_cases": self.test_cases,
            "performance_benchmarks": self.performance_benchmarks,
            "robustness_tests": self.robustness_tests,
            "success_criteria": {
                "correctness_threshold": 0.9,
                "performance_threshold": 50,  # ms
                "robustness_threshold": 0.8
            }
        }
        
        self.evaluator = EvaluationFramework(self.task_spec)
    
    def evaluate(self, candidate: Candidate) -> float:
        """
        Evaluate a Markdown to HTML conversion candidate.
        
        Args:
            candidate: Candidate to evaluate
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        # Add the convert function to the candidate code
        full_code = f"""
{candidate.code}

# Test function
def test_convert():
    return convert
"""
        
        # Create a modified candidate with the test function
        test_candidate = Candidate(
            id=candidate.id,
            code=full_code,
            prompt=candidate.prompt,
            tools=candidate.tools,
            memory=candidate.memory,
            generation=candidate.generation,
            parent_id=candidate.parent_id,
            mutation_type=candidate.mutation_type
        )
        
        # Evaluate the candidate
        result = self.evaluator.evaluate_candidate(test_candidate)
        
        return result.overall_score
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create test cases for Markdown to HTML conversion."""
        return [
            {
                "name": "basic_heading",
                "code": "result = convert('# Hello World')",
                "expected": "<h1>Hello World</h1>"
            },
            {
                "name": "basic_paragraph",
                "code": "result = convert('This is a paragraph.')",
                "expected": "<p>This is a paragraph.</p>"
            },
            {
                "name": "basic_list",
                "code": "result = convert('- Item 1\\n- Item 2')",
                "expected": "<ul>\\n<li>Item 1</li>\\n<li>Item 2</li>\\n</ul>"
            },
            {
                "name": "basic_link",
                "code": "result = convert('[Link](https://example.com)')",
                "expected": "<a href=\"https://example.com\">Link</a>"
            },
            {
                "name": "basic_bold",
                "code": "result = convert('**Bold text**')",
                "expected": "<strong>Bold text</strong>"
            }
        ]
    
    def _create_performance_benchmarks(self) -> List[Dict[str, Any]]:
        """Create performance benchmarks."""
        return [
            {
                "name": "small_text",
                "code": "text = '# Hello\\nThis is a small text.'",
                "function_call": "convert(text)",
                "iterations": 1000,
                "target": 0.001  # 1ms
            },
            {
                "name": "medium_text",
                "code": "text = '\\n'.join(['# Heading ' + str(i) + '\\nParagraph ' + str(i) for i in range(100)])",
                "function_call": "convert(text)",
                "iterations": 100,
                "target": 0.01  # 10ms
            }
        ]
    
    def _create_robustness_tests(self) -> List[Dict[str, Any]]:
        """Create robustness tests."""
        return [
            {
                "name": "empty_input",
                "code": "text = ''",
                "function_call": "convert(text)",
                "expect_exception": False
            },
            {
                "name": "none_input",
                "code": "text = None",
                "function_call": "convert(text)",
                "expect_exception": True
            },
            {
                "name": "large_input",
                "code": "text = '\\n'.join(['# Heading ' + str(i) for i in range(10000)])",
                "function_call": "convert(text)",
                "expect_exception": False
            }
        ]


def create_evaluator(task_type: str) -> Callable[[Candidate], float]:
    """
    Create an appropriate evaluator for the given task type.
    
    Args:
        task_type: Type of task to evaluate
        
    Returns:
        Evaluation function
    """
    if task_type == "markdown_to_html":
        evaluator = MarkdownToHTMLEvaluator()
        return evaluator.evaluate
    else:
        # Default evaluator
        def default_evaluator(candidate: Candidate) -> float:
            # Simple evaluation based on code length and complexity
            if not candidate.code.strip():
                return 0.0
            
            # Basic heuristics
            lines = len(candidate.code.split('\n'))
            chars = len(candidate.code)
            
            # Prefer medium-length, well-structured code
            if 10 <= lines <= 100 and 100 <= chars <= 5000:
                return 0.8
            elif lines < 10 or chars < 100:
                return 0.3
            else:
                return 0.5
        
        return default_evaluator 