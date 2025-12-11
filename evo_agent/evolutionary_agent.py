#!/usr/bin/env python3
"""
Evolutionary Agent - Combines agent framework with AlphaEvolve's evolutionary approach.
"""
import os
import sys
import json
import logging
import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import yaml
from dotenv import load_dotenv

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import existing agent framework
from src.agent import Agent, AgentManager
from src.retrieval_engine import RetrievalEngine
from src.embedding_manager import EmbeddingManager
from src.storage_manager import StorageManager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Evaluation helpers
from evaluation_framework import create_evaluator, create_spec_evaluator
from models import TaskSpec, Candidate, EvolutionConfig
from patch_manager import RobustLLMParser, PatchManager
from diversity_manager import DiversityManager, HumanInteractionManager, MetaEvolutionLogger, DiversityConfig
from human_interface import HumanInterfaceConfig, InterfaceType, HumanInterfaceFactory, AsyncHumanReviewManager
from experiment_writer import EvolutionRunWriter


class EvolutionaryAgent:
    """
    Evolutionary agent that combines interactive planning with evolutionary optimization.
    """
    
    def __init__(
        self,
        agent_id: str,
        config_path: Union[str, Path],
        evolution_config: Optional[EvolutionConfig] = None
    ):
        """
        Initialize the evolutionary agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config_path: Path to configuration file
            evolution_config: Evolutionary algorithm configuration
        """
        self.agent_id = agent_id
        self.config_path = config_path
        self.evolution_config = evolution_config or EvolutionConfig()
        
        # Initialize base agent
        self.agent_manager = AgentManager(config_path)
        self.agent = self.agent_manager.create_agent(agent_id)
        
        # Initialize robust parsing and patch management
        self.llm_parser = RobustLLMParser()
        self.patch_manager = PatchManager()
        
        # Initialize diversity management
        diversity_config = DiversityConfig()
        self.diversity_manager = DiversityManager(diversity_config)
        self.human_interaction = HumanInteractionManager()
        self.meta_logger = MetaEvolutionLogger()
        
        # Evolutionary state
        self.candidate_pool: List[Candidate] = []
        self.generation = 0
        self.best_candidate: Optional[Candidate] = None
        self.evaluation_function: Optional[Callable] = None
        self.task_spec: Optional[Dict[str, Any]] = None
        
        # Meta-evolution state
        self.prompt_templates: List[str] = []
        self.mutation_strategies: List[str] = []
        self.evaluation_metrics: List[str] = []
        
        logger.info(f"Evolutionary agent '{agent_id}' initialized")
    
    def set_task(
        self,
        task_description: str,
        initial_tools: Optional[Dict[str, Any]] = None,
        initial_spec: Optional[Dict[str, Any]] = None,
        initial_prompt: Optional[str] = None
    ) -> None:
        """
        Set the task for the evolutionary agent.
        
        Args:
            task_description: Description of the task to accomplish
            initial_tools: Optional starting tools
            initial_spec: Optional evaluation specification
            initial_prompt: Optional initial prompt
        """
        self.task_spec = {
            "description": task_description,
            "tools": initial_tools or {},
            "spec": initial_spec or {},
            "prompt": initial_prompt or "",
            "created_at": datetime.now().isoformat()
        }
        
        # Initialize candidate pool with baseline
        baseline_candidate = Candidate(
            id="baseline",
            code="",
            prompt=initial_prompt or "",
            tools=initial_tools or {},
            memory={},
            generation=0
        )
        self.candidate_pool = [baseline_candidate]
        
        logger.info(f"Task set: {task_description}")
    
    async def interactive_planning(self) -> Dict[str, Any]:
        """
        Step 1-4: Interactive planning with human input.
        """
        logger.info("Starting interactive planning phase")
        
        # Step 1: Task definition (already done in set_task)
        
        # Step 2: Edge case discovery
        edge_cases = await self._discover_edge_cases()
        human_edge_cases = await self._get_human_input("edge_cases", edge_cases)
        
        # Step 3: Tool refinement
        tool_recommendations = await self._analyze_tools()
        human_tool_input = await self._get_human_input("tools", tool_recommendations)
        
        # Step 4: Spec refinement
        spec_recommendations = await self._analyze_spec()
        human_spec_input = await self._get_human_input("spec", spec_recommendations)
        
        # Build hard evaluation suite
        hard_eval_suite = await self._build_hard_eval_suite()
        
        planning_output = {
            "edge_cases": human_edge_cases,
            "tool_recommendations": human_tool_input,
            "spec_recommendations": human_spec_input,
            "hard_eval_suite": hard_eval_suite
        }

        # Apply planning results to internal state and persist
        self._apply_planning_results(planning_output)
        self._persist_planning_results(planning_output)

        # Initialize run writer with planning snapshot
        try:
            self._run_writer = EvolutionRunWriter(self.agent_id)
            self._run_writer.start_run(self.task_spec or {}, planning_output)
        except Exception as e:
            logger.warning(f"Failed to initialize run writer: {e}")

        return planning_output
    
    async def _discover_edge_cases(self) -> List[str]:
        """Discover potential edge cases for the task."""
        prompt = f"""
        Analyze the following task and identify key questions, considerations, and edge cases:
        
        Task: {self.task_spec['description']}
        
        Consider:
        - Ambiguities in requirements
        - Performance edge cases
        - Error conditions
        - Input validation issues
        - Scalability concerns
        
        Provide a list of edge cases and considerations.
        """
        
        response = await self.agent.chat(prompt)
        # Parse response to extract edge cases
        edge_cases = self._parse_list_response(response)
        return edge_cases
    
    async def _analyze_tools(self) -> Dict[str, Any]:
        """Analyze and recommend tool improvements."""
        current_tools = self.task_spec.get("tools", {})
        
        if not current_tools:
            # Propose ideal toolset
            prompt = f"""
            For the task: {self.task_spec['description']}
            
            Design an ideal toolset that would be most effective for this task.
            Consider:
            - Core functionality needed
            - Performance requirements
            - Integration points
            - Error handling
            
            Provide a structured toolset proposal.
            """
        else:
            # Analyze existing tools
            prompt = f"""
            Analyze these existing tools for the task: {self.task_spec['description']}
            
            Tools: {json.dumps(current_tools, indent=2)}
            
            Provide recommendations for improvements, additions, or modifications.
            """
        
        response = await self.agent.chat(prompt)
        return self._parse_tool_recommendations(response)
    
    async def _analyze_spec(self) -> Dict[str, Any]:
        """Analyze and recommend spec improvements."""
        current_spec = self.task_spec.get("spec", {})
        
        if not current_spec:
            # Propose initial spec
            prompt = f"""
            For the task: {self.task_spec['description']}
            
            Design an evaluation specification that would effectively measure success.
            Include:
            - Success criteria
            - Performance metrics
            - Test cases
            - Edge case coverage
            
            Provide a structured evaluation spec.
            """
        else:
            # Analyze existing spec
            prompt = f"""
            Analyze this existing spec for the task: {self.task_spec['description']}
            
            Spec: {json.dumps(current_spec, indent=2)}
            
            Provide recommendations for improvements, additions, or modifications.
            """
        
        response = await self.agent.chat(prompt)
        return self._parse_spec_recommendations(response)
    
    async def _build_hard_eval_suite(self) -> Dict[str, Any]:
        """Build a hard evaluation suite with deliberate edge cases."""
        prompt = f"""
        For the task: {self.task_spec['description']}
        
        Build a "hard" evaluation suite with deliberate edge cases and ambiguities.
        Include:
        - Extreme input cases
        - Malformed data scenarios
        - Performance stress tests
        - Boundary conditions
        - Ambiguous requirements
        
        Provide a comprehensive test suite.
        """
        
        response = await self.agent.chat(prompt)
        return self._parse_eval_suite(response)
    
    async def _get_human_input(self, input_type: str, suggestions: Any) -> Any:
        """Get human input for planning decisions."""
        # Prefer real human interface when available, else fallback to simulated
        try:
            # Lazy-init a CLI interface and async review manager if not present
            if not hasattr(self, "_human_interface"):
                config = HumanInterfaceConfig(interface_type=InterfaceType.CLI, timeout_seconds=120)
                self._human_interface = HumanInterfaceFactory.create_interface(config)
                self._human_interface.start()
                self._human_review_manager = AsyncHumanReviewManager(self._human_interface)

            # Create a pseudo-candidate summarizing the decision for review
            candidate_like = Candidate(
                id=f"plan_{input_type}_{datetime.now().strftime('%H%M%S')}",
                code=json.dumps(suggestions, indent=2) if isinstance(suggestions, (dict, list)) else str(suggestions),
                prompt=f"Planning decision needed: {input_type}",
                tools={},
                memory={},
                generation=self.generation
            )

            response = await self._human_review_manager.request_review(candidate_like, context={"type": input_type})

            if response and response.approved:
                # If suggested_changes provided and suggestions were text-like, return those
                if response.suggested_changes:
                    return response.suggested_changes
                return suggestions
            elif response and not response.approved:
                # If rejected, attach feedback under metadata
                return {"rejected": True, "feedback": response.feedback, "original": suggestions}
            else:
                # Timeout or no response: fallback to auto-accept
                logger.info(f"No human response for {input_type}, auto-accepting suggestions")
                return suggestions
        except Exception as e:
            logger.info(f"Human interface unavailable, auto-accepting {input_type}: {e}")
            return suggestions
    
    def set_evaluation_function(self, eval_func: Callable) -> None:
        """Set the evaluation function for fitness scoring."""
        self.evaluation_function = eval_func

    def _apply_planning_results(self, planning: Dict[str, Any]) -> None:
        """Apply approved tools, spec, and prompt to task_spec and evaluator.
        This converts spec recommendations to a strict TaskSpec if possible and
        switches the evaluation function to a spec-driven evaluator.
        """
        if not self.task_spec:
            return

        # Update tools if present
        tools_update = planning.get("tool_recommendations")
        if tools_update:
            try:
                # Prefer structured list under 'tools'; else keep raw
                tools_list = tools_update.get('tools') if isinstance(tools_update, dict) else None
                if tools_list:
                    # Convert to simple tool dict {name: description}
                    updated_tools = {t.get('name'): t.get('description') for t in tools_list if isinstance(t, dict) and t.get('name')}
                    # Merge with existing
                    self.task_spec["tools"].update({k: v for k, v in updated_tools.items() if v is not None})
                else:
                    # If only raw recommendations exist, store under metadata
                    self.task_spec.setdefault("tools_meta", {})["raw_recommendations"] = tools_update
            except Exception as e:
                logger.warning(f"Failed to apply tool recommendations: {e}")

        # Build TaskSpec from spec recommendations + hard eval suite if possible
        spec_update = planning.get("spec_recommendations") or {}
        hard_eval = planning.get("hard_eval_suite") or {}

        try:
            # Attempt to assemble a TaskSpec dict
            current_spec = self.task_spec.get("spec", {}) or {}

            # Extract lists from recommendations when parser provided them
            rec_tests = spec_update.get('test_cases') if isinstance(spec_update, dict) else None
            rec_metrics = spec_update.get('metrics') if isinstance(spec_update, dict) else None
            rec_criteria = spec_update.get('criteria') if isinstance(spec_update, dict) else None

            # Start constructing a spec dictionary compatible with models.TaskSpec
            spec_dict: Dict[str, Any] = {
                "version": current_spec.get("version", "1.0.0"),
                "test_cases": current_spec.get("test_cases", []),
                "performance_benchmarks": current_spec.get("performance_benchmarks", []),
                "robustness_tests": current_spec.get("robustness_tests", []),
                "success_criteria": current_spec.get("success_criteria", {
                    "correctness_threshold": 0.8,
                    "performance_threshold_ms": 50.0,
                    "robustness_threshold": 0.7
                })
            }

            # Augment with hard eval suite where possible
            # Expecting lists of dict-like items; if strings, store under meta
            if isinstance(hard_eval, dict):
                # We do not directly map edge_cases/performance_tests/stress_tests to TaskSpec; keep for later use
                self.task_spec.setdefault("hard_eval_suite", hard_eval)

            # Keep spec_dict as-is if we don't have structured items; rely on defaults
            # Validate and set evaluator if spec is complete enough
            try:
                task_spec = TaskSpec(**spec_dict)
                # Switch to spec-driven evaluator
                self.set_evaluation_function(create_spec_evaluator(task_spec))
                # Persist normalized spec
                self.task_spec["spec"] = task_spec.model_dump()
            except Exception as e:
                logger.warning(f"Unable to build strict TaskSpec yet, continuing with existing evaluator: {e}")
        except Exception as e:
            logger.warning(f"Failed to apply spec recommendations: {e}")

    def _persist_planning_results(self, planning: Dict[str, Any]) -> None:
        """Persist planning outputs and task spec under experiments directory."""
        try:
            experiments_dir = Path(__file__).parent / "experiments"
            os.makedirs(experiments_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = experiments_dir / f"planning_{stamp}"
            os.makedirs(run_dir, exist_ok=True)

            # Save planning JSON
            with open(run_dir / "planning_results.json", "w", encoding="utf-8") as f:
                json.dump(planning, f, indent=2)

            # Save task spec (possibly updated)
            if self.task_spec:
                with open(run_dir / "task_spec.json", "w", encoding="utf-8") as f:
                    json.dump(self.task_spec, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist planning results: {e}")
    
    async def evolutionary_optimization(self) -> Candidate:
        """
        Step 5: Evolutionary optimization using AlphaEvolve approach.
        """
        logger.info("Starting evolutionary optimization phase")
        
        if not self.evaluation_function:
            raise ValueError("Evaluation function must be set before optimization")
        
        # Initialize population
        await self._initialize_population()
        
        # Evolutionary loop
        for generation in range(self.evolution_config.generations):
            logger.info(f"Generation {generation + 1}/{self.evolution_config.generations}")
            
            # Generate new candidates
            new_candidates = await self._generate_candidates()
            
            # Evaluate candidates
            await self._evaluate_candidates(new_candidates)
            
            # Select next generation using diversity-aware selection
            all_candidates = self.candidate_pool + new_candidates
            self.candidate_pool = self.diversity_manager.select_diverse_candidates(
                all_candidates, 
                self.evolution_config.population_size
            )
            
            # Track cluster evolution
            clusters = self.diversity_manager.cluster_candidates(self.candidate_pool)
            self.diversity_manager.track_cluster_evolution(clusters)
            
            # Update best candidate
            self._update_best_candidate()
            
            # Log diversity metrics
            diversity_metrics = self.diversity_manager.get_diversity_metrics()
            if diversity_metrics:
                logger.info(f"Diversity metrics: {diversity_metrics}")
            
            # Write generation snapshot for visualization
            try:
                if hasattr(self, "_run_writer"):
                    self._run_writer.write_generation(
                        generation_index=generation + 1,
                        candidates=self.candidate_pool,
                        best_candidate=self.best_candidate,
                        diversity_metrics=diversity_metrics
                    )
            except Exception as e:
                logger.warning(f"Failed to write generation snapshot: {e}")

            # Check convergence
            if self._check_convergence():
                logger.info("Convergence reached")
                break
            
            # Meta-evolution: evolve prompts and strategies
            if generation % 10 == 0:
                await self._meta_evolve()
        
        # Finalize run
        try:
            if hasattr(self, "_run_writer"):
                self._run_writer.end_run(final_best=self.best_candidate, evolution_stats=self.get_evolution_stats())
        except Exception as e:
            logger.warning(f"Failed to finalize run writer: {e}")

        return self.best_candidate
    
    async def _initialize_population(self) -> None:
        """Initialize the candidate population."""
        baseline = self.candidate_pool[0]
        
        # Generate initial population from baseline
        for i in range(self.evolution_config.population_size - 1):
            candidate = await self._mutate_candidate(baseline, f"init_{i}")
            self.candidate_pool.append(candidate)
        
        # Evaluate initial population
        await self._evaluate_candidates(self.candidate_pool)
    
    async def _generate_candidates(self) -> List[Candidate]:
        """Generate new candidates through mutation and crossover."""
        new_candidates = []
        
        # Elitism: keep best candidates
        elite = sorted(self.candidate_pool, key=lambda x: x.fitness_score, reverse=True)[:self.evolution_config.elite_size]
        new_candidates.extend(elite)
        
        # Generate remaining candidates
        while len(new_candidates) < self.evolution_config.population_size:
            if np.random.random() < self.evolution_config.mutation_rate:
                # Mutation
                parent = self._tournament_selection()
                child = await self._mutate_candidate(parent, f"mut_{len(new_candidates)}")
                new_candidates.append(child)
            else:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = await self._crossover_candidates(parent1, parent2, f"cross_{len(new_candidates)}")
                new_candidates.append(child)
        
        return new_candidates
    
    async def _mutate_candidate(self, parent: Candidate, child_id: str) -> Candidate:
        """Mutate a candidate using LLM-generated diffs."""
        mutation_prompt = f"""
        Given this parent candidate:
        
        Code: {parent.code}
        Prompt: {parent.prompt}
        Tools: {json.dumps(parent.tools, indent=2)}
        
        Task: {self.task_spec['description']}
        
        Generate a targeted mutation that improves the candidate.
        Focus on:
        - Code improvements
        - Prompt refinements
        - Tool optimizations
        
        Provide the mutation as a diff patch.
        """
        
        # Get mutation from LLM
        mutation_response = await self.agent.chat(mutation_prompt)
        mutation_patch = self._parse_mutation_patch(mutation_response)
        
        # Apply mutation
        mutated_code = self._apply_patch(parent.code, mutation_patch)
        mutated_prompt = self._apply_prompt_mutation(parent.prompt, mutation_response)

        # Optionally mutate tools and memory
        mutated_tools = await self._maybe_mutate_tools(parent.tools)
        mutated_memory = await self._maybe_mutate_memory(parent.memory)
        
        return Candidate(
            id=child_id,
            code=mutated_code,
            prompt=mutated_prompt,
            tools=mutated_tools,
            memory=mutated_memory,
            generation=parent.generation + 1,
            parent_id=parent.id,
            mutation_type="llm_diff"
        )
    
    async def _crossover_candidates(self, parent1: Candidate, parent2: Candidate, child_id: str) -> Candidate:
        """Crossover two candidates."""
        # Simple uniform crossover
        child_code = self._crossover_code(parent1.code, parent2.code)
        child_prompt = self._crossover_prompt(parent1.prompt, parent2.prompt)
        child_tools = self._crossover_tools(parent1.tools, parent2.tools)
        child_memory = self._crossover_memory(parent1.memory, parent2.memory)
        
        return Candidate(
            id=child_id,
            code=child_code,
            prompt=child_prompt,
            tools=child_tools,
            memory=child_memory,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_id=f"{parent1.id}+{parent2.id}",
            mutation_type="crossover"
        )
    
    async def _evaluate_candidates(self, candidates: List[Candidate]) -> None:
        """Evaluate candidates using the evaluation function."""
        if self.evolution_config.parallel_evaluation:
            await self._evaluate_parallel(candidates)
        else:
            await self._evaluate_sequential(candidates)
    
    async def _evaluate_parallel(self, candidates: List[Candidate]) -> None:
        """Evaluate candidates in parallel."""
        with ThreadPoolExecutor(max_workers=self.evolution_config.max_workers) as executor:
            futures = []
            for candidate in candidates:
                future = executor.submit(self.evaluation_function, candidate)
                futures.append((candidate, future))
            
            for candidate, future in futures:
                try:
                    fitness_score = future.result(timeout=self.evolution_config.evaluation_timeout)
                    candidate.fitness_score = fitness_score
                except Exception as e:
                    logger.error(f"Evaluation failed for candidate {candidate.id}: {e}")
                    candidate.fitness_score = 0.0
    
    async def _evaluate_sequential(self, candidates: List[Candidate]) -> None:
        """Evaluate candidates sequentially."""
        for candidate in candidates:
            try:
                fitness_score = self.evaluation_function(candidate)
                candidate.fitness_score = fitness_score
            except Exception as e:
                logger.error(f"Evaluation failed for candidate {candidate.id}: {e}")
                candidate.fitness_score = 0.0
    
    async def _select_next_generation(self, new_candidates: List[Candidate]) -> List[Candidate]:
        """Select the next generation using tournament selection."""
        # Combine current pool with new candidates
        all_candidates = self.candidate_pool + new_candidates
        
        # Select top candidates
        selected = []
        while len(selected) < self.evolution_config.population_size:
            tournament_candidates = np.random.choice(all_candidates, self.evolution_config.tournament_size, replace=False)
            winner = max(tournament_candidates, key=lambda x: x.fitness_score)
            selected.append(winner)
        
        return selected
    
    def _tournament_selection(self) -> Candidate:
        """Tournament selection for parent selection."""
        tournament = np.random.choice(self.candidate_pool, self.evolution_config.tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _update_best_candidate(self) -> None:
        """Update the best candidate found so far."""
        current_best = max(self.candidate_pool, key=lambda x: x.fitness_score)
        if not self.best_candidate or current_best.fitness_score > self.best_candidate.fitness_score:
            self.best_candidate = current_best
            logger.info(f"New best candidate: {current_best.id} (fitness: {current_best.fitness_score})")
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if not self.best_candidate:
            return False
        
        # Check if best fitness exceeds threshold
        if self.best_candidate.fitness_score >= self.evolution_config.fitness_threshold:
            return True
        
        # Check if fitness has plateaued
        recent_fitness = [c.fitness_score for c in self.candidate_pool[-10:]]
        if len(recent_fitness) >= 10:
            variance = np.var(recent_fitness)
            if variance < 0.01:  # Low variance indicates plateau
                return True
        
        return False
    
    async def _meta_evolve(self) -> None:
        """Meta-evolution: evolve prompts and strategies."""
        logger.info("Performing meta-evolution")
        
        # Analyze which mutation strategies work best
        successful_mutations = [c for c in self.candidate_pool if c.mutation_type == "llm_diff" and c.fitness_score > 0.5]
        
        if successful_mutations:
            # Evolve mutation prompts based on successful patterns
            await self._evolve_mutation_prompts(successful_mutations)
        
        # Evolve evaluation strategies
        await self._evolve_evaluation_strategies()

        # Evolve tools and memory templates occasionally based on best candidates
        try:
            top = sorted(self.candidate_pool, key=lambda x: x.fitness_score, reverse=True)[: max(1, self.evolution_config.elite_size)]
            await self._meta_evolve_tools_and_memory(top)
        except Exception as e:
            logger.warning(f"Meta-evolution of tools/memory failed: {e}")
    
    async def _evolve_mutation_prompts(self, successful_mutations: List[Candidate]) -> None:
        """Evolve mutation prompts based on successful mutations."""
        # Analyze patterns in successful mutations
        prompt = f"""
        Analyze these successful mutations and identify patterns:
        
        {[c.metadata.get('mutation_prompt', '') for c in successful_mutations]}
        
        Generate improved mutation prompts based on these patterns.
        """
        
        response = await self.agent.chat(prompt)
        # Update mutation prompts
        self.prompt_templates.append(response)
    
    async def _evolve_evaluation_strategies(self) -> None:
        """Evolve evaluation strategies."""
        # Analyze evaluation performance
        prompt = f"""
        Analyze the evaluation performance and suggest improvements:
        
        Current evaluation function: {self.evaluation_function.__name__}
        Best fitness: {self.best_candidate.fitness_score if self.best_candidate else 0}
        
        Suggest improvements to the evaluation strategy.
        """
        
        response = await self.agent.chat(prompt)
        # Update evaluation strategies
        self.evaluation_metrics.append(response)

    async def _meta_evolve_tools_and_memory(self, top_candidates: List[Candidate]) -> None:
        """Meta-evolve tool and memory templates guided by top-performing candidates."""
        if not top_candidates:
            return

        # Construct a summary prompt from top candidates' tool/memory payloads
        try:
            tools_corpus = []
            memory_corpus = []
            for c in top_candidates:
                if c.tools:
                    tools_corpus.append(json.dumps(c.tools)[:2000])
                if c.memory:
                    memory_corpus.append(json.dumps(c.memory)[:2000])

            if tools_corpus:
                prompt_tools = f"""
                Analyze these high-performing tool configurations and synthesize improved, minimal, and robust tool definitions.
                Prioritize clarity, correctness, and error handling. Provide JSON object mapping names to definitions.
                Examples:
                {chr(10).join(tools_corpus[:5])}
                """
                improved_tools_str = await self.agent.chat(prompt_tools)
                # Best-effort JSON parse
                try:
                    improved_tools = json.loads(improved_tools_str)
                    if isinstance(improved_tools, dict):
                        # Merge conservatively into task_spec tools
                        self.task_spec.setdefault("tools", {}).update(improved_tools)
                except Exception:
                    pass

            if memory_corpus:
                prompt_memory = f"""
                Analyze these high-performing memory configurations and synthesize improved, minimal, and robust memory components.
                Provide a JSON object mapping names to component implementations.
                Examples:
                {chr(10).join(memory_corpus[:5])}
                """
                improved_memory_str = await self.agent.chat(prompt_memory)
                try:
                    improved_memory = json.loads(improved_memory_str)
                    if isinstance(improved_memory, dict):
                        # Optionally seed into future offsprings by storing in task_spec
                        self.task_spec.setdefault("memory_templates", {}).update(improved_memory)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Tool/memory meta evolution error: {e}")
    
    
    async def _get_human_input(self, input_type: str, suggestions: Any) -> Any:
        """Get human input for planning decisions."""
        # For now, return suggestions as-is
        # In a real implementation, this would prompt the user
        logger.info(f"Human input needed for {input_type}: {suggestions}")
        return suggestions
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse a list response from the agent using robust parsing."""
        return self.llm_parser.parse_list_response(response)
    
    def _parse_tool_recommendations(self, response: str) -> Dict[str, Any]:
        """Parse tool recommendations from agent response using robust parsing."""
        return self.llm_parser.parse_tool_recommendations(response)
    
    def _parse_spec_recommendations(self, response: str) -> Dict[str, Any]:
        """Parse spec recommendations from agent response using robust parsing."""
        return self.llm_parser.parse_spec_recommendations(response)
    
    def _parse_eval_suite(self, response: str) -> Dict[str, Any]:
        """Parse evaluation suite from agent response using robust parsing."""
        return self.llm_parser.parse_eval_suite(response)
    
    def _parse_mutation_patch(self, response: str) -> List[Any]:
        """Parse mutation patch from agent response using robust parsing."""
        return self.llm_parser.parse_mutation_response(response)
    
    def _apply_patch(self, code: str, diff_blocks: List[Any]) -> str:
        """Apply patches to code using proper diff/patch system."""
        return self.patch_manager.apply_patch(code, diff_blocks)
    
    def _apply_prompt_mutation(self, prompt: str, mutation: str) -> str:
        """Apply mutation to prompt."""
        # Simple implementation - in practice, use more sophisticated mutation
        return prompt + "\n" + mutation
    
    def _crossover_code(self, code1: str, code2: str) -> str:
        """Crossover two code strings."""
        # Simple uniform crossover
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        result = []
        max_lines = max(len(lines1), len(lines2))
        
        for i in range(max_lines):
            if i < len(lines1) and i < len(lines2):
                result.append(lines1[i] if np.random.random() < 0.5 else lines2[i])
            elif i < len(lines1):
                result.append(lines1[i])
            else:
                result.append(lines2[i])
        
        return '\n'.join(result)
    
    def _crossover_prompt(self, prompt1: str, prompt2: str) -> str:
        """Crossover two prompts."""
        # Simple uniform crossover
        return prompt1 if np.random.random() < 0.5 else prompt2
    
    def _crossover_tools(self, tools1: Dict[str, Any], tools2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two tool dictionaries."""
        # Simple uniform crossover
        result = {}
        all_keys = set(tools1.keys()) | set(tools2.keys())
        
        for key in all_keys:
            if key in tools1 and key in tools2:
                result[key] = tools1[key] if np.random.random() < 0.5 else tools2[key]
            elif key in tools1:
                result[key] = tools1[key]
            else:
                result[key] = tools2[key]
        
        return result

    def _crossover_memory(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two memory dictionaries using uniform crossover."""
        result: Dict[str, Any] = {}
        all_keys = set(memory1.keys()) | set(memory2.keys())
        for key in all_keys:
            if key in memory1 and key in memory2:
                result[key] = memory1[key] if np.random.random() < 0.5 else memory2[key]
            elif key in memory1:
                result[key] = memory1[key]
            else:
                result[key] = memory2[key]
        return result

    async def _maybe_mutate_tools(self, tools: Dict[str, Any]) -> Dict[str, Any]:
        """With small probability, mutate the tools using the agent LLM."""
        if not tools:
            return {}
        if np.random.random() >= max(0.1, self.evolution_config.mutation_rate * 0.5):
            return tools.copy()

        updated = tools.copy()
        # Select a random tool to improve
        key = np.random.choice(list(updated.keys()))
        try:
            evolution_prompt = f"""
            Improve this agent tool to make it more effective and robust.
            Current tool name: {key}
            Current tool content:
            {updated[key]}

            Make it more comprehensive and error-resistant. Provide only the improved content:
            """
            improved_content = await self.agent.chat(evolution_prompt)
            updated[key] = improved_content.strip()
        except Exception as e:
            logger.warning(f"Tool mutation failed for {key}: {e}")
        return updated

    async def _maybe_mutate_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """With small probability, mutate the memory components using the agent LLM."""
        if not memory:
            return {}
        if np.random.random() >= max(0.1, self.evolution_config.mutation_rate * 0.5):
            return memory.copy()

        updated = memory.copy()
        key = np.random.choice(list(updated.keys()))
        try:
            evolution_prompt = f"""
            Improve this agent memory component to make it more effective at storing and retrieving information.
            Current component name: {key}
            Current component content:
            {updated[key]}

            Make it more comprehensive and useful. Provide only the improved content:
            """
            improved_content = await self.agent.chat(evolution_prompt)
            updated[key] = improved_content.strip()
        except Exception as e:
            logger.warning(f"Memory mutation failed for {key}: {e}")
        return updated
    
    def get_best_candidate(self) -> Optional[Candidate]:
        """Get the best candidate found so far."""
        return self.best_candidate
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics about the evolution process."""
        if not self.candidate_pool:
            return {}
        
        fitness_scores = [c.fitness_score for c in self.candidate_pool]
        
        return {
            "generation": self.generation,
            "population_size": len(self.candidate_pool),
            "best_fitness": max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
            "best_candidate_id": self.best_candidate.id if self.best_candidate else None
        } 