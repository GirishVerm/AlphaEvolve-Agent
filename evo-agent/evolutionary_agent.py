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


@dataclass
class Candidate:
    """Represents a candidate solution in the evolutionary pool."""
    id: str
    code: str
    prompt: str
    tools: Dict[str, Any]
    memory: Dict[str, Any]
    fitness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary parameters."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_size: int = 5
    tournament_size: int = 3
    parallel_evaluation: bool = True
    max_workers: int = 4
    evaluation_timeout: int = 300  # seconds
    fitness_threshold: float = 0.95
    diversity_weight: float = 0.1


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
        
        return {
            "edge_cases": human_edge_cases,
            "tool_recommendations": human_tool_input,
            "spec_recommendations": human_spec_input,
            "hard_eval_suite": hard_eval_suite
        }
    
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
        # For now, return suggestions as-is
        # In a real implementation, this would prompt the user
        logger.info(f"Human input needed for {input_type}: {suggestions}")
        return suggestions
    
    def set_evaluation_function(self, eval_func: Callable) -> None:
        """Set the evaluation function for fitness scoring."""
        self.evaluation_function = eval_func
    
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
            
            # Check convergence
            if self._check_convergence():
                logger.info("Convergence reached")
                break
            
            # Meta-evolution: evolve prompts and strategies
            if generation % 10 == 0:
                await self._meta_evolve()
        
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
        
        return Candidate(
            id=child_id,
            code=mutated_code,
            prompt=mutated_prompt,
            tools=parent.tools.copy(),
            memory=parent.memory.copy(),
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
        
        return Candidate(
            id=child_id,
            code=child_code,
            prompt=child_prompt,
            tools=child_tools,
            memory={},
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
    
    # Import robust parsing and patch management
    from patch_manager import RobustLLMParser, PatchManager
    from diversity_manager import DiversityManager, HumanInteractionManager, MetaEvolutionLogger, DiversityConfig
    

    
    async def _get_human_input(self, input_type: str, suggestions: Any) -> Any:
        """Get human input for planning decisions."""
        return await self.human_interaction.get_human_input(input_type, suggestions)
    
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