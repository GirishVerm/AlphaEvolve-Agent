#!/usr/bin/env python3
"""
Population Manager - Handles pool, selection, crossover operations.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from evolutionary_agent import Candidate
from diversity_manager import DiversityManager, DiversityConfig

logger = logging.getLogger(__name__)


@dataclass
class PopulationConfig:
    """Configuration for population management."""
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    parallel_evaluation: bool = True
    max_workers: int = 4
    evaluation_timeout: int = 300
    diversity_enabled: bool = True
    cache_vectors: bool = True


class PopulationManager:
    """
    Manages evolutionary population operations.
    """
    
    def __init__(
        self, 
        config: Optional[PopulationConfig] = None,
        diversity_config: Optional[DiversityConfig] = None
    ):
        """
        Initialize population manager.
        
        Args:
            config: Population configuration
            diversity_config: Diversity configuration
        """
        self.config = config or PopulationConfig()
        self.diversity_manager = DiversityManager(diversity_config) if self.config.diversity_enabled else None
        
        # Population state
        self.population: List[Candidate] = []
        self.generation = 0
        self.best_candidate: Optional[Candidate] = None
        
        # Metrics
        self.evaluation_count = 0
        self.selection_count = 0
        self.crossover_count = 0
        self.mutation_count = 0
    
    def initialize_population(self, baseline_candidate: Candidate) -> None:
        """
        Initialize population from baseline candidate.
        
        Args:
            baseline_candidate: Baseline candidate to start from
        """
        self.population = [baseline_candidate]
        self.best_candidate = baseline_candidate
        self.generation = 0
        
        logger.info(f"Initialized population with baseline candidate: {baseline_candidate.id}")
    
    def add_candidates(self, candidates: List[Candidate]) -> None:
        """
        Add candidates to population.
        
        Args:
            candidates: Candidates to add
        """
        self.population.extend(candidates)
        logger.info(f"Added {len(candidates)} candidates to population")
    
    def select_parents(self, num_parents: int) -> List[Candidate]:
        """
        Select parents using tournament selection.
        
        Args:
            num_parents: Number of parents to select
            
        Returns:
            Selected parent candidates
        """
        parents = []
        
        for _ in range(num_parents):
            tournament = np.random.choice(
                self.population, 
                min(self.config.tournament_size, len(self.population)), 
                replace=False
            )
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        self.selection_count += num_parents
        logger.debug(f"Selected {num_parents} parents using tournament selection")
        
        return parents
    
    def select_diverse_parents(self, num_parents: int) -> List[Candidate]:
        """
        Select diverse parents using diversity-aware selection.
        
        Args:
            num_parents: Number of parents to select
            
        Returns:
            Selected diverse parent candidates
        """
        if not self.diversity_manager:
            return self.select_parents(num_parents)
        
        # Use diversity manager to select diverse candidates
        diverse_candidates = self.diversity_manager.select_diverse_candidates(
            self.population, num_parents
        )
        
        self.selection_count += num_parents
        logger.debug(f"Selected {num_parents} diverse parents")
        
        return diverse_candidates
    
    def crossover_candidates(
        self, 
        parent1: Candidate, 
        parent2: Candidate, 
        child_id: str
    ) -> Candidate:
        """
        Perform crossover between two candidates.
        
        Args:
            parent1: First parent
            parent2: Second parent
            child_id: ID for the child candidate
            
        Returns:
            Child candidate
        """
        # Crossover code
        child_code = self._crossover_code(parent1.code, parent2.code)
        
        # Crossover prompt
        child_prompt = self._crossover_prompt(parent1.prompt, parent2.prompt)
        
        # Crossover tools
        child_tools = self._crossover_tools(parent1.tools, parent2.tools)
        
        # Crossover memory
        child_memory = self._crossover_memory(parent1.memory, parent2.memory)
        
        child = Candidate(
            id=child_id,
            code=child_code,
            prompt=child_prompt,
            tools=child_tools,
            memory=child_memory,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_id=f"{parent1.id}+{parent2.id}",
            mutation_type="crossover"
        )
        
        self.crossover_count += 1
        logger.debug(f"Created crossover child: {child_id}")
        
        return child
    
    def _crossover_code(self, code1: str, code2: str) -> str:
        """Crossover code using uniform crossover."""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        # Use uniform crossover
        max_lines = max(len(lines1), len(lines2))
        child_lines = []
        
        for i in range(max_lines):
            if i < len(lines1) and i < len(lines2):
                # Randomly choose from either parent
                child_lines.append(lines1[i] if np.random.random() < 0.5 else lines2[i])
            elif i < len(lines1):
                child_lines.append(lines1[i])
            else:
                child_lines.append(lines2[i])
        
        return '\n'.join(child_lines)
    
    def _crossover_prompt(self, prompt1: str, prompt2: str) -> str:
        """Crossover prompts using uniform crossover."""
        words1 = prompt1.split()
        words2 = prompt2.split()
        
        max_words = max(len(words1), len(words2))
        child_words = []
        
        for i in range(max_words):
            if i < len(words1) and i < len(words2):
                child_words.append(words1[i] if np.random.random() < 0.5 else words2[i])
            elif i < len(words1):
                child_words.append(words1[i])
            else:
                child_words.append(words2[i])
        
        return ' '.join(child_words)
    
    def _crossover_tools(self, tools1: Dict[str, Any], tools2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover tools using uniform crossover."""
        all_keys = set(tools1.keys()) | set(tools2.keys())
        child_tools = {}
        
        for key in all_keys:
            if key in tools1 and key in tools2:
                # Randomly choose from either parent
                child_tools[key] = tools1[key] if np.random.random() < 0.5 else tools2[key]
            elif key in tools1:
                child_tools[key] = tools1[key]
            else:
                child_tools[key] = tools2[key]
        
        return child_tools
    
    def _crossover_memory(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover memory using uniform crossover."""
        all_keys = set(memory1.keys()) | set(memory2.keys())
        child_memory = {}
        
        for key in all_keys:
            if key in memory1 and key in memory2:
                child_memory[key] = memory1[key] if np.random.random() < 0.5 else memory2[key]
            elif key in memory1:
                child_memory[key] = memory1[key]
            else:
                child_memory[key] = memory2[key]
        
        return child_memory
    
    async def evaluate_candidates(
        self, 
        candidates: List[Candidate], 
        evaluation_function: Callable
    ) -> None:
        """
        Evaluate candidates using the provided evaluation function.
        
        Args:
            candidates: Candidates to evaluate
            evaluation_function: Function to evaluate candidates
        """
        if self.config.parallel_evaluation:
            await self._evaluate_parallel(candidates, evaluation_function)
        else:
            await self._evaluate_sequential(candidates, evaluation_function)
    
    async def _evaluate_parallel(
        self, 
        candidates: List[Candidate], 
        evaluation_function: Callable
    ) -> None:
        """Evaluate candidates in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for candidate in candidates:
                future = executor.submit(evaluation_function, candidate)
                futures.append((candidate, future))
            
            for candidate, future in futures:
                try:
                    fitness_score = future.result(timeout=self.config.evaluation_timeout)
                    candidate.fitness_score = fitness_score
                    self.evaluation_count += 1
                except Exception as e:
                    logger.error(f"Evaluation failed for candidate {candidate.id}: {e}")
                    candidate.fitness_score = 0.0
    
    async def _evaluate_sequential(
        self, 
        candidates: List[Candidate], 
        evaluation_function: Callable
    ) -> None:
        """Evaluate candidates sequentially."""
        for candidate in candidates:
            try:
                fitness_score = evaluation_function(candidate)
                candidate.fitness_score = fitness_score
                self.evaluation_count += 1
            except Exception as e:
                logger.error(f"Evaluation failed for candidate {candidate.id}: {e}")
                candidate.fitness_score = 0.0
    
    def select_next_generation(self, new_candidates: List[Candidate]) -> List[Candidate]:
        """
        Select the next generation from current population and new candidates.
        
        Args:
            new_candidates: New candidates to consider
            
        Returns:
            Selected candidates for next generation
        """
        all_candidates = self.population + new_candidates
        
        if self.config.diversity_enabled and self.diversity_manager:
            # Use diversity-aware selection
            selected = self.diversity_manager.select_diverse_candidates(
                all_candidates, self.config.population_size
            )
            
            # Track cluster evolution
            clusters = self.diversity_manager.cluster_candidates(selected)
            self.diversity_manager.track_cluster_evolution(clusters)
            
        else:
            # Use tournament selection
            selected = []
            while len(selected) < self.config.population_size:
                tournament = np.random.choice(
                    all_candidates, 
                    min(self.config.tournament_size, len(all_candidates)), 
                    replace=False
                )
                winner = max(tournament, key=lambda x: x.fitness_score)
                selected.append(winner)
        
        self.population = selected
        self.generation += 1
        
        # Update best candidate
        current_best = max(self.population, key=lambda x: x.fitness_score)
        if not self.best_candidate or current_best.fitness_score > self.best_candidate.fitness_score:
            self.best_candidate = current_best
            logger.info(f"New best candidate: {current_best.id} (fitness: {current_best.fitness_score})")
        
        logger.info(f"Selected {len(selected)} candidates for generation {self.generation}")
        return selected
    
    def get_elite_candidates(self) -> List[Candidate]:
        """
        Get elite candidates (top performers).
        
        Returns:
            Elite candidates
        """
        sorted_candidates = sorted(
            self.population, 
            key=lambda x: x.fitness_score, 
            reverse=True
        )
        return sorted_candidates[:self.config.elite_size]
    
    def get_population_stats(self) -> Dict[str, Any]:
        """
        Get population statistics.
        
        Returns:
            Population statistics
        """
        if not self.population:
            return {}
        
        fitness_scores = [c.fitness_score for c in self.population]
        
        stats = {
            "population_size": len(self.population),
            "generation": self.generation,
            "best_fitness": max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
            "min_fitness": min(fitness_scores),
            "evaluation_count": self.evaluation_count,
            "selection_count": self.selection_count,
            "crossover_count": self.crossover_count,
            "mutation_count": self.mutation_count
        }
        
        if self.diversity_manager:
            diversity_metrics = self.diversity_manager.get_diversity_metrics()
            stats["diversity_metrics"] = diversity_metrics
        
        return stats
    
    def get_best_candidate(self) -> Optional[Candidate]:
        """
        Get the best candidate found so far.
        
        Returns:
            Best candidate or None
        """
        return self.best_candidate
    
    def check_convergence(self, fitness_threshold: float = 0.95) -> bool:
        """
        Check if evolution has converged.
        
        Args:
            fitness_threshold: Fitness threshold for convergence
            
        Returns:
            True if converged, False otherwise
        """
        if not self.best_candidate:
            return False
        
        # Check if best fitness exceeds threshold
        if self.best_candidate.fitness_score >= fitness_threshold:
            return True
        
        # Check if fitness has plateaued
        if len(self.population) >= 10:
            recent_fitness = [c.fitness_score for c in self.population[-10:]]
            variance = np.var(recent_fitness)
            if variance < 0.01:  # Low variance indicates plateau
                return True
        
        return False 