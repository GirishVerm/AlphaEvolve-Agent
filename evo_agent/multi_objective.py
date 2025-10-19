#!/usr/bin/env python3
"""
Multi-Objective Evaluation Framework with Pareto Fronts.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Selection modes for multi-objective optimization."""
    PARETO = "pareto"
    WEIGHTED_SUM = "weighted_sum"
    TOURNAMENT = "tournament"
    NSGA2 = "nsga2"


@dataclass
class Objective:
    """Represents an optimization objective."""
    name: str
    weight: float = 1.0
    minimize: bool = False
    bounds: Optional[Tuple[float, float]] = None
    priority: int = 0  # Higher priority objectives are considered first


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: List[Objective]
    selection_mode: SelectionMode = SelectionMode.PARETO
    pareto_epsilon: float = 0.01  # Epsilon for Pareto dominance
    tournament_size: int = 3
    crowding_distance: bool = True
    archive_size: int = 100


class ParetoFront:
    """
    Manages Pareto front for multi-objective optimization.
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        """
        Initialize Pareto front.
        
        Args:
            config: Multi-objective configuration
        """
        self.config = config
        self.front: List[Tuple[np.ndarray, Any]] = []  # (fitness_vector, candidate)
        self.archive: List[Tuple[np.ndarray, Any]] = []
    
    def add_candidate(self, fitness_vector: np.ndarray, candidate: Any) -> bool:
        """
        Add candidate to Pareto front.
        
        Args:
            fitness_vector: Vector of objective values
            candidate: Candidate object
            
        Returns:
            True if candidate was added to front
        """
        # Normalize fitness vector
        normalized_vector = self._normalize_fitness(fitness_vector)
        
        # Check if candidate dominates any existing front members
        dominated_indices = []
        for i, (front_vector, _) in enumerate(self.front):
            if self._dominates(normalized_vector, front_vector):
                dominated_indices.append(i)
        
        # Remove dominated candidates
        for index in reversed(dominated_indices):
            self.front.pop(index)
        
        # Check if candidate is dominated by any front member
        for front_vector, _ in self.front:
            if self._dominates(front_vector, normalized_vector):
                return False
        
        # Add candidate to front
        self.front.append((normalized_vector, candidate))
        
        # Update archive
        self._update_archive(normalized_vector, candidate)
        
        return True
    
    def _dominates(self, vector1: np.ndarray, vector2: np.ndarray) -> bool:
        """
        Check if vector1 dominates vector2.
        
        Args:
            vector1: First fitness vector
            vector2: Second fitness vector
            
        Returns:
            True if vector1 dominates vector2
        """
        # Check if vector1 is at least as good as vector2 in all objectives
        at_least_as_good = np.all(vector1 <= vector2 + self.config.pareto_epsilon)
        
        # Check if vector1 is strictly better in at least one objective
        strictly_better = np.any(vector1 < vector2 - self.config.pareto_epsilon)
        
        return at_least_as_good and strictly_better
    
    def _normalize_fitness(self, fitness_vector: np.ndarray) -> np.ndarray:
        """
        Normalize fitness vector based on objective bounds.
        
        Args:
            fitness_vector: Raw fitness vector
            
        Returns:
            Normalized fitness vector
        """
        normalized = np.copy(fitness_vector)
        
        for i, objective in enumerate(self.config.objectives):
            if objective.bounds:
                min_val, max_val = objective.bounds
                if max_val > min_val:
                    normalized[i] = (fitness_vector[i] - min_val) / (max_val - min_val)
            
            # Flip sign for minimization objectives
            if objective.minimize:
                normalized[i] = -normalized[i]
        
        return normalized
    
    def _update_archive(self, fitness_vector: np.ndarray, candidate: Any) -> None:
        """
        Update archive with new candidate.
        
        Args:
            fitness_vector: Fitness vector
            candidate: Candidate object
        """
        self.archive.append((fitness_vector, candidate))
        
        # Limit archive size
        if len(self.archive) > self.config.archive_size:
            # Remove least diverse candidate
            self._prune_archive()
    
    def _prune_archive(self) -> None:
        """Prune archive to maintain diversity."""
        if len(self.archive) <= self.config.archive_size:
            return
        
        # Calculate crowding distances
        fitness_vectors = np.array([vector for vector, _ in self.archive])
        crowding_distances = self._calculate_crowding_distance(fitness_vectors)
        
        # Remove candidate with lowest crowding distance
        min_index = np.argmin(crowding_distances)
        self.archive.pop(min_index)
    
    def _calculate_crowding_distance(self, fitness_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for diversity preservation.
        
        Args:
            fitness_vectors: Array of fitness vectors
            
        Returns:
            Crowding distances
        """
        if len(fitness_vectors) <= 2:
            return np.ones(len(fitness_vectors))
        
        distances = np.zeros(len(fitness_vectors))
        
        for i in range(fitness_vectors.shape[1]):
            # Sort by objective i
            sorted_indices = np.argsort(fitness_vectors[:, i])
            sorted_vectors = fitness_vectors[sorted_indices]
            
            # Calculate distances
            for j in range(len(sorted_vectors)):
                if j == 0 or j == len(sorted_vectors) - 1:
                    distances[sorted_indices[j]] = np.inf
                else:
                    distances[sorted_indices[j]] += (
                        sorted_vectors[j + 1, i] - sorted_vectors[j - 1, i]
                    )
        
        return distances
    
    def get_front_candidates(self) -> List[Any]:
        """
        Get candidates in Pareto front.
        
        Returns:
            List of candidates in Pareto front
        """
        return [candidate for _, candidate in self.front]
    
    def get_archive_candidates(self) -> List[Any]:
        """
        Get candidates in archive.
        
        Returns:
            List of candidates in archive
        """
        return [candidate for _, candidate in self.archive]
    
    def get_front_fitness(self) -> List[np.ndarray]:
        """
        Get fitness vectors of Pareto front.
        
        Returns:
            List of fitness vectors
        """
        return [vector for vector, _ in self.front]
    
    def plot_front(self, save_path: Optional[str] = None) -> None:
        """
        Plot Pareto front (for 2-3 objectives).
        
        Args:
            save_path: Optional path to save plot
        """
        if len(self.config.objectives) > 3:
            logger.warning("Can only plot 2-3 objectives")
            return
        
        fitness_vectors = np.array(self.get_front_fitness())
        
        if len(fitness_vectors) == 0:
            logger.warning("No candidates in Pareto front")
            return
        
        fig = plt.figure(figsize=(10, 8))
        
        if len(self.config.objectives) == 2:
            plt.scatter(fitness_vectors[:, 0], fitness_vectors[:, 1], alpha=0.7)
            plt.xlabel(self.config.objectives[0].name)
            plt.ylabel(self.config.objectives[1].name)
            plt.title("Pareto Front")
            
        elif len(self.config.objectives) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(fitness_vectors[:, 0], fitness_vectors[:, 1], fitness_vectors[:, 2], alpha=0.7)
            ax.set_xlabel(self.config.objectives[0].name)
            ax.set_ylabel(self.config.objectives[1].name)
            ax.set_zlabel(self.config.objectives[2].name)
            ax.set_title("Pareto Front")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class MultiObjectiveEvaluator:
    """
    Multi-objective evaluation framework.
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        """
        Initialize multi-objective evaluator.
        
        Args:
            config: Multi-objective configuration
        """
        self.config = config
        self.pareto_front = ParetoFront(config)
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_candidate(
        self, 
        candidate: Any, 
        evaluation_functions: List[Callable]
    ) -> np.ndarray:
        """
        Evaluate candidate across multiple objectives.
        
        Args:
            candidate: Candidate to evaluate
            evaluation_functions: List of evaluation functions
            
        Returns:
            Vector of objective values
        """
        if len(evaluation_functions) != len(self.config.objectives):
            raise ValueError("Number of evaluation functions must match number of objectives")
        
        fitness_vector = np.zeros(len(self.config.objectives))
        
        for i, eval_func in enumerate(evaluation_functions):
            try:
                fitness_vector[i] = eval_func(candidate)
            except Exception as e:
                logger.error(f"Evaluation function {i} failed: {e}")
                fitness_vector[i] = 0.0  # Default to worst value
        
        # Record evaluation
        self.evaluation_history.append({
            'candidate_id': getattr(candidate, 'id', 'unknown'),
            'fitness_vector': fitness_vector.copy(),
            'timestamp': np.datetime64('now')
        })
        
        return fitness_vector
    
    def select_candidates(
        self, 
        candidates: List[Any], 
        num_select: int
    ) -> List[Any]:
        """
        Select candidates based on multi-objective criteria.
        
        Args:
            candidates: List of candidates
            num_select: Number of candidates to select
            
        Returns:
            Selected candidates
        """
        if self.config.selection_mode == SelectionMode.PARETO:
            return self._pareto_selection(candidates, num_select)
        elif self.config.selection_mode == SelectionMode.WEIGHTED_SUM:
            return self._weighted_sum_selection(candidates, num_select)
        elif self.config.selection_mode == SelectionMode.TOURNAMENT:
            return self._tournament_selection(candidates, num_select)
        elif self.config.selection_mode == SelectionMode.NSGA2:
            return self._nsga2_selection(candidates, num_select)
        else:
            raise ValueError(f"Unknown selection mode: {self.config.selection_mode}")
    
    def _pareto_selection(self, candidates: List[Any], num_select: int) -> List[Any]:
        """Select candidates using Pareto dominance."""
        # Add all candidates to Pareto front
        for candidate in candidates:
            # This would need fitness vectors from previous evaluation
            # For now, use a placeholder
            fitness_vector = np.random.random(len(self.config.objectives))
            self.pareto_front.add_candidate(fitness_vector, candidate)
        
        # Get Pareto front candidates
        front_candidates = self.pareto_front.get_front_candidates()
        
        # If we need more candidates, add from archive
        if len(front_candidates) < num_select:
            archive_candidates = self.pareto_front.get_archive_candidates()
            front_candidates.extend(archive_candidates[:num_select - len(front_candidates)])
        
        return front_candidates[:num_select]
    
    def _weighted_sum_selection(self, candidates: List[Any], num_select: int) -> List[Any]:
        """Select candidates using weighted sum of objectives."""
        # This would need fitness vectors from previous evaluation
        # For now, return random selection
        return np.random.choice(candidates, min(num_select, len(candidates)), replace=False).tolist()
    
    def _tournament_selection(self, candidates: List[Any], num_select: int) -> List[Any]:
        """Select candidates using tournament selection."""
        selected = []
        
        for _ in range(num_select):
            tournament = np.random.choice(
                candidates, 
                min(self.config.tournament_size, len(candidates)), 
                replace=False
            )
            
            # Select winner based on Pareto dominance
            winner = tournament[0]  # Placeholder - would need fitness comparison
            selected.append(winner)
        
        return selected
    
    def _nsga2_selection(self, candidates: List[Any], num_select: int) -> List[Any]:
        """Select candidates using NSGA-II algorithm."""
        # Simplified NSGA-II implementation
        # In practice, this would implement the full NSGA-II algorithm
        return self._pareto_selection(candidates, num_select)
    
    def get_diversity_metrics(self) -> Dict[str, Any]:
        """
        Get diversity metrics for Pareto front.
        
        Returns:
            Diversity metrics
        """
        if not self.pareto_front.front:
            return {}
        
        fitness_vectors = np.array(self.pareto_front.get_front_fitness())
        
        # Calculate hypervolume (simplified)
        if len(fitness_vectors) > 0:
            # Use minimum values as reference point
            reference_point = np.min(fitness_vectors, axis=0) - 0.1
            hypervolume = self._calculate_hypervolume(fitness_vectors, reference_point)
        else:
            hypervolume = 0.0
        
        # Calculate spread
        if len(fitness_vectors) > 1:
            spread = self._calculate_spread(fitness_vectors)
        else:
            spread = 0.0
        
        return {
            'front_size': len(self.pareto_front.front),
            'archive_size': len(self.pareto_front.archive),
            'hypervolume': hypervolume,
            'spread': spread,
            'evaluations': len(self.evaluation_history)
        }
    
    def _calculate_hypervolume(
        self, 
        fitness_vectors: np.ndarray, 
        reference_point: np.ndarray
    ) -> float:
        """
        Calculate hypervolume indicator.
        
        Args:
            fitness_vectors: Pareto front fitness vectors
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        # Simplified hypervolume calculation
        # In practice, use a proper hypervolume library
        volume = 1.0
        for i in range(fitness_vectors.shape[1]):
            max_val = np.max(fitness_vectors[:, i])
            min_val = np.min(fitness_vectors[:, i])
            volume *= (max_val - min_val)
        
        return volume
    
    def _calculate_spread(self, fitness_vectors: np.ndarray) -> float:
        """
        Calculate spread of Pareto front.
        
        Args:
            fitness_vectors: Pareto front fitness vectors
            
        Returns:
            Spread value
        """
        # Calculate average distance between points
        distances = pdist(fitness_vectors)
        return np.mean(distances) if len(distances) > 0 else 0.0


class AdaptiveMutationScheduler:
    """
    Adaptive mutation scheduler based on convergence.
    """
    
    def __init__(self, initial_rate: float = 0.3, min_rate: float = 0.05):
        """
        Initialize adaptive mutation scheduler.
        
        Args:
            initial_rate: Initial mutation rate
            min_rate: Minimum mutation rate
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.current_rate = initial_rate
        self.fitness_history: List[float] = []
        self.window_size = 10
        self.convergence_threshold = 0.01
    
    def update_rate(self, best_fitness: float) -> float:
        """
        Update mutation rate based on fitness improvement.
        
        Args:
            best_fitness: Best fitness in current generation
            
        Returns:
            Updated mutation rate
        """
        self.fitness_history.append(best_fitness)
        
        if len(self.fitness_history) < self.window_size:
            return self.current_rate
        
        # Calculate rolling average improvement
        recent_fitness = self.fitness_history[-self.window_size:]
        if len(recent_fitness) >= 2:
            improvements = [recent_fitness[i] - recent_fitness[i-1] 
                          for i in range(1, len(recent_fitness))]
            avg_improvement = np.mean(improvements)
            
            # Adjust mutation rate based on improvement
            if avg_improvement < self.convergence_threshold:
                # Reduce mutation rate when convergence is slow
                self.current_rate = max(self.min_rate, self.current_rate * 0.9)
            else:
                # Increase mutation rate when making good progress
                self.current_rate = min(0.5, self.current_rate * 1.1)
        
        return self.current_rate
    
    def get_patch_size_factor(self) -> float:
        """
        Get patch size factor based on convergence.
        
        Returns:
            Patch size factor (0.1 to 1.0)
        """
        # Smaller patches as we converge
        return max(0.1, self.current_rate)
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_rate = self.initial_rate
        self.fitness_history.clear() 