#!/usr/bin/env python3
"""
Diversity Manager for Evolutionary Agent - Clustering and diversity preservation.
"""
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity management."""
    min_cluster_size: int = 3
    max_clusters: int = 10
    similarity_threshold: float = 0.8
    diversity_weight: float = 0.1
    novelty_bonus: float = 0.2


class DiversityManager:
    """
    Manages diversity in the evolutionary population.
    Based on AlphaEvolve's approach to maintaining exploration.
    """
    
    def __init__(self, config: DiversityConfig):
        """
        Initialize the diversity manager.
        
        Args:
            config: Diversity configuration
        """
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.cluster_history = []
    
    def cluster_candidates(self, candidates: List[Any]) -> List[List[Any]]:
        """
        Cluster candidates based on code similarity.
        
        Args:
            candidates: List of candidates to cluster
            
        Returns:
            List of clusters (each cluster is a list of candidates)
        """
        if len(candidates) < self.config.min_cluster_size:
            return [candidates]
        
        # Extract code features
        code_texts = [c.code for c in candidates]
        
        try:
            # Vectorize code
            vectors = self.vectorizer.fit_transform(code_texts)
            
            # Determine number of clusters
            n_clusters = min(
                self.config.max_clusters,
                len(candidates) // self.config.min_cluster_size
            )
            
            if n_clusters < 2:
                return [candidates]
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Group candidates by cluster
            clusters = [[] for _ in range(n_clusters)]
            for candidate, label in zip(candidates, cluster_labels):
                clusters[label].append(candidate)
            
            # Filter out empty clusters
            clusters = [cluster for cluster in clusters if cluster]
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return [candidates]
    
    def select_diverse_candidates(
        self, 
        candidates: List[Any], 
        target_size: int
    ) -> List[Any]:
        """
        Select diverse candidates from the population.
        
        Args:
            candidates: List of candidates
            target_size: Target number of candidates to select
            
        Returns:
            Selected diverse candidates
        """
        if len(candidates) <= target_size:
            return candidates
        
        # Cluster candidates
        clusters = self.cluster_candidates(candidates)
        
        # Select candidates from each cluster
        selected = []
        candidates_per_cluster = max(1, target_size // len(clusters))
        
        for cluster in clusters:
            # Sort cluster by fitness
            cluster.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Select top candidates from this cluster
            cluster_selection = cluster[:candidates_per_cluster]
            selected.extend(cluster_selection)
        
        # If we need more candidates, add the best remaining ones
        if len(selected) < target_size:
            remaining = [c for c in candidates if c not in selected]
            remaining.sort(key=lambda x: x.fitness_score, reverse=True)
            selected.extend(remaining[:target_size - len(selected)])
        
        return selected[:target_size]
    
    def calculate_novelty_score(self, candidate: Any, population: List[Any]) -> float:
        """
        Calculate novelty score for a candidate.
        
        Args:
            candidate: Candidate to evaluate
            population: Current population
            
        Returns:
            Novelty score (higher = more novel)
        """
        if not population:
            return 1.0
        
        # Calculate similarity to existing candidates
        similarities = []
        
        for existing in population:
            similarity = self._calculate_similarity(candidate, existing)
            similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities)
        novelty = 1.0 - avg_similarity
        
        return novelty
    
    def _calculate_similarity(self, candidate1: Any, candidate2: Any) -> float:
        """
        Calculate similarity between two candidates.
        
        Args:
            candidate1: First candidate
            candidate2: Second candidate
            
        Returns:
            Similarity score (0-1)
        """
        # Combine code and prompt for similarity calculation
        text1 = candidate1.code + " " + candidate1.prompt
        text2 = candidate2.code + " " + candidate2.prompt
        
        try:
            # Vectorize texts
            vectors = self.vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.5  # Default similarity
    
    def calculate_diversity_fitness(
        self, 
        candidate: Any, 
        population: List[Any]
    ) -> float:
        """
        Calculate fitness with diversity bonus.
        
        Args:
            candidate: Candidate to evaluate
            population: Current population
            
        Returns:
            Diversity-adjusted fitness score
        """
        base_fitness = candidate.fitness_score
        novelty_score = self.calculate_novelty_score(candidate, population)
        
        # Apply diversity bonus
        diversity_bonus = self.config.diversity_weight * novelty_score
        
        # Apply novelty bonus for very novel candidates
        if novelty_score > 0.8:
            diversity_bonus += self.config.novelty_bonus
        
        return base_fitness + diversity_bonus
    
    def track_cluster_evolution(self, clusters: List[List[Any]]) -> None:
        """
        Track how clusters evolve over time.
        
        Args:
            clusters: Current clusters
        """
        cluster_info = {
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'avg_fitness': [np.mean([c.fitness_score for c in cluster]) for cluster in clusters]
        }
        
        self.cluster_history.append(cluster_info)
    
    def get_diversity_metrics(self) -> Dict[str, Any]:
        """
        Get diversity metrics for the population.
        
        Returns:
            Diversity metrics
        """
        if not self.cluster_history:
            return {}
        
        latest = self.cluster_history[-1]
        
        return {
            'num_clusters': latest['num_clusters'],
            'cluster_sizes': latest['cluster_sizes'],
            'avg_fitness_by_cluster': latest['avg_fitness'],
            'diversity_score': self._calculate_diversity_score(latest)
        }
    
    def _calculate_diversity_score(self, cluster_info: Dict[str, Any]) -> float:
        """
        Calculate overall diversity score.
        
        Args:
            cluster_info: Cluster information
            
        Returns:
            Diversity score
        """
        num_clusters = cluster_info['num_clusters']
        cluster_sizes = cluster_info['cluster_sizes']
        
        if num_clusters <= 1:
            return 0.0
        
        # Diversity is higher when clusters are more balanced
        total_size = sum(cluster_sizes)
        if total_size == 0:
            return 0.0
        
        # Calculate entropy-like diversity measure
        proportions = [size / total_size for size in cluster_sizes]
        diversity = -sum(p * np.log(p + 1e-10) for p in proportions)
        
        # Normalize by number of clusters
        max_diversity = np.log(num_clusters)
        if max_diversity > 0:
            diversity /= max_diversity
        
        return diversity


class HumanInteractionManager:
    """
    Manages human-in-the-loop interactions for planning decisions.
    """
    
    def __init__(self):
        """Initialize the human interaction manager."""
        self.interaction_history = []
    
    async def get_human_input(
        self, 
        input_type: str, 
        suggestions: Any,
        auto_accept: bool = False
    ) -> Any:
        """
        Get human input for planning decisions.
        
        Args:
            input_type: Type of input needed
            suggestions: Agent suggestions
            auto_accept: Whether to auto-accept suggestions
            
        Returns:
            Human-approved suggestions
        """
        if auto_accept:
            logger.info(f"Auto-accepting {input_type} suggestions")
            return suggestions
        
        # For now, simulate human input
        # In a real implementation, this would prompt the user
        logger.info(f"Human input needed for {input_type}")
        logger.info(f"Suggestions: {suggestions}")
        
        # Simulate human approval with some modifications
        approved = self._simulate_human_approval(input_type, suggestions)
        
        # Record interaction
        self.interaction_history.append({
            'type': input_type,
            'suggestions': suggestions,
            'approved': approved,
            'timestamp': np.datetime64('now')
        })
        
        return approved
    
    def _simulate_human_approval(self, input_type: str, suggestions: Any) -> Any:
        """
        Simulate human approval of suggestions.
        
        Args:
            input_type: Type of input
            suggestions: Agent suggestions
            
        Returns:
            Approved suggestions
        """
        # Simple simulation - in practice, this would be real human input
        if input_type == "edge_cases":
            # Human might add a few more edge cases
            if isinstance(suggestions, list):
                additional_cases = [
                    "Unicode handling edge cases",
                    "Memory overflow scenarios"
                ]
                return suggestions + additional_cases
        
        elif input_type == "tools":
            # Human might refine tool recommendations
            if isinstance(suggestions, dict) and 'tools' in suggestions:
                # Add a validation tool
                suggestions['tools'].append({
                    'name': 'validate_input',
                    'description': 'Validate and sanitize user inputs'
                })
        
        elif input_type == "spec":
            # Human might add performance requirements
            if isinstance(suggestions, dict):
                suggestions['performance_requirements'] = {
                    'max_execution_time': '5 seconds',
                    'memory_limit': '100MB'
                }
        
        return suggestions
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get summary of human interactions.
        
        Returns:
            Interaction summary
        """
        if not self.interaction_history:
            return {}
        
        return {
            'total_interactions': len(self.interaction_history),
            'interaction_types': list(set(h['type'] for h in self.interaction_history)),
            'recent_interactions': self.interaction_history[-5:]  # Last 5
        }


class MetaEvolutionLogger:
    """
    Logs and analyzes meta-evolution patterns.
    """
    
    def __init__(self):
        """Initialize the meta-evolution logger."""
        self.prompt_performance = []
        self.mutation_success_rates = []
        self.evolution_patterns = []
    
    def log_prompt_performance(
        self, 
        prompt_template: str, 
        success_rate: float,
        avg_fitness: float
    ) -> None:
        """
        Log performance of a prompt template.
        
        Args:
            prompt_template: The prompt template
            success_rate: Success rate of mutations
            avg_fitness: Average fitness of generated candidates
        """
        self.prompt_performance.append({
            'template': prompt_template,
            'success_rate': success_rate,
            'avg_fitness': avg_fitness,
            'timestamp': np.datetime64('now')
        })
    
    def log_mutation_success(
        self, 
        mutation_type: str, 
        success: bool,
        fitness_gain: float
    ) -> None:
        """
        Log success of a mutation.
        
        Args:
            mutation_type: Type of mutation
            success: Whether mutation was successful
            fitness_gain: Fitness improvement
        """
        self.mutation_success_rates.append({
            'type': mutation_type,
            'success': success,
            'fitness_gain': fitness_gain,
            'timestamp': np.datetime64('now')
        })
    
    def get_best_prompts(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing prompt templates.
        
        Args:
            top_k: Number of top prompts to return
            
        Returns:
            Best performing prompts
        """
        if not self.prompt_performance:
            return []
        
        # Sort by success rate and average fitness
        sorted_prompts = sorted(
            self.prompt_performance,
            key=lambda x: x['success_rate'] * x['avg_fitness'],
            reverse=True
        )
        
        return sorted_prompts[:top_k]
    
    def get_mutation_insights(self) -> Dict[str, Any]:
        """
        Get insights about mutation performance.
        
        Returns:
            Mutation insights
        """
        if not self.mutation_success_rates:
            return {}
        
        # Group by mutation type
        type_stats = {}
        for record in self.mutation_success_rates:
            mut_type = record['type']
            if mut_type not in type_stats:
                type_stats[mut_type] = {
                    'total': 0,
                    'successful': 0,
                    'avg_fitness_gain': 0.0
                }
            
            type_stats[mut_type]['total'] += 1
            if record['success']:
                type_stats[mut_type]['successful'] += 1
            type_stats[mut_type]['avg_fitness_gain'] += record['fitness_gain']
        
        # Calculate averages
        for mut_type in type_stats:
            total = type_stats[mut_type]['total']
            type_stats[mut_type]['success_rate'] = type_stats[mut_type]['successful'] / total
            type_stats[mut_type]['avg_fitness_gain'] /= total
        
        return type_stats 