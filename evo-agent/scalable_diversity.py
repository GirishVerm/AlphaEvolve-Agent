#!/usr/bin/env python3
"""
Scalable Diversity Management with Incremental Clustering and Novelty Archives.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ScalableDiversityConfig:
    """Configuration for scalable diversity management."""
    max_candidates: int = 1000
    novelty_threshold: float = 0.8
    archive_size: int = 100
    cluster_cache_size: int = 50
    incremental_clustering: bool = True
    birch_threshold: float = 0.5
    min_cluster_size: int = 3
    similarity_cache_size: int = 1000


class NoveltyArchive:
    """
    Novelty archive for maintaining diverse solutions.
    """
    
    def __init__(self, config: ScalableDiversityConfig):
        """
        Initialize novelty archive.
        
        Args:
            config: Scalable diversity configuration
        """
        self.config = config
        self.archive: List[Tuple[np.ndarray, Any]] = []  # (features, candidate)
        self.archive_hashes: set = set()
        self.novelty_scores: Dict[str, float] = {}
        
        # Caching
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}
    
    def add_candidate(self, candidate: Any, features: np.ndarray) -> bool:
        """
        Add candidate to archive if it's novel enough.
        
        Args:
            candidate: Candidate to add
            features: Feature vector
            
        Returns:
            True if candidate was added
        """
        candidate_hash = self._get_candidate_hash(candidate)
        
        if candidate_hash in self.archive_hashes:
            return False
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(features)
        
        if novelty_score >= self.config.novelty_threshold:
            self.archive.append((features, candidate))
            self.archive_hashes.add(candidate_hash)
            self.novelty_scores[candidate_hash] = novelty_score
            
            # Maintain archive size
            if len(self.archive) > self.config.archive_size:
                self._prune_archive()
            
            logger.debug(f"Added novel candidate with score {novelty_score:.3f}")
            return True
        
        return False
    
    def _get_candidate_hash(self, candidate: Any) -> str:
        """Get hash for candidate."""
        content = f"{candidate.code}:{candidate.prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_novelty_score(self, features: np.ndarray) -> float:
        """
        Calculate novelty score based on similarity to archive.
        
        Args:
            features: Feature vector
            
        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if not self.archive:
            return 1.0
        
        # Calculate similarities to archive members
        similarities = []
        for archive_features, _ in self.archive:
            similarity = cosine_similarity(
                features.reshape(1, -1), 
                archive_features.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty_score = 1.0 - max_similarity
        
        return novelty_score
    
    def _prune_archive(self) -> None:
        """Prune archive to maintain size limit."""
        if len(self.archive) <= self.config.archive_size:
            return
        
        # Remove least novel candidates
        archive_with_scores = []
        for i, (features, candidate) in enumerate(self.archive):
            candidate_hash = self._get_candidate_hash(candidate)
            score = self.novelty_scores.get(candidate_hash, 0.0)
            archive_with_scores.append((score, i, features, candidate))
        
        # Sort by novelty score (ascending)
        archive_with_scores.sort()
        
        # Keep top candidates
        keep_indices = set(item[1] for item in archive_with_scores[-self.config.archive_size:])
        
        # Rebuild archive
        new_archive = []
        new_hashes = set()
        new_scores = {}
        
        for i, (features, candidate) in enumerate(self.archive):
            if i in keep_indices:
                new_archive.append((features, candidate))
                candidate_hash = self._get_candidate_hash(candidate)
                new_hashes.add(candidate_hash)
                new_scores[candidate_hash] = self.novelty_scores.get(candidate_hash, 0.0)
        
        self.archive = new_archive
        self.archive_hashes = new_hashes
        self.novelty_scores = new_scores
    
    def get_archive_candidates(self) -> List[Any]:
        """
        Get candidates in archive.
        
        Returns:
            List of archive candidates
        """
        return [candidate for _, candidate in self.archive]
    
    def get_archive_features(self) -> List[np.ndarray]:
        """
        Get feature vectors in archive.
        
        Returns:
            List of feature vectors
        """
        return [features for features, _ in self.archive]
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get archive statistics.
        
        Returns:
            Archive statistics
        """
        if not self.archive:
            return {}
        
        novelty_scores = list(self.novelty_scores.values())
        
        return {
            'archive_size': len(self.archive),
            'avg_novelty_score': np.mean(novelty_scores),
            'min_novelty_score': np.min(novelty_scores),
            'max_novelty_score': np.max(novelty_scores),
            'novelty_threshold': self.config.novelty_threshold
        }


class IncrementalClusterer:
    """
    Incremental clustering for large populations.
    """
    
    def __init__(self, config: ScalableDiversityConfig):
        """
        Initialize incremental clusterer.
        
        Args:
            config: Scalable diversity configuration
        """
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Clustering models
        self.birch_model = Birch(
            threshold=self.config.birch_threshold,
            n_clusters=None
        )
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=min(10, self.config.max_candidates // 10),
            batch_size=100
        )
        
        # Caching
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.cluster_cache: Dict[str, List[List[Any]]] = {}
        self.last_cluster_time = 0
        
        # State
        self.is_fitted = False
        self.cluster_centers = None
    
    def extract_features(self, candidates: List[Any]) -> np.ndarray:
        """
        Extract features from candidates.
        
        Args:
            candidates: List of candidates
            
        Returns:
            Feature matrix
        """
        # Combine code and prompts
        texts = []
        for candidate in candidates:
            text = f"{candidate.code} {candidate.prompt}"
            texts.append(text)
        
        # Vectorize
        features = self.vectorizer.fit_transform(texts)
        return features.toarray()
    
    def cluster_candidates(self, candidates: List[Any]) -> List[List[Any]]:
        """
        Cluster candidates using incremental clustering.
        
        Args:
            candidates: List of candidates
            
        Returns:
            List of clusters
        """
        if len(candidates) < self.config.min_cluster_size:
            return [candidates]
        
        # Extract features
        features = self.extract_features(candidates)
        
        # Use incremental clustering
        if self.config.incremental_clustering:
            return self._incremental_cluster(features, candidates)
        else:
            return self._batch_cluster(features, candidates)
    
    def _incremental_cluster(self, features: np.ndarray, candidates: List[Any]) -> List[List[Any]]:
        """
        Perform incremental clustering.
        
        Args:
            features: Feature matrix
            candidates: List of candidates
            
        Returns:
            List of clusters
        """
        # Use BIRCH for incremental clustering
        cluster_labels = self.birch_model.fit_predict(features)
        
        # Group candidates by cluster
        clusters = defaultdict(list)
        for candidate, label in zip(candidates, cluster_labels):
            clusters[label].append(candidate)
        
        # Convert to list of lists
        cluster_list = list(clusters.values())
        
        # Filter out very small clusters
        filtered_clusters = [
            cluster for cluster in cluster_list
            if len(cluster) >= self.config.min_cluster_size
        ]
        
        # If no clusters meet minimum size, return single cluster
        if not filtered_clusters:
            return [candidates]
        
        return filtered_clusters
    
    def _batch_cluster(self, features: np.ndarray, candidates: List[Any]) -> List[List[Any]]:
        """
        Perform batch clustering.
        
        Args:
            features: Feature matrix
            candidates: List of candidates
            
        Returns:
            List of clusters
        """
        # Use MiniBatchKMeans for batch clustering
        cluster_labels = self.kmeans_model.fit_predict(features)
        
        # Group candidates by cluster
        clusters = defaultdict(list)
        for candidate, label in zip(candidates, cluster_labels):
            clusters[label].append(candidate)
        
        # Convert to list of lists
        cluster_list = list(clusters.values())
        
        # Filter out very small clusters
        filtered_clusters = [
            cluster for cluster in cluster_list
            if len(cluster) >= self.config.min_cluster_size
        ]
        
        # If no clusters meet minimum size, return single cluster
        if not filtered_clusters:
            return [candidates]
        
        return filtered_clusters
    
    def update_cluster_centers(self, features: np.ndarray) -> None:
        """
        Update cluster centers incrementally.
        
        Args:
            features: New feature matrix
        """
        if not self.is_fitted:
            self.birch_model.fit(features)
            self.is_fitted = True
        else:
            # Partial fit for incremental learning
            self.birch_model.partial_fit(features)
        
        # Update cluster centers
        if hasattr(self.birch_model, 'cluster_centers_'):
            self.cluster_centers = self.birch_model.cluster_centers_


class ScalableDiversityManager:
    """
    Scalable diversity management with incremental clustering and novelty archives.
    """
    
    def __init__(self, config: ScalableDiversityConfig):
        """
        Initialize scalable diversity manager.
        
        Args:
            config: Scalable diversity configuration
        """
        self.config = config
        self.novelty_archive = NoveltyArchive(config)
        self.incremental_clusterer = IncrementalClusterer(config)
        
        # State
        self.population_history: List[List[Any]] = []
        self.cluster_history: List[List[List[Any]]] = []
        self.diversity_metrics: List[Dict[str, Any]] = []
    
    def add_candidates(self, candidates: List[Any]) -> None:
        """
        Add candidates to diversity management.
        
        Args:
            candidates: List of candidates
        """
        # Extract features
        features = self.incremental_clusterer.extract_features(candidates)
        
        # Add to novelty archive
        for candidate, feature_vector in zip(candidates, features):
            self.novelty_archive.add_candidate(candidate, feature_vector)
        
        # Update cluster centers
        self.incremental_clusterer.update_cluster_centers(features)
        
        # Store history
        self.population_history.append(candidates)
    
    def cluster_candidates(self, candidates: List[Any]) -> List[List[Any]]:
        """
        Cluster candidates using scalable clustering.
        
        Args:
            candidates: List of candidates
            
        Returns:
            List of clusters
        """
        clusters = self.incremental_clusterer.cluster_candidates(candidates)
        self.cluster_history.append(clusters)
        
        return clusters
    
    def select_diverse_candidates(
        self, 
        candidates: List[Any], 
        target_size: int
    ) -> List[Any]:
        """
        Select diverse candidates using novelty archive and clustering.
        
        Args:
            candidates: List of candidates
            target_size: Target number of candidates
            
        Returns:
            Selected diverse candidates
        """
        if len(candidates) <= target_size:
            return candidates
        
        # Get archive candidates
        archive_candidates = self.novelty_archive.get_archive_candidates()
        
        # Combine with current candidates
        all_candidates = candidates + archive_candidates
        
        # Cluster all candidates
        clusters = self.cluster_candidates(all_candidates)
        
        # Select from each cluster
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
            remaining = [c for c in all_candidates if c not in selected]
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
            Novelty score
        """
        # Extract features
        all_candidates = [candidate] + population
        features = self.incremental_clusterer.extract_features(all_candidates)
        candidate_features = features[0]
        population_features = features[1:]
        
        if len(population_features) == 0:
            return 1.0
        
        # Calculate similarities to population
        similarities = []
        for pop_features in population_features:
            similarity = cosine_similarity(
                candidate_features.reshape(1, -1),
                pop_features.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities)
        novelty_score = 1.0 - avg_similarity
        
        return novelty_score
    
    def get_diversity_metrics(self) -> Dict[str, Any]:
        """
        Get diversity metrics.
        
        Returns:
            Diversity metrics
        """
        archive_stats = self.novelty_archive.get_archive_stats()
        
        # Calculate cluster diversity
        cluster_diversity = 0.0
        if self.cluster_history:
            latest_clusters = self.cluster_history[-1]
            if len(latest_clusters) > 1:
                # Calculate entropy-like diversity
                total_size = sum(len(cluster) for cluster in latest_clusters)
                proportions = [len(cluster) / total_size for cluster in latest_clusters]
                cluster_diversity = -sum(p * np.log(p + 1e-10) for p in proportions)
        
        return {
            **archive_stats,
            'cluster_diversity': cluster_diversity,
            'num_clusters': len(self.cluster_history[-1]) if self.cluster_history else 0,
            'population_history_size': len(self.population_history),
            'cluster_history_size': len(self.cluster_history)
        }
    
    def get_archive_candidates(self) -> List[Any]:
        """
        Get candidates from novelty archive.
        
        Returns:
            List of archive candidates
        """
        return self.novelty_archive.get_archive_candidates()
    
    def clear_cache(self) -> None:
        """Clear feature and similarity caches."""
        self.incremental_clusterer.feature_cache.clear()
        self.novelty_archive.similarity_cache.clear()
        self.novelty_archive.feature_cache.clear()
        
        logger.info("Cleared diversity management caches")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Memory usage statistics
        """
        return {
            'feature_cache_size': len(self.incremental_clusterer.feature_cache),
            'similarity_cache_size': len(self.novelty_archive.similarity_cache),
            'archive_size': len(self.novelty_archive.archive),
            'population_history_size': len(self.population_history),
            'cluster_history_size': len(self.cluster_history)
        } 