#!/usr/bin/env python3
"""
Metrics Exporter - Rich metrics API for observability and monitoring.
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics export."""
    export_interval: int = 60  # seconds
    prometheus_enabled: bool = False
    json_enabled: bool = True
    log_enabled: bool = True
    metrics_file: Optional[str] = None
    backup_metrics: bool = True


class MetricsExporter:
    """
    Exports metrics for monitoring and observability.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics exporter.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.last_export = time.time()
        
        # Component metrics
        self.llm_metrics = {}
        self.evolution_metrics = {}
        self.diversity_metrics = {}
        self.evaluation_metrics = {}
        self.population_metrics = {}
        self.prompt_metrics = {}
    
    def update_llm_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update LLM metrics.
        
        Args:
            metrics: LLM metrics
        """
        self.llm_metrics.update(metrics)
        self.llm_metrics["timestamp"] = datetime.now().isoformat()
    
    def update_evolution_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update evolution metrics.
        
        Args:
            metrics: Evolution metrics
        """
        self.evolution_metrics.update(metrics)
        self.evolution_metrics["timestamp"] = datetime.now().isoformat()
    
    def update_diversity_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update diversity metrics.
        
        Args:
            metrics: Diversity metrics
        """
        self.diversity_metrics.update(metrics)
        self.diversity_metrics["timestamp"] = datetime.now().isoformat()
    
    def update_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update evaluation metrics.
        
        Args:
            metrics: Evaluation metrics
        """
        self.evaluation_metrics.update(metrics)
        self.evaluation_metrics["timestamp"] = datetime.now().isoformat()
    
    def update_population_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update population metrics.
        
        Args:
            metrics: Population metrics
        """
        self.population_metrics.update(metrics)
        self.population_metrics["timestamp"] = datetime.now().isoformat()
    
    def update_prompt_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update prompt metrics.
        
        Args:
            metrics: Prompt metrics
        """
        self.prompt_metrics.update(metrics)
        self.prompt_metrics["timestamp"] = datetime.now().isoformat()
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics from all components.
        
        Returns:
            Aggregated metrics
        """
        current_time = time.time()
        runtime = current_time - self.start_time
        
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": runtime,
            "runtime_hours": runtime / 3600,
            "llm_metrics": self.llm_metrics,
            "evolution_metrics": self.evolution_metrics,
            "diversity_metrics": self.diversity_metrics,
            "evaluation_metrics": self.evaluation_metrics,
            "population_metrics": self.population_metrics,
            "prompt_metrics": self.prompt_metrics
        }
        
        # Calculate summary statistics
        if self.metrics_history:
            aggregated["summary"] = self._calculate_summary_stats()
        
        return aggregated
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from metrics history."""
        if not self.metrics_history:
            return {}
        
        # Extract fitness scores over time
        fitness_scores = []
        for metrics in self.metrics_history:
            if "population_metrics" in metrics:
                pop_metrics = metrics["population_metrics"]
                if "best_fitness" in pop_metrics:
                    fitness_scores.append(pop_metrics["best_fitness"])
        
        if not fitness_scores:
            return {}
        
        return {
            "total_generations": len(fitness_scores),
            "best_fitness_ever": max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "fitness_improvement": max(fitness_scores) - min(fitness_scores),
            "convergence_rate": self._calculate_convergence_rate(fitness_scores)
        }
    
    def _calculate_convergence_rate(self, fitness_scores: List[float]) -> float:
        """Calculate convergence rate based on fitness improvement."""
        if len(fitness_scores) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(fitness_scores)):
            improvement = fitness_scores[i] - fitness_scores[i-1]
            improvements.append(improvement)
        
        # Calculate rate of improvement
        avg_improvement = np.mean(improvements)
        return avg_improvement
    
    def export_metrics(self, force: bool = False) -> None:
        """
        Export metrics if interval has passed or forced.
        
        Args:
            force: Force export regardless of interval
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_export) < self.config.export_interval:
            return
        
        metrics = self.get_aggregated_metrics()
        self.metrics_history.append(metrics)
        
        # Export to different formats
        if self.config.json_enabled:
            self._export_json(metrics)
        
        if self.config.log_enabled:
            self._export_log(metrics)
        
        if self.config.prometheus_enabled:
            self._export_prometheus(metrics)
        
        self.last_export = current_time
        logger.info("Metrics exported successfully")
    
    def _export_json(self, metrics: Dict[str, Any]) -> None:
        """Export metrics as JSON."""
        if self.config.metrics_file:
            with open(self.config.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Also save to timestamped file for backup
        if self.config.backup_metrics:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"metrics_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def _export_log(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to log."""
        logger.info("=== Metrics Export ===")
        
        # Log key metrics
        if "population_metrics" in metrics:
            pop_metrics = metrics["population_metrics"]
            logger.info(f"Generation: {pop_metrics.get('generation', 'N/A')}")
            logger.info(f"Best Fitness: {pop_metrics.get('best_fitness', 'N/A')}")
            logger.info(f"Avg Fitness: {pop_metrics.get('avg_fitness', 'N/A')}")
            logger.info(f"Population Size: {pop_metrics.get('population_size', 'N/A')}")
        
        if "llm_metrics" in metrics:
            llm_metrics = metrics["llm_metrics"]
            logger.info(f"LLM Requests: {llm_metrics.get('request_count', 'N/A')}")
            logger.info(f"LLM Success Rate: {llm_metrics.get('success_rate', 'N/A')}")
        
        if "summary" in metrics:
            summary = metrics["summary"]
            logger.info(f"Total Generations: {summary.get('total_generations', 'N/A')}")
            logger.info(f"Best Fitness Ever: {summary.get('best_fitness_ever', 'N/A')}")
            logger.info(f"Convergence Rate: {summary.get('convergence_rate', 'N/A')}")
    
    def _export_prometheus(self, metrics: Dict[str, Any]) -> None:
        """Export metrics in Prometheus format."""
        prometheus_lines = []
        
        # Add timestamp
        prometheus_lines.append(f"# HELP evo_agent_timestamp Current timestamp")
        prometheus_lines.append(f"# TYPE evo_agent_timestamp gauge")
        prometheus_lines.append(f"evo_agent_timestamp {time.time()}")
        
        # Population metrics
        if "population_metrics" in metrics:
            pop_metrics = metrics["population_metrics"]
            prometheus_lines.extend([
                f"# HELP evo_agent_best_fitness Best fitness score",
                f"# TYPE evo_agent_best_fitness gauge",
                f"evo_agent_best_fitness {pop_metrics.get('best_fitness', 0)}",
                f"# HELP evo_agent_avg_fitness Average fitness score",
                f"# TYPE evo_agent_avg_fitness gauge",
                f"evo_agent_avg_fitness {pop_metrics.get('avg_fitness', 0)}",
                f"# HELP evo_agent_population_size Population size",
                f"# TYPE evo_agent_population_size gauge",
                f"evo_agent_population_size {pop_metrics.get('population_size', 0)}",
                f"# HELP evo_agent_generation Current generation",
                f"# TYPE evo_agent_generation gauge",
                f"evo_agent_generation {pop_metrics.get('generation', 0)}"
            ])
        
        # LLM metrics
        if "llm_metrics" in metrics:
            llm_metrics = metrics["llm_metrics"]
            prometheus_lines.extend([
                f"# HELP evo_agent_llm_requests Total LLM requests",
                f"# TYPE evo_agent_llm_requests counter",
                f"evo_agent_llm_requests {llm_metrics.get('request_count', 0)}",
                f"# HELP evo_agent_llm_success_rate LLM success rate",
                f"# TYPE evo_agent_llm_success_rate gauge",
                f"evo_agent_llm_success_rate {llm_metrics.get('success_rate', 0)}"
            ])
        
        # Diversity metrics
        if "diversity_metrics" in metrics:
            div_metrics = metrics["diversity_metrics"]
            prometheus_lines.extend([
                f"# HELP evo_agent_diversity_score Population diversity score",
                f"# TYPE evo_agent_diversity_score gauge",
                f"evo_agent_diversity_score {div_metrics.get('diversity_score', 0)}",
                f"# HELP evo_agent_num_clusters Number of clusters",
                f"# TYPE evo_agent_num_clusters gauge",
                f"evo_agent_num_clusters {div_metrics.get('num_clusters', 0)}"
            ])
        
        # Write to file
        with open("metrics.prom", 'w') as f:
            f.write('\n'.join(prometheus_lines))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics.
        
        Returns:
            Metrics summary
        """
        metrics = self.get_aggregated_metrics()
        
        summary = {
            "runtime_hours": metrics.get("runtime_hours", 0),
            "total_generations": 0,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            "llm_requests": 0,
            "llm_success_rate": 0.0,
            "diversity_score": 0.0
        }
        
        if "population_metrics" in metrics:
            pop_metrics = metrics["population_metrics"]
            summary.update({
                "total_generations": pop_metrics.get("generation", 0),
                "best_fitness": pop_metrics.get("best_fitness", 0.0),
                "avg_fitness": pop_metrics.get("avg_fitness", 0.0)
            })
        
        if "llm_metrics" in metrics:
            llm_metrics = metrics["llm_metrics"]
            summary.update({
                "llm_requests": llm_metrics.get("request_count", 0),
                "llm_success_rate": llm_metrics.get("success_rate", 0.0)
            })
        
        if "diversity_metrics" in metrics:
            div_metrics = metrics["diversity_metrics"]
            summary["diversity_score"] = div_metrics.get("diversity_score", 0.0)
        
        return summary
    
    def save_metrics_history(self, filepath: str) -> None:
        """
        Save metrics history to file.
        
        Args:
            filepath: File path to save to
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Saved {len(self.metrics_history)} metrics records to {filepath}")
    
    def load_metrics_history(self, filepath: str) -> None:
        """
        Load metrics history from file.
        
        Args:
            filepath: File path to load from
        """
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        logger.info(f"Loaded {len(self.metrics_history)} metrics records from {filepath}")


class PerformanceMonitor:
    """
    Monitors performance of specific operations.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.operation_errors: Dict[str, int] = {}
    
    def start_operation(self, operation_name: str) -> float:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Start time
        """
        start_time = time.time()
        return start_time
    
    def end_operation(self, operation_name: str, start_time: float, success: bool = True) -> float:
        """
        End timing an operation.
        
        Args:
            operation_name: Name of the operation
            start_time: Start time from start_operation
            success: Whether operation was successful
            
        Returns:
            Duration in seconds
        """
        duration = time.time() - start_time
        
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
            self.operation_errors[operation_name] = 0
        
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        if not success:
            self.operation_errors[operation_name] += 1
        
        return duration
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation statistics
        """
        if operation_name not in self.operation_times:
            return {}
        
        times = self.operation_times[operation_name]
        count = self.operation_counts[operation_name]
        errors = self.operation_errors[operation_name]
        
        return {
            "count": count,
            "errors": errors,
            "success_rate": (count - errors) / count if count > 0 else 0.0,
            "avg_time": np.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": np.std(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operations.
        
        Returns:
            All operation statistics
        """
        return {
            name: self.get_operation_stats(name)
            for name in self.operation_times.keys()
        } 