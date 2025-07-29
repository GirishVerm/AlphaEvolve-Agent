#!/usr/bin/env python3
"""
Experiment Tracking with Reproducibility and ML-ops Integration.
"""
import logging
import json
import hashlib
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiment_name: str
    experiment_id: Optional[str] = None
    seed: Optional[int] = None
    mlflow_enabled: bool = False
    wandb_enabled: bool = False
    local_tracking: bool = True
    artifact_dir: str = "experiments"
    checkpoint_interval: int = 10  # generations


class ExperimentTracker:
    """
    Tracks experiments with full reproducibility and ML-ops integration.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment tracker.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_id = config.experiment_id or self._generate_experiment_id()
        self.start_time = datetime.now()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize tracking backends
        self.mlflow_run = None
        self.wandb_run = None
        self._init_tracking()
        
        # Experiment state
        self.metrics_history: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        
        # Create experiment directory
        self.experiment_dir = Path(self.config.artifact_dir) / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial configuration
        self._save_config()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"{self.config.experiment_name}_{timestamp}_{random_suffix}"
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            os.environ['PYTHONHASHSEED'] = str(self.config.seed)
            
            # Set seeds for other libraries
            try:
                import torch
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.config.seed)
            except ImportError:
                pass
            
            logger.info(f"Set random seeds to {self.config.seed}")
    
    def _init_tracking(self) -> None:
        """Initialize ML-ops tracking backends."""
        if self.config.mlflow_enabled:
            try:
                import mlflow
                mlflow.set_experiment(self.config.experiment_name)
                self.mlflow_run = mlflow.start_run(run_name=self.experiment_id)
                logger.info(f"Started MLflow run: {self.experiment_id}")
            except ImportError:
                logger.warning("MLflow not available, skipping MLflow tracking")
        
        if self.config.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.experiment_name,
                    name=self.experiment_id,
                    config=asdict(self.config)
                )
                logger.info(f"Started Weights & Biases run: {self.experiment_id}")
            except ImportError:
                logger.warning("Weights & Biases not available, skipping W&B tracking")
    
    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to all tracking backends.
        
        Args:
            metrics: Metrics to log
            step: Step number (generation)
        """
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # Store locally
        self.metrics_history.append(metrics)
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")
        
        # Log to Weights & Biases
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log to W&B: {e}")
        
        # Save to local file
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """
        Log artifact to tracking backends.
        
        Args:
            artifact_path: Path to artifact file
            artifact_name: Name for the artifact
        """
        if not os.path.exists(artifact_path):
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        artifact_name = artifact_name or os.path.basename(artifact_path)
        
        # Copy to experiment directory
        dest_path = self.experiment_dir / artifact_name
        import shutil
        shutil.copy2(artifact_path, dest_path)
        self.artifacts.append(str(dest_path))
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                logger.error(f"Failed to log artifact to MLflow: {e}")
        
        # Log to Weights & Biases
        if self.wandb_run:
            try:
                import wandb
                wandb.save(artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact to W&B: {e}")
    
    def create_checkpoint(self, state: Dict[str, Any], generation: int) -> None:
        """
        Create experiment checkpoint.
        
        Args:
            state: Current experiment state
            generation: Current generation
        """
        checkpoint = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        
        self.checkpoints.append(checkpoint)
        
        # Save checkpoint file
        checkpoint_file = self.experiment_dir / f"checkpoint_gen_{generation}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Log to tracking backends
        self.log_artifact(str(checkpoint_file), f"checkpoint_gen_{generation}")
        
        logger.info(f"Created checkpoint for generation {generation}")
    
    def load_checkpoint(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Load experiment checkpoint.
        
        Args:
            generation: Generation to load
            
        Returns:
            Checkpoint state or None
        """
        checkpoint_file = self.experiment_dir / f"checkpoint_gen_{generation}.pkl"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_file}")
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded checkpoint for generation {generation}")
            return checkpoint['state']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def log_pareto_front(self, pareto_front, generation: int) -> None:
        """
        Log Pareto front visualization.
        
        Args:
            pareto_front: Pareto front object
            generation: Current generation
        """
        try:
            plot_path = self.experiment_dir / f"pareto_front_gen_{generation}.png"
            pareto_front.plot_front(str(plot_path))
            self.log_artifact(str(plot_path), f"pareto_front_gen_{generation}")
        except Exception as e:
            logger.error(f"Failed to log Pareto front: {e}")
    
    def log_population_stats(self, population_stats: Dict[str, Any], generation: int) -> None:
        """
        Log population statistics.
        
        Args:
            population_stats: Population statistics
            generation: Current generation
        """
        # Add generation info
        stats = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            **population_stats
        }
        
        self.log_metrics(stats, step=generation)
    
    def log_llm_metrics(self, llm_metrics: Dict[str, Any]) -> None:
        """
        Log LLM usage metrics.
        
        Args:
            llm_metrics: LLM metrics
        """
        self.log_metrics(llm_metrics)
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to tracking backends
        self.log_metrics(error_info)
        
        # Save to error log
        error_log = self.experiment_dir / "errors.json"
        try:
            with open(error_log, 'r') as f:
                errors = json.load(f)
        except FileNotFoundError:
            errors = []
        
        errors.append(error_info)
        
        with open(error_log, 'w') as f:
            json.dump(errors, f, indent=2)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Experiment summary
        """
        runtime = datetime.now() - self.start_time
        
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.config.experiment_name,
            'start_time': self.start_time.isoformat(),
            'runtime_seconds': runtime.total_seconds(),
            'runtime_hours': runtime.total_seconds() / 3600,
            'metrics_count': len(self.metrics_history),
            'checkpoints_count': len(self.checkpoints),
            'artifacts_count': len(self.artifacts),
            'seed': self.config.seed,
            'mlflow_enabled': self.config.mlflow_enabled,
            'wandb_enabled': self.config.wandb_enabled
        }
    
    def save_experiment_state(self, state: Dict[str, Any]) -> None:
        """
        Save complete experiment state.
        
        Args:
            state: Complete experiment state
        """
        state_file = self.experiment_dir / "experiment_state.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        
        self.log_artifact(str(state_file), "experiment_state")
    
    def load_experiment_state(self) -> Optional[Dict[str, Any]]:
        """
        Load complete experiment state.
        
        Returns:
            Experiment state or None
        """
        state_file = self.experiment_dir / "experiment_state.pkl"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            return state
        except Exception as e:
            logger.error(f"Failed to load experiment state: {e}")
            return None
    
    def finish(self) -> None:
        """Finish experiment and close tracking backends."""
        # Log final summary
        summary = self.get_experiment_summary()
        self.log_metrics(summary)
        
        # Close MLflow run
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
                logger.info("Closed MLflow run")
            except Exception as e:
                logger.error(f"Failed to close MLflow run: {e}")
        
        # Close W&B run
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
                logger.info("Closed Weights & Biases run")
            except Exception as e:
                logger.error(f"Failed to close W&B run: {e}")
        
        # Save final summary
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Experiment {self.experiment_id} finished")


class ReproducibilityManager:
    """
    Manages reproducibility across experiments.
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize reproducibility manager.
        
        Args:
            base_seed: Base seed for experiment
        """
        self.base_seed = base_seed
        self.seed_history: List[int] = []
    
    def get_experiment_seed(self, experiment_name: str) -> int:
        """
        Get deterministic seed for experiment.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Deterministic seed
        """
        # Create hash from experiment name
        name_hash = hashlib.md5(experiment_name.encode()).hexdigest()
        seed = int(name_hash[:8], 16) + self.base_seed
        
        self.seed_history.append(seed)
        return seed
    
    def set_random_seeds(self, seed: int) -> None:
        """
        Set all random seeds for reproducibility.
        
        Args:
            seed: Seed to set
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set seeds for other libraries
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass
        
        logger.info(f"Set all random seeds to {seed}")
    
    def get_seed_history(self) -> List[int]:
        """
        Get history of used seeds.
        
        Returns:
            List of used seeds
        """
        return self.seed_history.copy()


class ExperimentRegistry:
    """
    Registry for managing multiple experiments.
    """
    
    def __init__(self, registry_dir: str = "experiment_registry"):
        """
        Initialize experiment registry.
        
        Args:
            registry_dir: Directory for registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "experiments.json"
        self.experiments = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save experiment registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def register_experiment(self, experiment_id: str, config: ExperimentConfig) -> None:
        """
        Register experiment in registry.
        
        Args:
            experiment_id: Experiment ID
            config: Experiment configuration
        """
        self.experiments[experiment_id] = {
            'config': asdict(config),
            'registration_time': datetime.now().isoformat(),
            'status': 'registered'
        }
        self._save_registry()
    
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """
        Update experiment status.
        
        Args:
            experiment_id: Experiment ID
            status: New status
        """
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = status
            self.experiments[experiment_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
    
    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment information.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment information or None
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[str] = None) -> List[str]:
        """
        List experiments.
        
        Args:
            status: Filter by status
            
        Returns:
            List of experiment IDs
        """
        if status is None:
            return list(self.experiments.keys())
        
        return [
            exp_id for exp_id, info in self.experiments.items()
            if info.get('status') == status
        ]
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get registry summary.
        
        Returns:
            Registry summary
        """
        status_counts = {}
        for info in self.experiments.values():
            status = info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_experiments': len(self.experiments),
            'status_counts': status_counts,
            'registry_file': str(self.registry_file)
        } 