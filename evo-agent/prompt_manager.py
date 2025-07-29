#!/usr/bin/env python3
"""
Prompt Manager - Stores and evolves prompt templates.
"""
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""
    id: str
    template: str
    category: str
    performance_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    avg_fitness: float = 0.0
    created_at: str = ""
    last_used: str = ""
    parent_id: Optional[str] = None
    mutation_type: str = "initial"


@dataclass
class PromptManagerConfig:
    """Configuration for prompt manager."""
    max_templates_per_category: int = 20
    performance_threshold: float = 0.1
    evolution_rate: float = 0.3
    template_cache_size: int = 100
    backup_enabled: bool = True
    backup_interval: int = 10  # generations


class PromptManager:
    """
    Manages prompt templates and their evolution.
    """
    
    def __init__(self, config: Optional[PromptManagerConfig] = None):
        """
        Initialize prompt manager.
        
        Args:
            config: Prompt manager configuration
        """
        self.config = config or PromptManagerConfig()
        self.templates: Dict[str, PromptTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Metrics
        self.template_count = 0
        self.evolution_count = 0
        self.backup_count = 0
    
    def add_template(
        self, 
        template: str, 
        category: str, 
        parent_id: Optional[str] = None,
        mutation_type: str = "initial"
    ) -> str:
        """
        Add a new prompt template.
        
        Args:
            template: Template string
            category: Template category
            parent_id: Parent template ID
            mutation_type: Type of mutation
            
        Returns:
            Template ID
        """
        template_id = self._generate_template_id(template, category)
        
        prompt_template = PromptTemplate(
            id=template_id,
            template=template,
            category=category,
            parent_id=parent_id,
            mutation_type=mutation_type,
            created_at=datetime.now().isoformat()
        )
        
        self.templates[template_id] = prompt_template
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(template_id)
        
        self.template_count += 1
        logger.info(f"Added template {template_id} to category {category}")
        
        return template_id
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template or None
        """
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[PromptTemplate]:
        """
        Get all templates in a category.
        
        Args:
            category: Template category
            
        Returns:
            List of templates
        """
        if category not in self.categories:
            return []
        
        return [self.templates[tid] for tid in self.categories[category]]
    
    def get_best_templates(
        self, 
        category: str, 
        top_k: int = 5
    ) -> List[PromptTemplate]:
        """
        Get the best performing templates in a category.
        
        Args:
            category: Template category
            top_k: Number of top templates to return
            
        Returns:
            Best templates
        """
        templates = self.get_templates_by_category(category)
        
        # Sort by performance score
        sorted_templates = sorted(
            templates, 
            key=lambda t: t.performance_score, 
            reverse=True
        )
        
        return sorted_templates[:top_k]
    
    def update_template_performance(
        self, 
        template_id: str, 
        success: bool, 
        fitness_score: float
    ) -> None:
        """
        Update template performance metrics.
        
        Args:
            template_id: Template ID
            success: Whether the template was successful
            fitness_score: Fitness score achieved
        """
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found")
            return
        
        template = self.templates[template_id]
        template.usage_count += 1
        template.last_used = datetime.now().isoformat()
        
        # Update success rate
        if template.usage_count == 1:
            template.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.1
            template.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * template.success_rate
        
        # Update average fitness
        if template.usage_count == 1:
            template.avg_fitness = fitness_score
        else:
            # Exponential moving average
            alpha = 0.1
            template.avg_fitness = alpha * fitness_score + (1 - alpha) * template.avg_fitness
        
        # Update performance score
        template.performance_score = template.success_rate * template.avg_fitness
        
        logger.debug(f"Updated template {template_id} performance: {template.performance_score}")
    
    def evolve_templates(self, category: str) -> List[str]:
        """
        Evolve templates in a category.
        
        Args:
            category: Template category
            
        Returns:
            List of new template IDs
        """
        templates = self.get_templates_by_category(category)
        if len(templates) < 2:
            return []
        
        new_template_ids = []
        
        # Select parents based on performance
        sorted_templates = sorted(
            templates, 
            key=lambda t: t.performance_score, 
            reverse=True
        )
        
        # Create new templates through crossover and mutation
        num_evolutions = max(1, int(len(templates) * self.config.evolution_rate))
        
        for i in range(num_evolutions):
            # Select parents using tournament selection
            parent1 = self._tournament_select(sorted_templates)
            parent2 = self._tournament_select(sorted_templates)
            
            # Crossover
            if np.random.random() < 0.7:  # 70% crossover rate
                child_template = self._crossover_templates(parent1, parent2)
            else:
                # Mutation
                child_template = self._mutate_template(parent1)
            
            # Add new template
            template_id = self.add_template(
                child_template,
                category,
                parent_id=f"{parent1.id}+{parent2.id}",
                mutation_type="evolution"
            )
            
            new_template_ids.append(template_id)
            self.evolution_count += 1
        
        # Prune low-performing templates
        self._prune_templates(category)
        
        logger.info(f"Evolved {len(new_template_ids)} new templates in category {category}")
        return new_template_ids
    
    def _tournament_select(self, templates: List[PromptTemplate]) -> PromptTemplate:
        """Select template using tournament selection."""
        tournament_size = min(3, len(templates))
        tournament = np.random.choice(templates, tournament_size, replace=False)
        return max(tournament, key=lambda t: t.performance_score)
    
    def _crossover_templates(self, template1: PromptTemplate, template2: PromptTemplate) -> str:
        """Crossover two templates."""
        words1 = template1.template.split()
        words2 = template2.template.split()
        
        max_words = max(len(words1), len(words2))
        child_words = []
        
        for i in range(max_words):
            if i < len(words1) and i < len(words2):
                # Randomly choose from either parent
                child_words.append(words1[i] if np.random.random() < 0.5 else words2[i])
            elif i < len(words1):
                child_words.append(words1[i])
            else:
                child_words.append(words2[i])
        
        return ' '.join(child_words)
    
    def _mutate_template(self, template: PromptTemplate) -> str:
        """Mutate a template."""
        words = template.template.split()
        
        # Random mutations
        if np.random.random() < 0.1:  # 10% chance to add word
            words.append("improved")
        if np.random.random() < 0.1:  # 10% chance to remove word
            if len(words) > 1:
                words.pop(np.random.randint(len(words)))
        if np.random.random() < 0.2:  # 20% chance to swap words
            if len(words) > 1:
                i, j = np.random.choice(len(words), 2, replace=False)
                words[i], words[j] = words[j], words[i]
        
        return ' '.join(words)
    
    def _prune_templates(self, category: str) -> None:
        """Remove low-performing templates."""
        if category not in self.categories:
            return
        
        templates = self.get_templates_by_category(category)
        if len(templates) <= self.config.max_templates_per_category:
            return
        
        # Sort by performance and remove worst
        sorted_templates = sorted(templates, key=lambda t: t.performance_score)
        templates_to_remove = sorted_templates[:-self.config.max_templates_per_category]
        
        for template in templates_to_remove:
            del self.templates[template.id]
            self.categories[category].remove(template.id)
        
        logger.info(f"Pruned {len(templates_to_remove)} low-performing templates from {category}")
    
    def _generate_template_id(self, template: str, category: str) -> str:
        """Generate unique template ID."""
        content = f"{template}:{category}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get template statistics.
        
        Returns:
            Template statistics
        """
        if not self.templates:
            return {}
        
        all_templates = list(self.templates.values())
        
        stats = {
            "total_templates": len(all_templates),
            "categories": list(self.categories.keys()),
            "avg_performance": np.mean([t.performance_score for t in all_templates]),
            "avg_success_rate": np.mean([t.success_rate for t in all_templates]),
            "avg_fitness": np.mean([t.avg_fitness for t in all_templates]),
            "evolution_count": self.evolution_count,
            "template_count": self.template_count
        }
        
        # Per-category stats
        category_stats = {}
        for category in self.categories:
            templates = self.get_templates_by_category(category)
            if templates:
                category_stats[category] = {
                    "count": len(templates),
                    "avg_performance": np.mean([t.performance_score for t in templates]),
                    "best_performance": max([t.performance_score for t in templates])
                }
        
        stats["category_stats"] = category_stats
        return stats
    
    def save_templates(self, filepath: str) -> None:
        """
        Save templates to file.
        
        Args:
            filepath: File path to save to
        """
        data = {
            "templates": {tid: asdict(template) for tid, template in self.templates.items()},
            "categories": self.categories,
            "performance_history": self.performance_history,
            "config": asdict(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.templates)} templates to {filepath}")
    
    def load_templates(self, filepath: str) -> None:
        """
        Load templates from file.
        
        Args:
            filepath: File path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load templates
        self.templates = {}
        for tid, template_data in data["templates"].items():
            template = PromptTemplate(**template_data)
            self.templates[tid] = template
        
        # Load categories
        self.categories = data["categories"]
        
        # Load performance history
        self.performance_history = data.get("performance_history", [])
        
        logger.info(f"Loaded {len(self.templates)} templates from {filepath}")
    
    def backup_templates(self, backup_dir: str) -> None:
        """
        Create backup of templates.
        
        Args:
            backup_dir: Backup directory
        """
        if not self.config.backup_enabled:
            return
        
        import os
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"templates_backup_{timestamp}.json")
        
        self.save_templates(backup_file)
        self.backup_count += 1
        
        logger.info(f"Created backup: {backup_file}")


class PromptTemplateRegistry:
    """
    Registry of predefined prompt templates.
    """
    
    @staticmethod
    def get_mutation_template() -> str:
        """Get template for code mutation."""
        return """You are an expert programmer tasked with improving code through targeted mutations.

Your goal is to generate specific, surgical improvements to the provided code that will enhance its performance, correctness, or robustness.

IMPORTANT: Use the AlphaEvolve diff format:
<<<<<<< SEARCH
# Original code to find and replace
=======
# New code to replace the original
>>>>>>> REPLACE

Focus on:
1. Performance optimizations
2. Bug fixes and edge case handling
3. Code clarity and maintainability
4. Algorithm improvements
5. Error handling enhancements

Provide only the diff blocks, no additional explanation."""
    
    @staticmethod
    def get_edge_case_template() -> str:
        """Get template for edge case analysis."""
        return """You are an expert software engineer specializing in edge case analysis.

Your task is to identify potential edge cases, ambiguities, and failure modes for the given task.

Consider:
1. Input validation edge cases
2. Performance edge cases
3. Resource constraints
4. Error conditions
5. Boundary conditions
6. Integration edge cases

Provide a clear, numbered list of edge cases."""
    
    @staticmethod
    def get_tool_analysis_template() -> str:
        """Get template for tool analysis."""
        return """You are an expert software architect specializing in tool and API design.

Analyze the current tools and edge cases to recommend improvements or new tools that would enhance the solution.

Consider:
1. Missing functionality
2. Performance bottlenecks
3. Error handling gaps
4. Integration needs
5. Scalability requirements

Provide specific recommendations with clear justifications."""
    
    @staticmethod
    def get_spec_analysis_template() -> str:
        """Get template for specification analysis."""
        return """You are an expert in software testing and evaluation specification.

Analyze the current evaluation specification and edge cases to recommend improvements that would create a more comprehensive and robust evaluation suite.

Consider:
1. Missing test cases
2. Performance benchmarks
3. Stress testing scenarios
4. Error condition testing
5. Edge case validation
6. Integration testing

Provide specific recommendations for test cases, metrics, and evaluation criteria."""
    
    @staticmethod
    def get_eval_suite_template() -> str:
        """Get template for evaluation suite generation."""
        return """You are an expert in software testing and evaluation.

Create a comprehensive "hard" evaluation suite with deliberate edge cases and challenging scenarios that will thoroughly test the robustness of the solution.

Include:
1. Edge case test scenarios
2. Performance stress tests
3. Error condition tests
4. Boundary value tests
5. Integration tests
6. Scalability tests

Provide specific test cases with expected inputs and outputs.""" 