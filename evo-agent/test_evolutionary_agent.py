#!/usr/bin/env python3
"""
Unit tests for the Evolutionary Agent system.
"""
import unittest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import components to test
from evolutionary_agent import EvolutionaryAgent, Candidate, EvolutionConfig
from llm_interface import LLMInterface, LLMConfig
from patch_manager import PatchManager, RobustLLMParser, DiffBlock
from diversity_manager import DiversityManager, DiversityConfig
from population_manager import PopulationManager, PopulationConfig
from prompt_manager import PromptManager, PromptManagerConfig
from metrics_exporter import MetricsExporter, MetricsConfig, PerformanceMonitor


class TestCandidate(unittest.TestCase):
    """Test Candidate class."""
    
    def test_candidate_creation(self):
        """Test candidate creation with all fields."""
        candidate = Candidate(
            id="test_001",
            code="def test(): return True",
            prompt="Write a test function",
            tools={"test_tool": "def test_tool(): pass"},
            memory={"context": "test context"},
            generation=1,
            parent_id="parent_001",
            mutation_type="mutation"
        )
        
        self.assertEqual(candidate.id, "test_001")
        self.assertEqual(candidate.code, "def test(): return True")
        self.assertEqual(candidate.prompt, "Write a test function")
        self.assertEqual(candidate.tools, {"test_tool": "def test_tool(): pass"})
        self.assertEqual(candidate.memory, {"context": "test context"})
        self.assertEqual(candidate.generation, 1)
        self.assertEqual(candidate.parent_id, "parent_001")
        self.assertEqual(candidate.mutation_type, "mutation")
        self.assertEqual(candidate.fitness_score, 0.0)


class TestLLMInterface(unittest.TestCase):
    """Test LLM Interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000
        )
    
    @patch('openai.AsyncOpenAI')
    def test_llm_interface_initialization(self, mock_openai):
        """Test LLM interface initialization."""
        llm = LLMInterface(self.config)
        
        self.assertEqual(llm.config.model, "gpt-3.5-turbo")
        self.assertEqual(llm.config.temperature, 0.5)
        self.assertEqual(llm.config.max_tokens, 1000)
        self.assertEqual(llm.request_count, 0)
        self.assertEqual(llm.error_count, 0)
    
    def test_llm_config_defaults(self):
        """Test LLM config defaults."""
        config = LLMConfig()
        
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 2000)
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.retry_delay, 1.0)


class TestPatchManager(unittest.TestCase):
    """Test Patch Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patch_manager = PatchManager()
        self.parser = RobustLLMParser()
    
    def test_diff_block_creation(self):
        """Test DiffBlock creation."""
        diff_block = DiffBlock(
            search_text="def old_function():",
            replace_text="def new_function():",
            start_line=10,
            end_line=15
        )
        
        self.assertEqual(diff_block.search_text, "def old_function():")
        self.assertEqual(diff_block.replace_text, "def new_function():")
        self.assertEqual(diff_block.start_line, 10)
        self.assertEqual(diff_block.end_line, 15)
    
    def test_parse_llm_diff_alpha_evolve_format(self):
        """Test parsing AlphaEvolve format diffs."""
        llm_response = """
        Here are the improvements:
        
        <<<<<<< SEARCH
        def old_function():
            return False
        =======
        def new_function():
            return True
        >>>>>>> REPLACE
        """
        
        diff_blocks = self.patch_manager.parse_llm_diff(llm_response)
        
        self.assertEqual(len(diff_blocks), 1)
        self.assertEqual(diff_blocks[0].search_text.strip(), "def old_function():\n    return False")
        self.assertEqual(diff_blocks[0].replace_text.strip(), "def new_function():\n    return True")
    
    def test_apply_patch_simple(self):
        """Test simple patch application."""
        original_code = "def test():\n    return False"
        diff_blocks = [
            DiffBlock(
                search_text="def test():",
                replace_text="def improved_test():"
            )
        ]
        
        patched_code = self.patch_manager.apply_patch(original_code, diff_blocks)
        
        self.assertIn("def improved_test():", patched_code)
        self.assertNotIn("def test():", patched_code)
    
    def test_validate_patch_valid(self):
        """Test patch validation with valid code."""
        original_code = "def test(): pass"
        patched_code = "def test():\n    return True"
        
        is_valid = self.patch_manager.validate_patch(original_code, patched_code)
        self.assertTrue(is_valid)
    
    def test_validate_patch_invalid(self):
        """Test patch validation with invalid code."""
        original_code = "def test(): pass"
        patched_code = "def test():\n    return True\n    invalid syntax"
        
        is_valid = self.patch_manager.validate_patch(original_code, patched_code)
        self.assertFalse(is_valid)
    
    def test_robust_parser_fallback(self):
        """Test robust parser fallback strategies."""
        # Test code block extraction
        response_with_code_blocks = """
        Here's the improved function:
        
        ```python
        def improved_function():
            return True
        ```
        """
        
        diff_blocks = self.parser.parse_mutation_response(response_with_code_blocks)
        self.assertGreater(len(diff_blocks), 0)
    
    def test_parse_list_response(self):
        """Test list response parsing."""
        list_response = """
        Edge cases to consider:
        1. Unicode characters
        2. Nested lists
        3. Mixed content
        """
        
        items = self.parser.parse_list_response(list_response)
        self.assertEqual(len(items), 3)
        self.assertIn("Unicode characters", items[0])


class TestDiversityManager(unittest.TestCase):
    """Test Diversity Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DiversityConfig(
            min_cluster_size=2,
            max_clusters=5,
            diversity_weight=0.2
        )
        self.diversity_manager = DiversityManager(self.config)
    
    def test_diversity_config(self):
        """Test diversity configuration."""
        self.assertEqual(self.config.min_cluster_size, 2)
        self.assertEqual(self.config.max_clusters, 5)
        self.assertEqual(self.config.diversity_weight, 0.2)
        self.assertEqual(self.config.similarity_threshold, 0.8)
    
    def test_cluster_candidates_small_population(self):
        """Test clustering with small population."""
        # Create mock candidates
        candidates = []
        for i in range(3):
            candidate = Mock()
            candidate.code = f"def approach_{i}(): pass"
            candidate.prompt = f"prompt_{i}"
            candidate.fitness_score = 0.5 + (i * 0.1)
            candidates.append(candidate)
        
        clusters = self.diversity_manager.cluster_candidates(candidates)
        
        # With small population, should return single cluster
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 3)
    
    def test_select_diverse_candidates(self):
        """Test diverse candidate selection."""
        # Create mock candidates
        candidates = []
        for i in range(10):
            candidate = Mock()
            candidate.code = f"def approach_{i}(): pass"
            candidate.prompt = f"prompt_{i}"
            candidate.fitness_score = 0.5 + (i * 0.1)
            candidates.append(candidate)
        
        selected = self.diversity_manager.select_diverse_candidates(candidates, 5)
        
        self.assertEqual(len(selected), 5)
        # Should have different fitness scores
        fitness_scores = [c.fitness_score for c in selected]
        self.assertGreater(len(set(fitness_scores)), 1)
    
    def test_calculate_novelty_score(self):
        """Test novelty score calculation."""
        # Create mock candidates
        candidate = Mock()
        candidate.code = "def unique_approach(): pass"
        candidate.prompt = "unique prompt"
        
        population = []
        for i in range(3):
            pop_candidate = Mock()
            pop_candidate.code = f"def approach_{i}(): pass"
            pop_candidate.prompt = f"prompt_{i}"
            population.append(pop_candidate)
        
        novelty_score = self.diversity_manager.calculate_novelty_score(candidate, population)
        
        self.assertGreaterEqual(novelty_score, 0.0)
        self.assertLessEqual(novelty_score, 1.0)


class TestPopulationManager(unittest.TestCase):
    """Test Population Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PopulationConfig(
            population_size=10,
            elite_size=2,
            tournament_size=3
        )
        self.population_manager = PopulationManager(self.config)
    
    def test_population_config(self):
        """Test population configuration."""
        self.assertEqual(self.config.population_size, 10)
        self.assertEqual(self.config.elite_size, 2)
        self.assertEqual(self.config.tournament_size, 3)
        self.assertEqual(self.config.mutation_rate, 0.3)
        self.assertEqual(self.config.crossover_rate, 0.7)
    
    def test_initialize_population(self):
        """Test population initialization."""
        baseline = Candidate(
            id="baseline",
            code="def baseline(): pass",
            prompt="baseline prompt",
            tools={},
            memory={},
            generation=0
        )
        
        self.population_manager.initialize_population(baseline)
        
        self.assertEqual(len(self.population_manager.population), 1)
        self.assertEqual(self.population_manager.best_candidate, baseline)
        self.assertEqual(self.population_manager.generation, 0)
    
    def test_select_parents_tournament(self):
        """Test tournament selection."""
        # Add candidates to population
        for i in range(5):
            candidate = Candidate(
                id=f"candidate_{i}",
                code=f"def candidate_{i}(): pass",
                prompt=f"prompt_{i}",
                tools={},
                memory={},
                generation=0,
                fitness_score=0.5 + (i * 0.1)
            )
            self.population_manager.population.append(candidate)
        
        parents = self.population_manager.select_parents(3)
        
        self.assertEqual(len(parents), 3)
        # Should select different candidates
        parent_ids = [p.id for p in parents]
        self.assertGreater(len(set(parent_ids)), 1)
    
    def test_crossover_candidates(self):
        """Test candidate crossover."""
        parent1 = Candidate(
            id="parent1",
            code="def parent1(): return 1",
            prompt="parent1 prompt",
            tools={"tool1": "def tool1(): pass"},
            memory={"mem1": "value1"},
            generation=0
        )
        
        parent2 = Candidate(
            id="parent2",
            code="def parent2(): return 2",
            prompt="parent2 prompt",
            tools={"tool2": "def tool2(): pass"},
            memory={"mem2": "value2"},
            generation=0
        )
        
        child = self.population_manager.crossover_candidates(parent1, parent2, "child")
        
        self.assertEqual(child.id, "child")
        self.assertEqual(child.generation, 1)
        self.assertEqual(child.parent_id, "parent1+parent2")
        self.assertEqual(child.mutation_type, "crossover")
        
        # Child should have elements from both parents
        self.assertIn("parent1", child.code)
        self.assertIn("parent2", child.code)


class TestPromptManager(unittest.TestCase):
    """Test Prompt Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PromptManagerConfig(
            max_templates_per_category=5,
            performance_threshold=0.1,
            evolution_rate=0.3
        )
        self.prompt_manager = PromptManager(self.config)
    
    def test_prompt_manager_config(self):
        """Test prompt manager configuration."""
        self.assertEqual(self.config.max_templates_per_category, 5)
        self.assertEqual(self.config.performance_threshold, 0.1)
        self.assertEqual(self.config.evolution_rate, 0.3)
        self.assertEqual(self.config.template_cache_size, 100)
    
    def test_add_template(self):
        """Test adding a template."""
        template = "You are an expert programmer."
        category = "mutation"
        
        template_id = self.prompt_manager.add_template(template, category)
        
        self.assertIsNotNone(template_id)
        self.assertIn(template_id, self.prompt_manager.templates)
        self.assertIn(template_id, self.prompt_manager.categories[category])
    
    def test_get_templates_by_category(self):
        """Test getting templates by category."""
        # Add templates
        template1 = "Template 1"
        template2 = "Template 2"
        
        self.prompt_manager.add_template(template1, "mutation")
        self.prompt_manager.add_template(template2, "mutation")
        
        templates = self.prompt_manager.get_templates_by_category("mutation")
        
        self.assertEqual(len(templates), 2)
        template_texts = [t.template for t in templates]
        self.assertIn(template1, template_texts)
        self.assertIn(template2, template_texts)
    
    def test_update_template_performance(self):
        """Test updating template performance."""
        template_id = self.prompt_manager.add_template("Test template", "mutation")
        
        # Update performance
        self.prompt_manager.update_template_performance(template_id, True, 0.8)
        
        template = self.prompt_manager.templates[template_id]
        self.assertEqual(template.usage_count, 1)
        self.assertEqual(template.success_rate, 1.0)
        self.assertEqual(template.avg_fitness, 0.8)
        self.assertEqual(template.performance_score, 0.8)
    
    def test_evolve_templates(self):
        """Test template evolution."""
        # Add multiple templates
        for i in range(3):
            self.prompt_manager.add_template(f"Template {i}", "mutation")
        
        # Update performance to create diversity
        for template_id in list(self.prompt_manager.templates.keys()):
            self.prompt_manager.update_template_performance(template_id, True, 0.5 + (i * 0.1))
        
        # Evolve templates
        new_template_ids = self.prompt_manager.evolve_templates("mutation")
        
        self.assertGreater(len(new_template_ids), 0)
        self.assertLessEqual(len(self.prompt_manager.get_templates_by_category("mutation")), 5)


class TestMetricsExporter(unittest.TestCase):
    """Test Metrics Exporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MetricsConfig(
            export_interval=30,
            json_enabled=True,
            log_enabled=True
        )
        self.metrics_exporter = MetricsExporter(self.config)
    
    def test_metrics_config(self):
        """Test metrics configuration."""
        self.assertEqual(self.config.export_interval, 30)
        self.assertTrue(self.config.json_enabled)
        self.assertTrue(self.config.log_enabled)
        self.assertFalse(self.config.prometheus_enabled)
    
    def test_update_metrics(self):
        """Test updating metrics."""
        llm_metrics = {
            "request_count": 10,
            "success_rate": 0.9
        }
        
        self.metrics_exporter.update_llm_metrics(llm_metrics)
        
        self.assertEqual(self.metrics_exporter.llm_metrics["request_count"], 10)
        self.assertEqual(self.metrics_exporter.llm_metrics["success_rate"], 0.9)
        self.assertIn("timestamp", self.metrics_exporter.llm_metrics)
    
    def test_get_aggregated_metrics(self):
        """Test getting aggregated metrics."""
        # Update some metrics
        self.metrics_exporter.update_llm_metrics({"request_count": 5})
        self.metrics_exporter.update_population_metrics({"generation": 10})
        
        aggregated = self.metrics_exporter.get_aggregated_metrics()
        
        self.assertIn("timestamp", aggregated)
        self.assertIn("runtime_seconds", aggregated)
        self.assertIn("llm_metrics", aggregated)
        self.assertIn("population_metrics", aggregated)
    
    def test_performance_monitor(self):
        """Test performance monitor."""
        monitor = PerformanceMonitor()
        
        # Monitor an operation
        start_time = monitor.start_operation("test_operation")
        duration = monitor.end_operation("test_operation", start_time, True)
        
        stats = monitor.get_operation_stats("test_operation")
        
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["success_rate"], 1.0)
        self.assertGreater(stats["avg_time"], 0)


class TestEvolutionaryAgentIntegration(unittest.TestCase):
    """Integration tests for Evolutionary Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evolution_config = EvolutionConfig(
            population_size=5,
            generations=3,
            mutation_rate=0.3,
            elite_size=1
        )
    
    @patch('evolutionary_agent.AgentManager')
    def test_evolutionary_agent_initialization(self, mock_agent_manager):
        """Test evolutionary agent initialization."""
        # Mock the agent manager
        mock_manager = Mock()
        mock_agent = Mock()
        mock_manager.create_agent.return_value = mock_agent
        mock_agent_manager.return_value = mock_manager
        
        agent = EvolutionaryAgent("test_agent", "config.yaml", self.evolution_config)
        
        self.assertEqual(agent.generation, 0)
        self.assertEqual(len(agent.candidate_pool), 0)
        self.assertIsNone(agent.best_candidate)
        self.assertIsNone(agent.evaluation_function)
    
    def test_candidate_evolution_workflow(self):
        """Test basic candidate evolution workflow."""
        # Create baseline candidate
        baseline = Candidate(
            id="baseline",
            code="def baseline(): return 0",
            prompt="baseline prompt",
            tools={},
            memory={},
            generation=0
        )
        
        # Create population manager
        population_manager = PopulationManager()
        population_manager.initialize_population(baseline)
        
        # Add some candidates
        for i in range(3):
            candidate = Candidate(
                id=f"candidate_{i}",
                code=f"def candidate_{i}(): return {i}",
                prompt=f"prompt_{i}",
                tools={},
                memory={},
                generation=0,
                fitness_score=0.5 + (i * 0.1)
            )
            population_manager.add_candidates([candidate])
        
        # Test selection
        parents = population_manager.select_parents(2)
        self.assertEqual(len(parents), 2)
        
        # Test crossover
        if len(parents) >= 2:
            child = population_manager.crossover_candidates(parents[0], parents[1], "child")
            self.assertEqual(child.generation, 1)
            self.assertEqual(child.mutation_type, "crossover")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 