#!/usr/bin/env python3
"""
Diff Generator - Wraps RobustLLMParser and PatchManager for code mutations.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from patch_manager import RobustLLMParser, PatchManager, DiffBlock
from llm_interface import LLMInterface, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class DiffGeneratorConfig:
    """Configuration for diff generator."""
    max_diff_blocks: int = 5
    min_diff_size: int = 10
    max_diff_size: int = 1000
    validation_enabled: bool = True
    rollback_on_failure: bool = True


class DiffGenerator:
    """
    Generates and applies code diffs using LLM and robust parsing.
    """
    
    def __init__(
        self, 
        llm_config: Optional[LLMConfig] = None,
        diff_config: Optional[DiffGeneratorConfig] = None
    ):
        """
        Initialize diff generator.
        
        Args:
            llm_config: LLM configuration
            diff_config: Diff generation configuration
        """
        self.llm_interface = LLMInterface(llm_config)
        self.parser = RobustLLMParser()
        self.patch_manager = PatchManager()
        self.config = diff_config or DiffGeneratorConfig()
        
        # Metrics
        self.generation_count = 0
        self.successful_applications = 0
        self.failed_applications = 0
        self.rollback_count = 0
    
    async def generate_mutation(
        self, 
        parent_code: str, 
        task_description: str,
        previous_solutions: Optional[List[str]] = None
    ) -> List[DiffBlock]:
        """
        Generate mutation diffs for a parent candidate.
        
        Args:
            parent_code: Parent code to mutate
            task_description: Task description
            previous_solutions: Previous successful solutions
            
        Returns:
            List of diff blocks
        """
        try:
            # Generate mutation using LLM
            llm_response = await self.llm_interface.generate_mutation(
                parent_code, task_description, previous_solutions
            )
            
            # Parse response into diff blocks
            diff_blocks = self.parser.parse_mutation_response(llm_response)
            
            # Validate and filter diff blocks
            valid_blocks = self._validate_diff_blocks(diff_blocks)
            
            self.generation_count += 1
            logger.info(f"Generated {len(valid_blocks)} valid diff blocks")
            
            return valid_blocks
            
        except Exception as e:
            logger.error(f"Failed to generate mutation: {e}")
            return []
    
    def apply_mutation(
        self, 
        original_code: str, 
        diff_blocks: List[DiffBlock]
    ) -> Tuple[str, bool]:
        """
        Apply mutation diffs to code.
        
        Args:
            original_code: Original code
            diff_blocks: Diff blocks to apply
            
        Returns:
            Tuple of (patched_code, success)
        """
        if not diff_blocks:
            return original_code, False
        
        try:
            # Apply patches
            patched_code = self.patch_manager.apply_patch(original_code, diff_blocks)
            
            # Validate result
            if self.config.validation_enabled:
                is_valid = self.patch_manager.validate_patch(original_code, patched_code)
                if not is_valid:
                    logger.warning("Patch validation failed")
                    if self.config.rollback_on_failure:
                        return original_code, False
            
            self.successful_applications += 1
            logger.info(f"Successfully applied {len(diff_blocks)} diff blocks")
            
            return patched_code, True
            
        except Exception as e:
            self.failed_applications += 1
            logger.error(f"Failed to apply mutation: {e}")
            
            if self.config.rollback_on_failure:
                self.rollback_count += 1
                return original_code, False
            
            raise
    
    def _validate_diff_blocks(self, diff_blocks: List[DiffBlock]) -> List[DiffBlock]:
        """
        Validate and filter diff blocks.
        
        Args:
            diff_blocks: Raw diff blocks
            
        Returns:
            Valid diff blocks
        """
        valid_blocks = []
        
        for block in diff_blocks:
            # Check size constraints
            if len(block.search_text) < self.config.min_diff_size:
                logger.debug(f"Skipping diff block - too small: {len(block.search_text)} chars")
                continue
            
            if len(block.search_text) > self.config.max_diff_size:
                logger.debug(f"Skipping diff block - too large: {len(block.search_text)} chars")
                continue
            
            # Check for empty blocks
            if not block.search_text.strip() and not block.replace_text.strip():
                logger.debug("Skipping empty diff block")
                continue
            
            valid_blocks.append(block)
        
        # Limit number of blocks
        if len(valid_blocks) > self.config.max_diff_blocks:
            valid_blocks = valid_blocks[:self.config.max_diff_blocks]
            logger.info(f"Limited to {self.config.max_diff_blocks} diff blocks")
        
        return valid_blocks
    
    def create_diff_summary(
        self, 
        original_code: str, 
        patched_code: str
    ) -> Dict[str, Any]:
        """
        Create summary of applied diffs.
        
        Args:
            original_code: Original code
            patched_code: Patched code
            
        Returns:
            Diff summary
        """
        summary = self.patch_manager.create_diff_summary(original_code, patched_code)
        
        # Add generator metrics
        summary.update({
            "generation_count": self.generation_count,
            "successful_applications": self.successful_applications,
            "failed_applications": self.failed_applications,
            "rollback_count": self.rollback_count,
            "success_rate": self.successful_applications / max(self.generation_count, 1)
        })
        
        return summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get diff generator metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "generation_count": self.generation_count,
            "successful_applications": self.successful_applications,
            "failed_applications": self.failed_applications,
            "rollback_count": self.rollback_count,
            "success_rate": self.successful_applications / max(self.generation_count, 1),
            "llm_metrics": self.llm_interface.get_metrics()
        }


class BatchDiffGenerator:
    """
    Generates and applies multiple mutations in batch.
    """
    
    def __init__(self, diff_generator: DiffGenerator):
        """
        Initialize batch diff generator.
        
        Args:
            diff_generator: Base diff generator
        """
        self.diff_generator = diff_generator
        self.batch_count = 0
        self.batch_success_count = 0
    
    async def generate_batch_mutations(
        self, 
        parent_codes: List[str], 
        task_description: str,
        previous_solutions: Optional[List[str]] = None
    ) -> List[List[DiffBlock]]:
        """
        Generate mutations for multiple parent candidates.
        
        Args:
            parent_codes: List of parent codes
            task_description: Task description
            previous_solutions: Previous successful solutions
            
        Returns:
            List of diff block lists for each parent
        """
        batch_mutations = []
        
        for i, parent_code in enumerate(parent_codes):
            try:
                diff_blocks = await self.diff_generator.generate_mutation(
                    parent_code, task_description, previous_solutions
                )
                batch_mutations.append(diff_blocks)
                logger.info(f"Generated mutations for parent {i+1}/{len(parent_codes)}")
                
            except Exception as e:
                logger.error(f"Failed to generate mutations for parent {i+1}: {e}")
                batch_mutations.append([])
        
        self.batch_count += 1
        return batch_mutations
    
    def apply_batch_mutations(
        self, 
        original_codes: List[str], 
        batch_mutations: List[List[DiffBlock]]
    ) -> List[Tuple[str, bool]]:
        """
        Apply batch mutations to multiple codes.
        
        Args:
            original_codes: List of original codes
            batch_mutations: List of diff block lists
            
        Returns:
            List of (patched_code, success) tuples
        """
        results = []
        
        for i, (original_code, mutations) in enumerate(zip(original_codes, batch_mutations)):
            try:
                patched_code, success = self.diff_generator.apply_mutation(original_code, mutations)
                results.append((patched_code, success))
                
                if success:
                    self.batch_success_count += 1
                
                logger.info(f"Applied mutations for code {i+1}/{len(original_codes)}")
                
            except Exception as e:
                logger.error(f"Failed to apply mutations for code {i+1}: {e}")
                results.append((original_code, False))
        
        return results
    
    def get_batch_metrics(self) -> Dict[str, Any]:
        """
        Get batch diff generator metrics.
        
        Returns:
            Batch metrics dictionary
        """
        return {
            "batch_count": self.batch_count,
            "batch_success_count": self.batch_success_count,
            "batch_success_rate": self.batch_success_count / max(self.batch_count, 1),
            "diff_generator_metrics": self.diff_generator.get_metrics()
        } 