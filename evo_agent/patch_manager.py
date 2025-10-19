#!/usr/bin/env python3
"""
Patch Manager for Evolutionary Agent - Proper diff/patch application system.
"""
import difflib
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DiffBlock:
    """Represents a single diff block."""
    search_text: str
    replace_text: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class PatchManager:
    """
    Manages application of LLM-generated diffs to code.
    Based on AlphaEvolve's SEARCH/REPLACE format.
    """
    
    def __init__(self):
        """Initialize the patch manager."""
        self.diff_pattern = re.compile(
            r'<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE',
            re.DOTALL
        )
    
    def parse_llm_diff(self, llm_response: str) -> List[DiffBlock]:
        """
        Parse LLM response for diff blocks in AlphaEvolve format.
        
        Args:
            llm_response: LLM response containing diff blocks
            
        Returns:
            List of parsed diff blocks
        """
        diff_blocks = []
        
        # Find all diff blocks in the response
        matches = self.diff_pattern.findall(llm_response)
        
        for search_text, replace_text in matches:
            # Clean up whitespace
            search_text = search_text.strip()
            replace_text = replace_text.strip()
            
            if search_text and replace_text:
                diff_block = DiffBlock(
                    search_text=search_text,
                    replace_text=replace_text
                )
                diff_blocks.append(diff_block)
        
        return diff_blocks
    
    def apply_patch(self, original_code: str, diff_blocks: List[DiffBlock]) -> str:
        """
        Apply diff blocks to original code.
        
        Args:
            original_code: Original code to patch
            diff_blocks: List of diff blocks to apply
            
        Returns:
            Patched code
        """
        if not diff_blocks:
            return original_code
        
        patched_code = original_code
        
        for diff_block in diff_blocks:
            try:
                patched_code = self._apply_single_diff(patched_code, diff_block)
            except Exception as e:
                logger.warning(f"Failed to apply diff block: {e}")
                continue
        
        return patched_code
    
    def _apply_single_diff(self, code: str, diff_block: DiffBlock) -> str:
        """
        Apply a single diff block to code.
        
        Args:
            code: Code to patch
            diff_block: Diff block to apply
            
        Returns:
            Patched code
        """
        # Find the search text in the code
        search_lines = diff_block.search_text.split('\n')
        
        # Use difflib to find the best match
        code_lines = code.split('\n')
        matcher = difflib.SequenceMatcher(None, code_lines, search_lines)
        
        # Find the best matching block
        best_match = None
        best_ratio = 0
        
        for block in matcher.get_matching_blocks():
            if block.size > 0:
                # Calculate similarity ratio for this block
                ratio = block.size / len(search_lines)
                if ratio > best_ratio and ratio > 0.8:  # Require 80% similarity
                    best_ratio = ratio
                    best_match = block
        
        if best_match is None:
            # If no good match found, try simple string replacement
            if diff_block.search_text in code:
                return code.replace(diff_block.search_text, diff_block.replace_text)
            else:
                # Add the replacement at the end if no match found
                return code + '\n' + diff_block.replace_text
        
        # Apply the replacement
        start_line = best_match.a
        end_line = best_match.a + best_match.size
        
        # Replace the matched section
        new_lines = code_lines[:start_line]
        new_lines.extend(diff_block.replace_text.split('\n'))
        new_lines.extend(code_lines[end_line:])
        
        return '\n'.join(new_lines)
    
    def validate_patch(self, original_code: str, patched_code: str) -> bool:
        """
        Validate that a patch produces valid Python code.
        
        Args:
            original_code: Original code
            patched_code: Patched code
            
        Returns:
            True if patch produces valid Python code
        """
        try:
            compile(patched_code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def rollback_patch(self, original_code: str, diff_blocks: List[DiffBlock]) -> str:
        """
        Rollback applied patches to restore original code.
        
        Args:
            original_code: Original code
            diff_blocks: Diff blocks that were applied
            
        Returns:
            Original code (rollback is just returning original)
        """
        return original_code
    
    def create_diff_summary(self, original_code: str, patched_code: str) -> Dict[str, Any]:
        """
        Create a summary of changes made by the patch.
        
        Args:
            original_code: Original code
            patched_code: Patched code
            
        Returns:
            Summary of changes
        """
        original_lines = original_code.split('\n')
        patched_lines = patched_code.split('\n')
        
        # Use difflib to generate a unified diff
        diff = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile='original',
            tofile='patched',
            lineterm=''
        )
        
        diff_text = '\n'.join(diff)
        
        # Count changes
        added_lines = sum(1 for line in patched_lines if line not in original_lines)
        removed_lines = sum(1 for line in original_lines if line not in patched_lines)
        
        return {
            'diff_text': diff_text,
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'total_changes': added_lines + removed_lines,
            'original_length': len(original_lines),
            'patched_length': len(patched_lines)
        }


class RobustLLMParser:
    """
    Robust parser for LLM responses with fallback strategies.
    """
    
    def __init__(self):
        """Initialize the parser."""
        self.patch_manager = PatchManager()
    
    def parse_mutation_response(self, llm_response: str) -> List[DiffBlock]:
        """
        Parse LLM response for mutations with multiple strategies.
        
        Args:
            llm_response: LLM response
            
        Returns:
            List of diff blocks
        """
        # Strategy 1: Try AlphaEvolve format
        diff_blocks = self.patch_manager.parse_llm_diff(llm_response)
        
        if diff_blocks:
            return diff_blocks
        
        # Strategy 2: Try to extract code blocks
        code_blocks = self._extract_code_blocks(llm_response)
        
        if code_blocks:
            # Convert code blocks to diff format
            return self._code_blocks_to_diffs(code_blocks)
        
        # Strategy 3: Try to find function definitions
        functions = self._extract_functions(llm_response)
        
        if functions:
            return self._functions_to_diffs(functions)
        
        # Strategy 4: Fallback - treat entire response as replacement
        return [DiffBlock(
            search_text="",
            replace_text=llm_response.strip()
        )]
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """Extract code blocks from response."""
        # Look for code blocks marked with ```
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        return [match.strip() for match in matches]
    
    def _extract_functions(self, response: str) -> List[str]:
        """Extract function definitions from response."""
        # Look for function definitions
        function_pattern = r'def\s+\w+\s*\([^)]*\)\s*:.*?(?=\n\S|\Z)'
        matches = re.findall(function_pattern, response, re.DOTALL)
        return matches
    
    def _code_blocks_to_diffs(self, code_blocks: List[str]) -> List[DiffBlock]:
        """Convert code blocks to diff format."""
        diffs = []
        for code_block in code_blocks:
            diffs.append(DiffBlock(
                search_text="",
                replace_text=code_block
            ))
        return diffs
    
    def _functions_to_diffs(self, functions: List[str]) -> List[DiffBlock]:
        """Convert function definitions to diff format."""
        diffs = []
        for function in functions:
            diffs.append(DiffBlock(
                search_text="",
                replace_text=function
            ))
        return diffs
    
    def parse_list_response(self, response: str) -> List[str]:
        """
        Parse a list response from the agent with robust parsing.
        
        Args:
            response: Agent response
            
        Returns:
            List of items
        """
        # Strategy 1: Look for numbered or bulleted lists
        list_patterns = [
            r'^\s*[-*]\s+(.+)$',  # Bullet points
            r'^\s*\d+\.\s+(.+)$',  # Numbered lists
            r'^\s*•\s+(.+)$',  # Bullet points (unicode)
        ]
        
        lines = response.split('\n')
        items = []
        
        for line in lines:
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    items.append(match.group(1).strip())
                    break
        
        if items:
            return items
        
        # Strategy 2: Look for lines that start with common prefixes
        common_prefixes = ['-', '*', '•', '1.', '2.', '3.', '4.', '5.']
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove common prefixes
                for prefix in common_prefixes:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                if line and len(line) > 3:  # Minimum length
                    items.append(line)
        
        return items
    
    def parse_tool_recommendations(self, response: str) -> Dict[str, Any]:
        """
        Parse tool recommendations from agent response.
        
        Args:
            response: Agent response
            
        Returns:
            Structured tool recommendations
        """
        # Extract tool names and descriptions
        tool_pattern = r'(?:tool|function|method)\s*[:\-]\s*(\w+)\s*[:\-]\s*(.+)'
        tools = re.findall(tool_pattern, response, re.IGNORECASE)
        
        if tools:
            return {
                'tools': [{'name': name, 'description': desc.strip()} for name, desc in tools],
                'raw_response': response
            }
        
        # Fallback: return structured response
        return {
            'recommendations': response,
            'raw_response': response
        }
    
    def parse_spec_recommendations(self, response: str) -> Dict[str, Any]:
        """
        Parse spec recommendations from agent response.
        
        Args:
            response: Agent response
            
        Returns:
            Structured spec recommendations
        """
        # Look for test cases, metrics, criteria
        test_pattern = r'test\s+case[s]?\s*[:\-]\s*(.+)'
        metric_pattern = r'metric[s]?\s*[:\-]\s*(.+)'
        criteria_pattern = r'criteria\s*[:\-]\s*(.+)'
        
        tests = re.findall(test_pattern, response, re.IGNORECASE)
        metrics = re.findall(metric_pattern, response, re.IGNORECASE)
        criteria = re.findall(criteria_pattern, response, re.IGNORECASE)
        
        return {
            'test_cases': tests,
            'metrics': metrics,
            'criteria': criteria,
            'raw_response': response
        }
    
    def parse_eval_suite(self, response: str) -> Dict[str, Any]:
        """
        Parse evaluation suite from agent response.
        
        Args:
            response: Agent response
            
        Returns:
            Structured evaluation suite
        """
        # Extract test cases, edge cases, performance tests
        edge_case_pattern = r'edge\s+case[s]?\s*[:\-]\s*(.+)'
        performance_pattern = r'performance\s+test[s]?\s*[:\-]\s*(.+)'
        stress_pattern = r'stress\s+test[s]?\s*[:\-]\s*(.+)'
        
        edge_cases = re.findall(edge_case_pattern, response, re.IGNORECASE)
        performance_tests = re.findall(performance_pattern, response, re.IGNORECASE)
        stress_tests = re.findall(stress_pattern, response, re.IGNORECASE)
        
        return {
            'edge_cases': edge_cases,
            'performance_tests': performance_tests,
            'stress_tests': stress_tests,
            'raw_response': response
        } 