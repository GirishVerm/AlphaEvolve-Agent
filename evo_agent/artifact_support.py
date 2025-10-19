#!/usr/bin/env python3
"""
Extensible Artifact Support for Generic Text Evolution.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import ast
import json
import yaml

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts that can be evolved."""
    PYTHON_CODE = "python_code"
    SQL_QUERY = "sql_query"
    CONFIG_FILE = "config_file"
    DOCUMENTATION = "documentation"
    MARKDOWN = "markdown"
    JSON_SCHEMA = "json_schema"
    YAML_CONFIG = "yaml_config"
    GENERIC_TEXT = "generic_text"


@dataclass
class DiffBlock:
    """Generic diff block for any text artifact."""
    start_line: int
    end_line: int
    original_text: str
    new_text: str
    artifact_type: ArtifactType
    metadata: Dict[str, Any] = None


@dataclass
class ArtifactCandidate:
    """Generic artifact candidate."""
    id: str
    content: str
    artifact_type: ArtifactType
    fitness_score: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = None


class ArtifactValidator:
    """Validates artifacts based on their type."""
    
    def __init__(self):
        """Initialize artifact validator."""
        self.validators: Dict[ArtifactType, Callable] = {
            ArtifactType.PYTHON_CODE: self._validate_python_code,
            ArtifactType.SQL_QUERY: self._validate_sql_query,
            ArtifactType.JSON_SCHEMA: self._validate_json_schema,
            ArtifactType.YAML_CONFIG: self._validate_yaml_config,
            ArtifactType.MARKDOWN: self._validate_markdown,
            ArtifactType.GENERIC_TEXT: self._validate_generic_text
        }
    
    def validate_artifact(self, content: str, artifact_type: ArtifactType) -> bool:
        """
        Validate artifact content.
        
        Args:
            content: Artifact content
            artifact_type: Type of artifact
            
        Returns:
            True if valid
        """
        validator = self.validators.get(artifact_type)
        if validator:
            try:
                return validator(content)
            except Exception as e:
                logger.error(f"Validation error for {artifact_type}: {e}")
                return False
        return True
    
    def _validate_python_code(self, content: str) -> bool:
        """Validate Python code."""
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False
    
    def _validate_sql_query(self, content: str) -> bool:
        """Validate SQL query."""
        # Basic SQL validation
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']
        content_upper = content.upper()
        
        # Check for basic SQL structure
        has_select = 'SELECT' in content_upper
        has_from = 'FROM' in content_upper
        has_semicolon = content.strip().endswith(';')
        
        return has_select and has_from and has_semicolon
    
    def _validate_json_schema(self, content: str) -> bool:
        """Validate JSON schema."""
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False
    
    def _validate_yaml_config(self, content: str) -> bool:
        """Validate YAML config."""
        try:
            yaml.safe_load(content)
            return True
        except yaml.YAMLError:
            return False
    
    def _validate_markdown(self, content: str) -> bool:
        """Validate Markdown content."""
        # Basic markdown validation
        has_headers = re.search(r'^#+\s+', content, re.MULTILINE)
        has_content = len(content.strip()) > 0
        return has_content
    
    def _validate_generic_text(self, content: str) -> bool:
        """Validate generic text."""
        return len(content.strip()) > 0


class ArtifactParser:
    """Parses different artifact types."""
    
    def __init__(self):
        """Initialize artifact parser."""
        self.parsers: Dict[ArtifactType, Callable] = {
            ArtifactType.PYTHON_CODE: self._parse_python_code,
            ArtifactType.SQL_QUERY: self._parse_sql_query,
            ArtifactType.JSON_SCHEMA: self._parse_json_schema,
            ArtifactType.YAML_CONFIG: self._parse_yaml_config,
            ArtifactType.MARKDOWN: self._parse_markdown,
            ArtifactType.GENERIC_TEXT: self._parse_generic_text
        }
    
    def parse_artifact(self, content: str, artifact_type: ArtifactType) -> Dict[str, Any]:
        """
        Parse artifact content.
        
        Args:
            content: Artifact content
            artifact_type: Type of artifact
            
        Returns:
            Parsed structure
        """
        parser = self.parsers.get(artifact_type)
        if parser:
            try:
                return parser(content)
            except Exception as e:
                logger.error(f"Parsing error for {artifact_type}: {e}")
                return {'error': str(e)}
        
        return {'content': content, 'type': artifact_type.value}
    
    def _parse_python_code(self, content: str) -> Dict[str, Any]:
        """Parse Python code."""
        try:
            tree = ast.parse(content)
            
            # Extract functions, classes, imports
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            
            return {
                'type': 'python_code',
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'line_count': len(content.splitlines()),
                'ast_valid': True
            }
        except SyntaxError as e:
            return {
                'type': 'python_code',
                'error': str(e),
                'ast_valid': False
            }
    
    def _parse_sql_query(self, content: str) -> Dict[str, Any]:
        """Parse SQL query."""
        # Basic SQL parsing
        lines = content.splitlines()
        keywords = []
        
        for line in lines:
            line_upper = line.upper()
            if 'SELECT' in line_upper:
                keywords.append('SELECT')
            if 'FROM' in line_upper:
                keywords.append('FROM')
            if 'WHERE' in line_upper:
                keywords.append('WHERE')
            if 'JOIN' in line_upper:
                keywords.append('JOIN')
        
        return {
            'type': 'sql_query',
            'keywords': list(set(keywords)),
            'line_count': len(lines),
            'has_semicolon': content.strip().endswith(';')
        }
    
    def _parse_json_schema(self, content: str) -> Dict[str, Any]:
        """Parse JSON schema."""
        try:
            data = json.loads(content)
            return {
                'type': 'json_schema',
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'structure': type(data).__name__,
                'valid_json': True
            }
        except json.JSONDecodeError as e:
            return {
                'type': 'json_schema',
                'error': str(e),
                'valid_json': False
            }
    
    def _parse_yaml_config(self, content: str) -> Dict[str, Any]:
        """Parse YAML config."""
        try:
            data = yaml.safe_load(content)
            return {
                'type': 'yaml_config',
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'structure': type(data).__name__,
                'valid_yaml': True
            }
        except yaml.YAMLError as e:
            return {
                'type': 'yaml_config',
                'error': str(e),
                'valid_yaml': False
            }
    
    def _parse_markdown(self, content: str) -> Dict[str, Any]:
        """Parse Markdown content."""
        lines = content.splitlines()
        headers = []
        links = []
        code_blocks = []
        
        for line in lines:
            # Extract headers
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2)
                headers.append({'level': level, 'text': text})
            
            # Extract links
            link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            links.extend(link_matches)
            
            # Extract code blocks
            if line.startswith('```'):
                code_blocks.append(line)
        
        return {
            'type': 'markdown',
            'headers': headers,
            'links': links,
            'code_blocks': len(code_blocks),
            'line_count': len(lines)
        }
    
    def _parse_generic_text(self, content: str) -> Dict[str, Any]:
        """Parse generic text."""
        lines = content.splitlines()
        words = content.split()
        
        return {
            'type': 'generic_text',
            'line_count': len(lines),
            'word_count': len(words),
            'char_count': len(content),
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1)
        }


class GenericDiffApplier:
    """Applies diffs to generic text artifacts."""
    
    def __init__(self):
        """Initialize generic diff applier."""
        self.validator = ArtifactValidator()
    
    def apply_diff(
        self, 
        original_content: str, 
        diff_blocks: List[DiffBlock], 
        artifact_type: ArtifactType
    ) -> tuple[str, bool]:
        """
        Apply diff blocks to original content.
        
        Args:
            original_content: Original content
            diff_blocks: List of diff blocks
            artifact_type: Type of artifact
            
        Returns:
            Tuple of (new_content, success)
        """
        try:
            lines = original_content.splitlines()
            new_lines = lines.copy()
            
            # Sort diff blocks by line number (descending to avoid index issues)
            sorted_blocks = sorted(diff_blocks, key=lambda x: x.start_line, reverse=True)
            
            for block in sorted_blocks:
                # Validate line numbers
                if (block.start_line < 0 or block.end_line >= len(lines) or 
                    block.start_line > block.end_line):
                    logger.error(f"Invalid line numbers in diff block: {block.start_line}-{block.end_line}")
                    return original_content, False
                
                # Apply diff
                original_lines = lines[block.start_line:block.end_line + 1]
                new_lines_block = block.new_text.splitlines()
                
                # Replace lines
                new_lines[block.start_line:block.end_line + 1] = new_lines_block
            
            new_content = '\n'.join(new_lines)
            
            # Validate result
            if self.validator.validate_artifact(new_content, artifact_type):
                return new_content, True
            else:
                logger.warning(f"Applied diff resulted in invalid {artifact_type}")
                return original_content, False
                
        except Exception as e:
            logger.error(f"Error applying diff: {e}")
            return original_content, False
    
    def create_diff_block(
        self, 
        original_content: str, 
        new_content: str, 
        artifact_type: ArtifactType,
        start_line: int = 0
    ) -> List[DiffBlock]:
        """
        Create diff blocks between original and new content.
        
        Args:
            original_content: Original content
            new_content: New content
            artifact_type: Type of artifact
            start_line: Starting line number
            
        Returns:
            List of diff blocks
        """
        original_lines = original_content.splitlines()
        new_lines = new_content.splitlines()
        
        diff_blocks = []
        
        # Simple line-by-line comparison
        max_lines = max(len(original_lines), len(new_lines))
        
        for i in range(max_lines):
            original_line = original_lines[i] if i < len(original_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""
            
            if original_line != new_line:
                diff_block = DiffBlock(
                    start_line=start_line + i,
                    end_line=start_line + i,
                    original_text=original_line,
                    new_text=new_line,
                    artifact_type=artifact_type
                )
                diff_blocks.append(diff_block)
        
        return diff_blocks


class ArtifactEvaluator:
    """Evaluates artifacts based on their type."""
    
    def __init__(self):
        """Initialize artifact evaluator."""
        self.evaluators: Dict[ArtifactType, Callable] = {
            ArtifactType.PYTHON_CODE: self._evaluate_python_code,
            ArtifactType.SQL_QUERY: self._evaluate_sql_query,
            ArtifactType.JSON_SCHEMA: self._evaluate_json_schema,
            ArtifactType.YAML_CONFIG: self._evaluate_yaml_config,
            ArtifactType.MARKDOWN: self._evaluate_markdown,
            ArtifactType.GENERIC_TEXT: self._evaluate_generic_text
        }
    
    def evaluate_artifact(
        self, 
        content: str, 
        artifact_type: ArtifactType, 
        criteria: Dict[str, Any] = None
    ) -> float:
        """
        Evaluate artifact quality.
        
        Args:
            content: Artifact content
            artifact_type: Type of artifact
            criteria: Evaluation criteria
            
        Returns:
            Quality score (0-1)
        """
        evaluator = self.evaluators.get(artifact_type)
        if evaluator:
            try:
                return evaluator(content, criteria or {})
            except Exception as e:
                logger.error(f"Evaluation error for {artifact_type}: {e}")
                return 0.0
        
        return 0.5  # Default score
    
    def _evaluate_python_code(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate Python code quality."""
        try:
            # Parse code
            tree = ast.parse(content)
            
            # Count elements
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            
            # Calculate metrics
            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Quality factors
            syntax_valid = 1.0  # Already parsed successfully
            has_functions = min(1.0, functions / 10.0)  # Bonus for functions
            has_classes = min(1.0, classes / 5.0)  # Bonus for classes
            has_imports = min(1.0, imports / 5.0)  # Bonus for imports
            code_density = len(non_empty_lines) / max(len(lines), 1)
            
            # Weighted score
            score = (
                0.3 * syntax_valid +
                0.2 * has_functions +
                0.2 * has_classes +
                0.1 * has_imports +
                0.2 * code_density
            )
            
            return min(1.0, score)
            
        except SyntaxError:
            return 0.0
    
    def _evaluate_sql_query(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate SQL query quality."""
        content_upper = content.upper()
        
        # Basic SQL structure
        has_select = 1.0 if 'SELECT' in content_upper else 0.0
        has_from = 1.0 if 'FROM' in content_upper else 0.0
        has_where = 1.0 if 'WHERE' in content_upper else 0.0
        has_semicolon = 1.0 if content.strip().endswith(';') else 0.0
        
        # Complexity factors
        has_join = 1.0 if 'JOIN' in content_upper else 0.0
        has_group_by = 1.0 if 'GROUP BY' in content_upper else 0.0
        has_order_by = 1.0 if 'ORDER BY' in content_upper else 0.0
        
        # Calculate score
        basic_score = (has_select + has_from + has_semicolon) / 3.0
        complexity_score = (has_where + has_join + has_group_by + has_order_by) / 4.0
        
        score = 0.7 * basic_score + 0.3 * complexity_score
        return min(1.0, score)
    
    def _evaluate_json_schema(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate JSON schema quality."""
        try:
            data = json.loads(content)
            
            # Structure analysis
            if isinstance(data, dict):
                has_properties = 1.0 if 'properties' in data else 0.0
                has_type = 1.0 if 'type' in data else 0.0
                has_required = 1.0 if 'required' in data else 0.0
                
                # Calculate score
                score = (has_properties + has_type + has_required) / 3.0
                return min(1.0, score)
            else:
                return 0.5  # Valid JSON but not schema-like
                
        except json.JSONDecodeError:
            return 0.0
    
    def _evaluate_yaml_config(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate YAML config quality."""
        try:
            data = yaml.safe_load(content)
            
            # Structure analysis
            if isinstance(data, dict):
                # Count nested levels
                def count_nesting(obj, level=0):
                    if isinstance(obj, dict):
                        return max(count_nesting(v, level + 1) for v in obj.values())
                    elif isinstance(obj, list):
                        return max(count_nesting(item, level + 1) for item in obj)
                    else:
                        return level
                
                nesting_level = count_nesting(data)
                complexity_score = min(1.0, nesting_level / 5.0)
                
                return 0.5 + 0.5 * complexity_score
            else:
                return 0.5  # Valid YAML but simple
                
        except yaml.YAMLError:
            return 0.0
    
    def _evaluate_markdown(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate Markdown quality."""
        lines = content.splitlines()
        
        # Count markdown elements
        headers = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        links = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
        code_blocks = len(re.findall(r'```', content))
        lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        
        # Calculate metrics
        has_headers = min(1.0, headers / 3.0)
        has_links = min(1.0, links / 2.0)
        has_code = min(1.0, code_blocks / 2.0)
        has_lists = min(1.0, lists / 3.0)
        
        # Content quality
        non_empty_lines = [line for line in lines if line.strip()]
        content_density = len(non_empty_lines) / max(len(lines), 1)
        
        # Calculate score
        score = (
            0.3 * has_headers +
            0.2 * has_links +
            0.2 * has_code +
            0.1 * has_lists +
            0.2 * content_density
        )
        
        return min(1.0, score)
    
    def _evaluate_generic_text(self, content: str, criteria: Dict[str, Any]) -> float:
        """Evaluate generic text quality."""
        lines = content.splitlines()
        words = content.split()
        
        # Basic metrics
        line_count = len(lines)
        word_count = len(words)
        char_count = len(content)
        
        # Quality factors
        has_content = 1.0 if word_count > 0 else 0.0
        avg_line_length = sum(len(line) for line in lines) / max(line_count, 1)
        line_length_score = min(1.0, avg_line_length / 80.0)  # Optimal line length
        
        # Calculate score
        score = (
            0.4 * has_content +
            0.3 * line_length_score +
            0.3 * min(1.0, word_count / 100.0)  # Content length bonus
        )
        
        return min(1.0, score)


class ArtifactManager:
    """Manages different types of artifacts."""
    
    def __init__(self):
        """Initialize artifact manager."""
        self.validator = ArtifactValidator()
        self.parser = ArtifactParser()
        self.diff_applier = GenericDiffApplier()
        self.evaluator = ArtifactEvaluator()
    
    def create_candidate(
        self, 
        content: str, 
        artifact_type: ArtifactType, 
        candidate_id: str = None
    ) -> ArtifactCandidate:
        """
        Create artifact candidate.
        
        Args:
            content: Artifact content
            artifact_type: Type of artifact
            candidate_id: Optional candidate ID
            
        Returns:
            Artifact candidate
        """
        import uuid
        
        candidate_id = candidate_id or f"candidate_{uuid.uuid4().hex[:8]}"
        
        # Validate content
        is_valid = self.validator.validate_artifact(content, artifact_type)
        
        # Parse content
        parsed = self.parser.parse_artifact(content, artifact_type)
        
        # Evaluate quality
        quality_score = self.evaluator.evaluate_artifact(content, artifact_type)
        
        return ArtifactCandidate(
            id=candidate_id,
            content=content,
            artifact_type=artifact_type,
            fitness_score=quality_score if is_valid else 0.0,
            metadata={
                'valid': is_valid,
                'parsed': parsed,
                'line_count': len(content.splitlines()),
                'char_count': len(content)
            }
        )
    
    def apply_mutation(
        self, 
        candidate: ArtifactCandidate, 
        diff_blocks: List[DiffBlock]
    ) -> ArtifactCandidate:
        """
        Apply mutation to candidate.
        
        Args:
            candidate: Original candidate
            diff_blocks: Diff blocks to apply
            
        Returns:
            New candidate
        """
        new_content, success = self.diff_applier.apply_diff(
            candidate.content, 
            diff_blocks, 
            candidate.artifact_type
        )
        
        if success:
            return self.create_candidate(
                new_content, 
                candidate.artifact_type,
                f"{candidate.id}_mutated"
            )
        else:
            # Return original candidate if mutation failed
            return candidate
    
    def get_artifact_stats(self, candidate: ArtifactCandidate) -> Dict[str, Any]:
        """
        Get statistics for artifact candidate.
        
        Args:
            candidate: Artifact candidate
            
        Returns:
            Statistics
        """
        parsed = self.parser.parse_artifact(candidate.content, candidate.artifact_type)
        
        return {
            'candidate_id': candidate.id,
            'artifact_type': candidate.artifact_type.value,
            'fitness_score': candidate.fitness_score,
            'generation': candidate.generation,
            'valid': candidate.metadata.get('valid', False),
            'line_count': candidate.metadata.get('line_count', 0),
            'char_count': candidate.metadata.get('char_count', 0),
            'parsed': parsed
        } 