#!/usr/bin/env python3
"""
Analysis & Recommendation Engine
===============================
Sophisticated system that analyzes evolution results and provides
intelligent suggestions for feedback and next steps.
"""

import re
import ast
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """Metrics from evolution cycle."""
    generation: int
    initial_fitness: float
    final_fitness: float
    improvement: float
    improvement_percentage: float
    code_length: int
    complexity_score: float
    error_rate: float
    performance_score: float

@dataclass
class ComponentAnalysis:
    """Analysis of agent component evolution."""
    component_name: str
    initial_content: str
    final_content: str
    change_magnitude: float
    key_changes: List[str]
    effectiveness_score: float

@dataclass
class CodeAnalysis:
    """Analysis of evolved code."""
    initial_code: str
    final_code: str
    added_features: List[str]
    removed_features: List[str]
    complexity_changes: Dict[str, float]
    potential_issues: List[str]
    improvement_areas: List[str]

@dataclass
class Recommendation:
    """Recommendation for next steps."""
    should_continue: bool
    confidence: float
    reasoning: str
    suggested_feedback: str
    agent_improvements: List[str]
    code_improvements: List[str]
    risk_assessment: str
    next_generation_focus: str

class AnalysisEngine:
    """Sophisticated analysis and recommendation system."""
    
    def __init__(self):
        self.evolution_history: List[EvolutionMetrics] = []
        self.component_history: List[ComponentAnalysis] = []
        self.code_history: List[CodeAnalysis] = []
        
    def analyze_evolution_cycle(self, 
                               generation: int,
                               initial_code: str,
                               final_code: str,
                               initial_fitness: float,
                               final_fitness: float,
                               initial_components: Dict[str, str],
                               final_components: Dict[str, str]) -> Tuple[CodeAnalysis, List[ComponentAnalysis], EvolutionMetrics]:
        """Analyze a complete evolution cycle."""
        
        # Analyze code evolution
        code_analysis = self._analyze_code_evolution(initial_code, final_code)
        
        # Analyze component evolution
        component_analyses = []
        for component_name in initial_components.keys():
            if component_name in final_components:
                analysis = self._analyze_component_evolution(
                    component_name,
                    initial_components[component_name],
                    final_components[component_name]
                )
                component_analyses.append(analysis)
        
        # Calculate evolution metrics
        improvement = final_fitness - initial_fitness
        improvement_percentage = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
        
        metrics = EvolutionMetrics(
            generation=generation,
            initial_fitness=initial_fitness,
            final_fitness=final_fitness,
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            code_length=len(final_code),
            complexity_score=self._calculate_complexity_score(final_code),
            error_rate=self._estimate_error_rate(final_code),
            performance_score=self._estimate_performance_score(final_code)
        )
        
        # Store in history
        self.evolution_history.append(metrics)
        self.component_history.extend(component_analyses)
        self.code_history.append(code_analysis)
        
        return code_analysis, component_analyses, metrics
    
    def _analyze_code_evolution(self, initial_code: str, final_code: str) -> CodeAnalysis:
        """Analyze how code has evolved."""
        
        # Extract functions and classes
        initial_functions = self._extract_functions(initial_code)
        final_functions = self._extract_functions(final_code)
        
        # Find added and removed features
        added_features = []
        removed_features = []
        
        for func_name, func_code in final_functions.items():
            if func_name not in initial_functions:
                added_features.append(f"Added function: {func_name}")
            else:
                # Compare function implementations
                if func_code != initial_functions[func_name]:
                    added_features.append(f"Modified function: {func_name}")
        
        for func_name in initial_functions:
            if func_name not in final_functions:
                removed_features.append(f"Removed function: {func_name}")
        
        # Analyze complexity changes
        complexity_changes = {
            'lines': len(final_code.split('\n')) - len(initial_code.split('\n')),
            'functions': len(final_functions) - len(initial_functions),
            'comments': self._count_comments(final_code) - self._count_comments(initial_code),
            'error_handling': self._count_error_handling(final_code) - self._count_error_handling(initial_code)
        }
        
        # Identify potential issues
        potential_issues = self._identify_potential_issues(final_code)
        
        # Identify improvement areas
        improvement_areas = self._identify_improvement_areas(final_code)
        
        return CodeAnalysis(
            initial_code=initial_code,
            final_code=final_code,
            added_features=added_features,
            removed_features=removed_features,
            complexity_changes=complexity_changes,
            potential_issues=potential_issues,
            improvement_areas=improvement_areas
        )
    
    def _analyze_component_evolution(self, component_name: str, initial_content: str, final_content: str) -> ComponentAnalysis:
        """Analyze how an agent component has evolved."""
        
        # Calculate change magnitude (simple character-based similarity)
        change_magnitude = 1 - (self._similarity_score(initial_content, final_content))
        
        # Extract key changes
        key_changes = self._extract_key_changes(initial_content, final_content)
        
        # Estimate effectiveness (based on length, structure, etc.)
        effectiveness_score = self._estimate_component_effectiveness(final_content)
        
        return ComponentAnalysis(
            component_name=component_name,
            initial_content=initial_content,
            final_content=final_content,
            change_magnitude=change_magnitude,
            key_changes=key_changes,
            effectiveness_score=effectiveness_score
        )
    
    def generate_recommendations(self, 
                                current_generation: int,
                                max_generations: int,
                                recent_metrics: List[EvolutionMetrics],
                                recent_code_analyses: List[CodeAnalysis],
                                recent_component_analyses: List[ComponentAnalysis]) -> Recommendation:
        """Generate intelligent recommendations for next steps."""
        
        if not recent_metrics:
            return self._generate_default_recommendation()
        
        latest_metrics = recent_metrics[-1]
        
        # Analyze trends
        trend_analysis = self._analyze_trends(recent_metrics)
        
        # Assess current state
        state_assessment = self._assess_current_state(latest_metrics, recent_code_analyses, recent_component_analyses)
        
        # Generate recommendations
        should_continue, confidence, reasoning = self._determine_continuation(
            current_generation, max_generations, latest_metrics, trend_analysis
        )
        
        suggested_feedback = self._generate_suggested_feedback(latest_metrics, recent_code_analyses)
        agent_improvements = self._suggest_agent_improvements(recent_component_analyses)
        code_improvements = self._suggest_code_improvements(recent_code_analyses)
        risk_assessment = self._assess_risks(latest_metrics, trend_analysis)
        next_generation_focus = self._determine_next_focus(latest_metrics, recent_code_analyses)
        
        return Recommendation(
            should_continue=should_continue,
            confidence=confidence,
            reasoning=reasoning,
            suggested_feedback=suggested_feedback,
            agent_improvements=agent_improvements,
            code_improvements=code_improvements,
            risk_assessment=risk_assessment,
            next_generation_focus=next_generation_focus
        )
    
    def _analyze_trends(self, metrics: List[EvolutionMetrics]) -> Dict[str, Any]:
        """Analyze evolution trends."""
        if len(metrics) < 2:
            return {"trend": "insufficient_data", "direction": "unknown"}
        
        recent_improvements = [m.improvement for m in metrics[-3:]]
        
        if len(recent_improvements) >= 2:
            if recent_improvements[-1] > recent_improvements[-2]:
                trend = "accelerating"
            elif recent_improvements[-1] < recent_improvements[-2]:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "direction": "positive" if recent_improvements[-1] > 0 else "negative",
            "average_improvement": sum(recent_improvements) / len(recent_improvements),
            "consistency": self._calculate_consistency(recent_improvements)
        }
    
    def _assess_current_state(self, metrics: EvolutionMetrics, code_analyses: List[CodeAnalysis], component_analyses: List[ComponentAnalysis]) -> Dict[str, Any]:
        """Assess the current state of evolution."""
        
        latest_code = code_analyses[-1] if code_analyses else None
        
        return {
            "fitness_level": "high" if metrics.final_fitness > 0.8 else "medium" if metrics.final_fitness > 0.6 else "low",
            "improvement_rate": "good" if metrics.improvement_percentage > 10 else "moderate" if metrics.improvement_percentage > 5 else "poor",
            "code_quality": "high" if latest_code and len(latest_code.potential_issues) == 0 else "medium",
            "complexity": "appropriate" if metrics.complexity_score < 0.7 else "high",
            "error_risk": "low" if metrics.error_rate < 0.1 else "medium" if metrics.error_rate < 0.3 else "high"
        }
    
    def _determine_continuation(self, current_generation: int, max_generations: int, metrics: EvolutionMetrics, trend_analysis: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Determine if evolution should continue."""
        
        # Base confidence on improvement
        confidence = min(0.9, max(0.1, abs(metrics.improvement_percentage) / 20))
        
        # Early termination conditions
        if metrics.final_fitness > 0.95:
            reasoning = f"Excellent fitness achieved ({metrics.final_fitness:.3f}). Consider stopping."
            return False, confidence, reasoning
        
        if metrics.improvement_percentage < 1 and current_generation > 2:
            reasoning = f"Minimal improvement ({metrics.improvement_percentage:.1f}%). Consider stopping."
            return False, confidence, reasoning
        
        if trend_analysis["trend"] == "decelerating" and current_generation > 3:
            reasoning = "Improvement rate is decreasing. Consider stopping."
            return False, confidence, reasoning
        
        # Continue conditions
        if current_generation < max_generations:
            if metrics.improvement_percentage > 5:
                reasoning = f"Good improvement ({metrics.improvement_percentage:.1f}%). Continue evolution."
                return True, confidence, reasoning
            elif metrics.final_fitness < 0.8:
                reasoning = f"Fitness still below target ({metrics.final_fitness:.3f}). Continue evolution."
                return True, confidence, reasoning
            else:
                reasoning = f"Moderate improvement. Continue for {max_generations - current_generation} more generations."
                return True, confidence, reasoning
        
        reasoning = "Maximum generations reached."
        return False, confidence, reasoning
    
    def _generate_suggested_feedback(self, metrics: EvolutionMetrics, code_analyses: List[CodeAnalysis]) -> str:
        """Generate suggested feedback for the user."""
        
        latest_code = code_analyses[-1] if code_analyses else None
        
        suggestions = []
        
        if metrics.final_fitness < 0.7:
            suggestions.append("Focus on improving correctness and error handling")
        
        if latest_code and latest_code.potential_issues:
            suggestions.append(f"Address potential issues: {', '.join(latest_code.potential_issues[:2])}")
        
        if metrics.complexity_score > 0.7:
            suggestions.append("Consider simplifying the code structure")
        
        if metrics.error_rate > 0.2:
            suggestions.append("Add more robust error handling")
        
        if not suggestions:
            suggestions.append("Focus on performance optimization and edge cases")
        
        return "; ".join(suggestions)
    
    def _suggest_agent_improvements(self, component_analyses: List[ComponentAnalysis]) -> List[str]:
        """Suggest improvements for agent components."""
        suggestions = []
        
        for analysis in component_analyses:
            if analysis.change_magnitude < 0.1:
                suggestions.append(f"Consider more significant changes to {analysis.component_name}")
            if analysis.effectiveness_score < 0.6:
                suggestions.append(f"Improve effectiveness of {analysis.component_name}")
        
        return suggestions
    
    def _suggest_code_improvements(self, code_analyses: List[CodeAnalysis]) -> List[str]:
        """Suggest improvements for the code."""
        suggestions = []
        
        latest_code = code_analyses[-1] if code_analyses else None
        if latest_code:
            if latest_code.potential_issues:
                suggestions.extend([f"Fix: {issue}" for issue in latest_code.potential_issues[:3]])
            if latest_code.improvement_areas:
                suggestions.extend([f"Improve: {area}" for area in latest_code.improvement_areas[:3]])
        
        return suggestions
    
    def _assess_risks(self, metrics: EvolutionMetrics, trend_analysis: Dict[str, Any]) -> str:
        """Assess risks of continuing evolution."""
        risks = []
        
        if metrics.error_rate > 0.3:
            risks.append("High error rate")
        
        if metrics.complexity_score > 0.8:
            risks.append("Over-complexity")
        
        if trend_analysis["trend"] == "decelerating":
            risks.append("Diminishing returns")
        
        if not risks:
            return "Low risk - safe to continue"
        else:
            return f"Risks: {', '.join(risks)}"
    
    def _determine_next_focus(self, metrics: EvolutionMetrics, code_analyses: List[CodeAnalysis]) -> str:
        """Determine focus for next generation."""
        if metrics.final_fitness < 0.6:
            return "Correctness and basic functionality"
        elif metrics.final_fitness < 0.8:
            return "Performance optimization and edge cases"
        else:
            return "Fine-tuning and polish"
    
    # Helper methods
    def _extract_functions(self, code: str) -> Dict[str, str]:
        """Extract function definitions from code."""
        functions = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = ast.unparse(node)
        except:
            pass
        return functions
    
    def _count_comments(self, code: str) -> int:
        """Count comment lines in code."""
        return len([line for line in code.split('\n') if line.strip().startswith('#')])
    
    def _count_error_handling(self, code: str) -> int:
        """Count error handling constructs."""
        error_patterns = ['try:', 'except:', 'raise', 'assert']
        count = 0
        for pattern in error_patterns:
            count += code.count(pattern)
        return count
    
    def _calculate_complexity_score(self, code: str) -> float:
        """Calculate code complexity score."""
        lines = len(code.split('\n'))
        functions = len(self._extract_functions(code))
        nesting_level = max([len(line) - len(line.lstrip()) for line in code.split('\n')], default=0)
        
        # Simple complexity formula
        return min(1.0, (lines * 0.1 + functions * 0.2 + nesting_level * 0.05) / 10)
    
    def _estimate_error_rate(self, code: str) -> float:
        """Estimate potential error rate."""
        # Simple heuristic based on code characteristics
        try_count = code.count('try:')
        except_count = code.count('except:')
        raise_count = code.count('raise')
        
        if try_count == 0:
            return 0.3  # No error handling
        elif except_count < try_count:
            return 0.2  # Incomplete error handling
        else:
            return 0.1  # Good error handling
    
    def _estimate_performance_score(self, code: str) -> float:
        """Estimate performance score."""
        # Simple heuristic
        loops = code.count('for ') + code.count('while ')
        recursion = code.count('def ') * 0.1  # Assume some recursion
        
        return max(0.1, min(1.0, 1.0 - (loops * 0.1 + recursion)))
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_key_changes(self, initial: str, final: str) -> List[str]:
        """Extract key changes between two texts."""
        changes = []
        
        # Simple word-based comparison
        initial_words = set(initial.lower().split())
        final_words = set(final.lower().split())
        
        added_words = final_words - initial_words
        removed_words = initial_words - final_words
        
        if added_words:
            changes.append(f"Added {len(added_words)} new concepts")
        if removed_words:
            changes.append(f"Removed {len(removed_words)} concepts")
        
        return changes
    
    def _estimate_component_effectiveness(self, content: str) -> float:
        """Estimate effectiveness of a component."""
        # Simple heuristic based on length and structure
        length_score = min(1.0, len(content) / 1000)
        structure_score = 0.5 if 'def ' in content or 'class ' in content else 0.3
        return (length_score + structure_score) / 2
    
    def _identify_potential_issues(self, code: str) -> List[str]:
        """Identify potential issues in code."""
        issues = []
        
        if 'TODO' in code or 'FIXME' in code:
            issues.append("Contains TODO/FIXME comments")
        
        if code.count('pass') > 3:
            issues.append("Multiple pass statements - incomplete implementation")
        
        if code.count('print(') > 5:
            issues.append("Excessive print statements - consider logging")
        
        if code.count('global ') > 2:
            issues.append("Multiple global variables - consider encapsulation")
        
        return issues
    
    def _identify_improvement_areas(self, code: str) -> List[str]:
        """Identify areas for improvement."""
        areas = []
        
        if len(code.split('\n')) < 10:
            areas.append("Add more comprehensive implementation")
        
        if code.count('def ') < 2:
            areas.append("Consider breaking into smaller functions")
        
        if code.count('try:') == 0:
            areas.append("Add error handling")
        
        if code.count('#') < 5:
            areas.append("Add more documentation")
        
        return areas
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency of improvement values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return 1.0 / (1.0 + variance)  # Higher variance = lower consistency
    
    def _generate_default_recommendation(self) -> Recommendation:
        """Generate default recommendation when no data is available."""
        return Recommendation(
            should_continue=True,
            confidence=0.5,
            reasoning="No previous data available. Starting fresh evolution.",
            suggested_feedback="Focus on basic functionality and correctness",
            agent_improvements=["All components are new"],
            code_improvements=["Implement core functionality"],
            risk_assessment="Low risk - initial generation",
            next_generation_focus="Establish baseline functionality"
        ) 