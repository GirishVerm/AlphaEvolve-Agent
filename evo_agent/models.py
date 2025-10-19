#!/usr/bin/env python3
"""
Pydantic models for evaluation specifications and results.
"""
from __future__ import annotations

from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


NAME_MIN_LEN = 1


class TestCase(BaseModel):
    """Single correctness test case."""
    name: str = Field(min_length=NAME_MIN_LEN)
    code: str = Field(min_length=1)  # code that sets `result`
    expected: Any


class Benchmark(BaseModel):
    """Performance benchmark specification."""
    name: str = Field(min_length=NAME_MIN_LEN)
    code: str = Field(min_length=1)
    function_call: str = Field(min_length=1)
    iterations: int = Field(default=100, ge=1)
    target: float = Field(gt=0)
    baseline: Optional[float] = Field(default=None, gt=0)


class RobustnessTest(BaseModel):
    """Robustness test specification."""
    name: str = Field(min_length=NAME_MIN_LEN)
    code: str = Field(min_length=1)
    function_call: str = Field(min_length=1)
    expect_exception: bool = False


class SuccessCriteria(BaseModel):
    """Success thresholds for evaluation dimensions."""
    correctness_threshold: float = Field(ge=0, le=1)
    performance_threshold_ms: float = Field(gt=0)
    robustness_threshold: float = Field(ge=0, le=1)


class TaskSpec(BaseModel):
    """Full task evaluation specification."""
    version: str = Field(default="1.0.0")
    test_cases: List[TestCase] = Field(default_factory=list)
    performance_benchmarks: List[Benchmark] = Field(default_factory=list)
    robustness_tests: List[RobustnessTest] = Field(default_factory=list)
    success_criteria: SuccessCriteria

    @field_validator('test_cases')
    @classmethod
    def _validate_test_cases(cls, v: List[TestCase]) -> List[TestCase]:
        cls._validate_no_banned(v)
        return v

    @field_validator('performance_benchmarks')
    @classmethod
    def _validate_benchmarks(cls, v: List[Benchmark]) -> List[Benchmark]:
        cls._validate_no_banned(v)
        return v

    @field_validator('robustness_tests')
    @classmethod
    def _validate_robustness(cls, v: List[RobustnessTest]) -> List[RobustnessTest]:
        cls._validate_no_banned(v)
        return v

    @staticmethod
    def _validate_no_banned(items: List[BaseModel]) -> None:
        banned_tokens = ("import os", "subprocess.", "open(")
        for item in items:
            text_fields: List[str] = []
            for field_name in ("code", "function_call"):
                value = getattr(item, field_name, None)
                if isinstance(value, str):
                    text_fields.append(value)
            text = "\n".join(text_fields)
            if any(token in text for token in banned_tokens):
                raise ValueError("Evaluation spec contains banned tokens (os/subprocess/open)")


class EvaluationResult(BaseModel):
    """Structured evaluation result."""
    candidate_id: str
    overall_score: float
    correctness_score: float
    performance_score: float
    robustness_score: float
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_time: float


