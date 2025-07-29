#!/usr/bin/env python3
"""
Lightweight LLM Interface - Decoupled from legacy document processing framework.
"""
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI, AzureOpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    model: str = "o4-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    # Azure OpenAI settings
    azure_endpoint: str = "https://vinod-m7y6fqof-eastus2.cognitiveservices.azure.com/"
    api_version: str = "2024-12-01-preview"
    deployment_name: str = "o4-mini"


class LLMInterface:
    """
    Lightweight LLM interface decoupled from legacy modules.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM interface.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        
        # Use Azure OpenAI
        self.client = AzureOpenAI(
            api_key="CxjrfpmQJB9TxEWZSTRzKTDIbqozO3kvx8S6yO0MGnfa8cdQ7HDMJQQJ99BCACHYHv6XJ3w3AAAAACOGevLG",
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.api_version
        )
        self.request_count = 0
        self.error_count = 0
        
    async def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Merge config with kwargs
        params = {
            "model": self.config.deployment_name,  # Use deployment name for Azure
            "max_completion_tokens": self.config.max_tokens,  # o4-mini uses max_completion_tokens
            **kwargs
        }
        # o4-mini only supports temperature=1 (default), so we don't set it
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **params
                )
                
                self.request_count += 1
                return response.choices[0].message.content
                
            except Exception as e:
                self.error_count += 1
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"LLM request failed after {self.config.retry_attempts} attempts")
                    raise
    
    async def generate_mutation(
        self, 
        parent_code: str, 
        task_description: str,
        previous_solutions: Optional[List[str]] = None
    ) -> str:
        """
        Generate a mutation for evolutionary optimization.
        
        Args:
            parent_code: Parent code to mutate
            task_description: Task description
            previous_solutions: Previous successful solutions
            
        Returns:
            Generated mutation
        """
        system_message = """You are an expert programmer tasked with improving code through targeted mutations.

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

        prompt = f"""
Task: {task_description}

Parent Code:
```python
{parent_code}
```

{f"Previous successful solutions for inspiration:\n{chr(10).join(previous_solutions) if previous_solutions else ''}"}

Generate targeted mutations to improve this code. Use the SEARCH/REPLACE format for each change.
"""

        return await self.generate(prompt, system_message)
    
    async def generate_edge_cases(
        self, 
        task_description: str,
        initial_tools: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate edge cases for task analysis.
        
        Args:
            task_description: Task description
            initial_tools: Initial tools if provided
            
        Returns:
            Generated edge cases
        """
        system_message = """You are an expert software engineer specializing in edge case analysis.

Your task is to identify potential edge cases, ambiguities, and failure modes for the given task.

Consider:
1. Input validation edge cases
2. Performance edge cases
3. Resource constraints
4. Error conditions
5. Boundary conditions
6. Integration edge cases

Provide a clear, numbered list of edge cases."""

        prompt = f"""
Task: {task_description}

{f"Initial tools: {initial_tools if initial_tools else 'None provided'}"}

Identify key edge cases, considerations, and potential failure modes for this task.
"""

        return await self.generate(prompt, system_message)
    
    async def generate_tool_recommendations(
        self, 
        task_description: str,
        current_tools: Dict[str, Any],
        edge_cases: List[str]
    ) -> str:
        """
        Generate tool recommendations.
        
        Args:
            task_description: Task description
            current_tools: Current tools
            edge_cases: Identified edge cases
            
        Returns:
            Generated tool recommendations
        """
        system_message = """You are an expert software architect specializing in tool and API design.

Analyze the current tools and edge cases to recommend improvements or new tools that would enhance the solution.

Consider:
1. Missing functionality
2. Performance bottlenecks
3. Error handling gaps
4. Integration needs
5. Scalability requirements

Provide specific recommendations with clear justifications."""

        prompt = f"""
Task: {task_description}

Current Tools:
{chr(10).join(f"- {name}: {desc}" for name, desc in current_tools.items())}

Identified Edge Cases:
{chr(10).join(f"- {case}" for case in edge_cases)}

Recommend improvements to the toolset to better handle these edge cases and improve the solution.
"""

        return await self.generate(prompt, system_message)
    
    async def generate_spec_recommendations(
        self, 
        task_description: str,
        current_spec: Dict[str, Any],
        edge_cases: List[str]
    ) -> str:
        """
        Generate specification recommendations.
        
        Args:
            task_description: Task description
            current_spec: Current specification
            edge_cases: Identified edge cases
            
        Returns:
            Generated spec recommendations
        """
        system_message = """You are an expert in software testing and evaluation specification.

Analyze the current evaluation specification and edge cases to recommend improvements that would create a more comprehensive and robust evaluation suite.

Consider:
1. Missing test cases
2. Performance benchmarks
3. Stress testing scenarios
4. Error condition testing
5. Edge case validation
6. Integration testing

Provide specific recommendations for test cases, metrics, and evaluation criteria."""

        prompt = f"""
Task: {task_description}

Current Specification:
{chr(10).join(f"- {key}: {value}" for key, value in current_spec.items())}

Identified Edge Cases:
{chr(10).join(f"- {case}" for case in edge_cases)}

Recommend improvements to the evaluation specification to better test these edge cases and ensure robust evaluation.
"""

        return await self.generate(prompt, system_message)
    
    async def generate_eval_suite(
        self, 
        task_description: str,
        spec_recommendations: Dict[str, Any]
    ) -> str:
        """
        Generate hard evaluation suite.
        
        Args:
            task_description: Task description
            spec_recommendations: Specification recommendations
            
        Returns:
            Generated evaluation suite
        """
        system_message = """You are an expert in software testing and evaluation.

Create a comprehensive "hard" evaluation suite with deliberate edge cases and challenging scenarios that will thoroughly test the robustness of the solution.

Include:
1. Edge case test scenarios
2. Performance stress tests
3. Error condition tests
4. Boundary value tests
5. Integration tests
6. Scalability tests

Provide specific test cases with expected inputs and outputs."""

        prompt = f"""
Task: {task_description}

Specification Recommendations:
{chr(10).join(f"- {key}: {value}" for key, value in spec_recommendations.items())}

Create a comprehensive "hard" evaluation suite with challenging test cases that will thoroughly validate the solution's robustness and performance.
"""

        return await self.generate(prompt, system_message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get LLM interface metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        } 