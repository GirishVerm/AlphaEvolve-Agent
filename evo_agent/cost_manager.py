#!/usr/bin/env python3
"""
Cost & Rate-Limit Safeguards for LLM Usage.
"""
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Configuration for cost management."""
    max_tokens_per_request: int = 2000
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_cost_per_hour: float = 10.0  # USD
    max_cost_per_experiment: float = 50.0  # USD
    token_cost_per_1k: float = 0.015  # USD per 1k tokens (o4-mini is cheaper)
    retry_on_rate_limit: bool = True
    backoff_factor: float = 2.0
    max_retries: int = 5


class TokenTracker:
    """
    Tracks token usage and costs.
    """
    
    def __init__(self, config: CostConfig):
        """
        Initialize token tracker.
        
        Args:
            config: Cost configuration
        """
        self.config = config
        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_history: deque = deque(maxlen=1000)
        self.cost_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
    
    def add_request(self, tokens_used: int, cost: float) -> None:
        """
        Add a completed request to tracking.
        
        Args:
            tokens_used: Number of tokens used
            cost: Cost of the request
        """
        with self._lock:
            self.total_tokens += tokens_used
            self.total_cost += cost
            
            timestamp = time.time()
            self.request_history.append({
                'timestamp': timestamp,
                'tokens': tokens_used,
                'cost': cost
            })
            
            self.cost_history.append({
                'timestamp': timestamp,
                'cost': cost
            })
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Usage statistics
        """
        with self._lock:
            current_time = time.time()
            
            # Calculate rates
            requests_last_minute = sum(
                1 for req in self.request_history
                if current_time - req['timestamp'] < 60
            )
            
            requests_last_hour = sum(
                1 for req in self.request_history
                if current_time - req['timestamp'] < 3600
            )
            
            cost_last_hour = sum(
                req['cost'] for req in self.cost_history
                if current_time - req['timestamp'] < 3600
            )
            
            return {
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'requests_last_minute': requests_last_minute,
                'requests_last_hour': requests_last_hour,
                'cost_last_hour': cost_last_hour,
                'avg_tokens_per_request': self.total_tokens / max(len(self.request_history), 1),
                'avg_cost_per_request': self.total_cost / max(len(self.request_history), 1)
            }
    
    def estimate_cost(self, tokens: int) -> float:
        """
        Estimate cost for a given number of tokens.
        
        Args:
            tokens: Number of tokens
            
        Returns:
            Estimated cost
        """
        return (tokens / 1000) * self.config.token_cost_per_1k
    
    def can_make_request(self, estimated_tokens: int) -> bool:
        """
        Check if a request can be made within limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if request can be made
        """
        stats = self.get_usage_stats()
        estimated_cost = self.estimate_cost(estimated_tokens)
        
        # Check rate limits
        if stats['requests_last_minute'] >= self.config.max_requests_per_minute:
            logger.warning("Rate limit exceeded: too many requests per minute")
            return False
        
        if stats['requests_last_hour'] >= self.config.max_requests_per_hour:
            logger.warning("Rate limit exceeded: too many requests per hour")
            return False
        
        # Check cost limits
        if stats['cost_last_hour'] + estimated_cost > self.config.max_cost_per_hour:
            logger.warning("Cost limit exceeded: would exceed hourly cost limit")
            return False
        
        if self.total_cost + estimated_cost > self.config.max_cost_per_experiment:
            logger.warning("Cost limit exceeded: would exceed experiment cost limit")
            return False
        
        return True


class RateLimiter:
    """
    Rate limiter with exponential backoff.
    """
    
    def __init__(self, config: CostConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Cost configuration
        """
        self.config = config
        self.request_times: deque = deque(maxlen=1000)
        self.last_request_time = 0
        self.consecutive_failures = 0
        self.backoff_until = 0
        
        # Thread safety
        self._lock = threading.Lock()
    
    async def wait_if_needed(self) -> None:
        """
        Wait if rate limits are exceeded.
        """
        with self._lock:
            current_time = time.time()
            
            # Check backoff
            if current_time < self.backoff_until:
                wait_time = self.backoff_until - current_time
                logger.info(f"Rate limit backoff: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            
            # Check rate limits
            requests_last_minute = sum(
                1 for req_time in self.request_times
                if current_time - req_time < 60
            )
            
            if requests_last_minute >= self.config.max_requests_per_minute:
                wait_time = 60 - (current_time - self.request_times[0])
                logger.info(f"Rate limit: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
    
    def record_request(self, success: bool = True) -> None:
        """
        Record a request attempt.
        
        Args:
            success: Whether the request was successful
        """
        with self._lock:
            current_time = time.time()
            self.request_times.append(current_time)
            self.last_request_time = current_time
            
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                
                # Apply exponential backoff
                if self.consecutive_failures > 0:
                    backoff_time = min(
                        self.config.backoff_factor ** self.consecutive_failures,
                        300  # Max 5 minutes
                    )
                    self.backoff_until = current_time + backoff_time
                    logger.warning(f"Rate limit backoff: {backoff_time:.2f} seconds")
    
    def get_rate_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Returns:
            Rate limiting statistics
        """
        with self._lock:
            current_time = time.time()
            
            requests_last_minute = sum(
                1 for req_time in self.request_times
                if current_time - req_time < 60
            )
            
            requests_last_hour = sum(
                1 for req_time in self.request_times
                if current_time - req_time < 3600
            )
            
            return {
                'requests_last_minute': requests_last_minute,
                'requests_last_hour': requests_last_hour,
                'consecutive_failures': self.consecutive_failures,
                'backoff_until': self.backoff_until,
                'total_requests': len(self.request_times)
            }


class CostManager:
    """
    Manages LLM costs and rate limits.
    """
    
    def __init__(self, config: CostConfig):
        """
        Initialize cost manager.
        
        Args:
            config: Cost configuration
        """
        self.config = config
        self.token_tracker = TokenTracker(config)
        self.rate_limiter = RateLimiter(config)
        self.budget_exceeded = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    async def check_and_wait(self, estimated_tokens: int) -> bool:
        """
        Check if request can be made and wait if needed.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if request can proceed
        """
        with self._lock:
            if self.budget_exceeded:
                logger.error("Budget exceeded, cannot make more requests")
                return False
            
            # Check cost limits
            if not self.token_tracker.can_make_request(estimated_tokens):
                self.budget_exceeded = True
                logger.error("Cost limits exceeded, stopping requests")
                return False
            
            # Wait for rate limits
            await self.rate_limiter.wait_if_needed()
            return True
    
    def record_request(self, tokens_used: int, cost: float, success: bool = True) -> None:
        """
        Record a completed request.
        
        Args:
            tokens_used: Number of tokens used
            cost: Cost of the request
            success: Whether the request was successful
        """
        with self._lock:
            self.token_tracker.add_request(tokens_used, cost)
            self.rate_limiter.record_request(success)
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost and rate limiting statistics.
        
        Returns:
            Combined statistics
        """
        with self._lock:
            token_stats = self.token_tracker.get_usage_stats()
            rate_stats = self.rate_limiter.get_rate_stats()
            
            return {
                **token_stats,
                **rate_stats,
                'budget_exceeded': self.budget_exceeded
            }
    
    def estimate_remaining_requests(self) -> int:
        """
        Estimate remaining requests within budget.
        
        Returns:
            Estimated remaining requests
        """
        stats = self.get_cost_stats()
        remaining_cost = self.config.max_cost_per_experiment - stats['total_cost']
        avg_cost_per_request = stats['avg_cost_per_request']
        
        if avg_cost_per_request <= 0:
            return 0
        
        return int(remaining_cost / avg_cost_per_request)
    
    def get_cost_alerts(self) -> List[str]:
        """
        Get cost-related alerts.
        
        Returns:
            List of alerts
        """
        alerts = []
        stats = self.get_cost_stats()
        
        # Cost alerts
        if stats['total_cost'] > self.config.max_cost_per_experiment * 0.8:
            alerts.append(f"Warning: {stats['total_cost']:.2f} USD spent (80% of budget)")
        
        if stats['cost_last_hour'] > self.config.max_cost_per_hour * 0.8:
            alerts.append(f"Warning: {stats['cost_last_hour']:.2f} USD in last hour (80% of hourly limit)")
        
        # Rate limit alerts
        if stats['requests_last_minute'] > self.config.max_requests_per_minute * 0.8:
            alerts.append(f"Warning: {stats['requests_last_minute']} requests in last minute (80% of rate limit)")
        
        if stats['consecutive_failures'] > 0:
            alerts.append(f"Warning: {stats['consecutive_failures']} consecutive failures")
        
        return alerts


class AdaptiveThrottler:
    """
    Adaptive throttling based on error rates and costs.
    """
    
    def __init__(self, cost_manager: CostManager):
        """
        Initialize adaptive throttler.
        
        Args:
            cost_manager: Cost manager
        """
        self.cost_manager = cost_manager
        self.error_rate_window = 100
        self.error_history: deque = deque(maxlen=self.error_rate_window)
        self.throttle_factor = 1.0
        self.min_throttle_factor = 0.1
        self.max_throttle_factor = 2.0
    
    def record_request_result(self, success: bool, cost: float) -> None:
        """
        Record request result for adaptive throttling.
        
        Args:
            success: Whether request was successful
            cost: Cost of the request
        """
        self.error_history.append({
            'success': success,
            'cost': cost,
            'timestamp': time.time()
        })
        
        # Calculate error rate
        if len(self.error_history) >= 10:
            recent_requests = list(self.error_history)[-10:]
            error_rate = 1 - (sum(1 for req in recent_requests if req['success']) / len(recent_requests))
            
            # Adjust throttle factor based on error rate
            if error_rate > 0.1:  # More than 10% errors
                self.throttle_factor = max(self.min_throttle_factor, self.throttle_factor * 0.9)
                logger.info(f"High error rate ({error_rate:.2%}), reducing throttle factor to {self.throttle_factor:.2f}")
            elif error_rate < 0.02:  # Less than 2% errors
                self.throttle_factor = min(self.max_throttle_factor, self.throttle_factor * 1.1)
                logger.info(f"Low error rate ({error_rate:.2%}), increasing throttle factor to {self.throttle_factor:.2f}")
    
    def get_throttle_delay(self) -> float:
        """
        Get adaptive throttle delay.
        
        Returns:
            Delay in seconds
        """
        base_delay = 1.0 / self.cost_manager.config.max_requests_per_minute
        return base_delay * self.throttle_factor
    
    async def adaptive_wait(self) -> None:
        """
        Wait with adaptive throttling.
        """
        delay = self.get_throttle_delay()
        if delay > 0:
            await asyncio.sleep(delay)


class BudgetAwareLLMInterface:
    """
    Budget-aware wrapper for LLM interface.
    """
    
    def __init__(self, llm_interface, cost_config: CostConfig):
        """
        Initialize budget-aware LLM interface.
        
        Args:
            llm_interface: Base LLM interface
            cost_config: Cost configuration
        """
        self.llm_interface = llm_interface
        self.cost_manager = CostManager(cost_config)
        self.adaptive_throttler = AdaptiveThrottler(self.cost_manager)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response with budget awareness.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        # Estimate tokens
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
        
        # Check budget and rate limits
        can_proceed = await self.cost_manager.check_and_wait(estimated_tokens)
        if not can_proceed:
            raise Exception("Budget or rate limits exceeded")
        
        # Adaptive throttling
        await self.adaptive_throttler.adaptive_wait()
        
        try:
            # Make request
            start_time = time.time()
            response = await self.llm_interface.generate(prompt, **kwargs)
            end_time = time.time()
            
            # Estimate actual tokens used
            actual_tokens = len(prompt.split()) + len(response.split())
            estimated_cost = self.cost_manager.token_tracker.estimate_cost(actual_tokens)
            
            # Record successful request
            self.cost_manager.record_request(actual_tokens, estimated_cost, success=True)
            self.adaptive_throttler.record_request_result(True, estimated_cost)
            
            return response
            
        except Exception as e:
            # Record failed request
            estimated_cost = self.cost_manager.token_tracker.estimate_cost(estimated_tokens)
            self.cost_manager.record_request(estimated_tokens, estimated_cost, success=False)
            self.adaptive_throttler.record_request_result(False, estimated_cost)
            
            raise e
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost statistics.
        
        Returns:
            Cost statistics
        """
        return self.cost_manager.get_cost_stats()
    
    def get_alerts(self) -> List[str]:
        """
        Get cost and rate limit alerts.
        
        Returns:
            List of alerts
        """
        return self.cost_manager.get_cost_alerts()
    
    def is_budget_exceeded(self) -> bool:
        """
        Check if budget is exceeded.
        
        Returns:
            True if budget exceeded
        """
        return self.cost_manager.budget_exceeded 