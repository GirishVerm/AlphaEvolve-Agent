#!/usr/bin/env python3
"""
Pluggable Human-in-the-Loop Interfaces with Web Callbacks and Message Queues.
"""
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
from queue import Queue, Empty
import uuid

logger = logging.getLogger(__name__)


class InterfaceType(Enum):
    """Types of human interface."""
    CLI = "cli"
    WEB_CALLBACK = "web_callback"
    MESSAGE_QUEUE = "message_queue"
    WEBSOCKET = "websocket"
    EMAIL = "email"


@dataclass
class HumanReviewRequest:
    """Request for human review."""
    request_id: str
    candidate_id: str
    candidate_code: str
    candidate_prompt: str
    fitness_score: float
    generation: int
    timestamp: datetime
    context: Dict[str, Any]
    priority: int = 1  # Higher = more urgent


@dataclass
class HumanReviewResponse:
    """Response from human review."""
    request_id: str
    approved: bool
    feedback: str
    suggested_changes: Optional[str] = None
    confidence: float = 1.0
    timestamp: datetime = None


class HumanInterfaceConfig:
    """Configuration for human interface."""
    
    def __init__(
        self,
        interface_type: InterfaceType = InterfaceType.CLI,
        timeout_seconds: int = 300,
        batch_size: int = 10,
        max_queue_size: int = 1000,
        webhook_url: Optional[str] = None,
        websocket_url: Optional[str] = None,
        email_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize human interface configuration.
        
        Args:
            interface_type: Type of interface to use
            timeout_seconds: Timeout for human responses
            batch_size: Number of requests to batch
            max_queue_size: Maximum queue size
            webhook_url: Webhook URL for callbacks
            websocket_url: WebSocket URL
            email_config: Email configuration
        """
        self.interface_type = interface_type
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.webhook_url = webhook_url
        self.websocket_url = websocket_url
        self.email_config = email_config or {}


class BaseHumanInterface:
    """Base class for human interfaces."""
    
    def __init__(self, config: HumanInterfaceConfig):
        """
        Initialize base human interface.
        
        Args:
            config: Human interface configuration
        """
        self.config = config
        self.pending_requests: Dict[str, HumanReviewRequest] = {}
        self.completed_responses: Dict[str, HumanReviewResponse] = {}
        self.request_queue: Queue = Queue(maxsize=config.max_queue_size)
        self.response_queue: Queue = Queue(maxsize=config.max_queue_size)
        
        # Thread safety
        self._lock = threading.Lock()
        self._running = False
    
    async def submit_for_review(self, request: HumanReviewRequest) -> str:
        """
        Submit candidate for human review.
        
        Args:
            request: Review request
            
        Returns:
            Request ID
        """
        with self._lock:
            self.pending_requests[request.request_id] = request
            self.request_queue.put(request)
            
            logger.info(f"Submitted review request {request.request_id}")
            return request.request_id
    
    async def get_review_response(self, request_id: str, timeout: Optional[int] = None) -> Optional[HumanReviewResponse]:
        """
        Get review response for a request.
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            Review response or None if timeout
        """
        timeout = timeout or self.config.timeout_seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if response is ready
            if request_id in self.completed_responses:
                response = self.completed_responses[request_id]
                del self.completed_responses[request_id]
                return response
            
            # Wait for response
            try:
                response = self.response_queue.get(timeout=1.0)
                if response.request_id == request_id:
                    return response
                else:
                    # Put back in queue for other consumers
                    self.response_queue.put(response)
            except Empty:
                continue
        
        logger.warning(f"Timeout waiting for review response {request_id}")
        return None
    
    def start(self) -> None:
        """Start the human interface."""
        self._running = True
        logger.info(f"Started human interface: {self.config.interface_type}")
    
    def stop(self) -> None:
        """Stop the human interface."""
        self._running = False
        logger.info("Stopped human interface")
    
    def is_running(self) -> bool:
        """Check if interface is running."""
        return self._running


class CLIHumanInterface(BaseHumanInterface):
    """Command-line interface for human review."""
    
    def __init__(self, config: HumanInterfaceConfig):
        """
        Initialize CLI human interface.
        
        Args:
            config: Human interface configuration
        """
        super().__init__(config)
        self.review_thread = None
    
    def start(self) -> None:
        """Start CLI interface."""
        super().start()
        self.review_thread = threading.Thread(target=self._review_loop, daemon=True)
        self.review_thread.start()
    
    def _review_loop(self) -> None:
        """Main review loop for CLI interface."""
        while self._running:
            try:
                # Get next request
                request = self.request_queue.get(timeout=1.0)
                
                # Present request to user
                response = self._present_request_cli(request)
                
                # Add response to queue
                self.response_queue.put(response)
                self.completed_responses[request.request_id] = response
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in CLI review loop: {e}")
    
    def _present_request_cli(self, request: HumanReviewRequest) -> HumanReviewResponse:
        """
        Present request to user via CLI.
        
        Args:
            request: Review request
            
        Returns:
            Human response
        """
        print("\n" + "="*80)
        print(f"HUMAN REVIEW REQUEST - Generation {request.generation}")
        print("="*80)
        print(f"Request ID: {request.request_id}")
        print(f"Candidate ID: {request.candidate_id}")
        print(f"Fitness Score: {request.fitness_score:.3f}")
        print(f"Priority: {request.priority}")
        print(f"Timestamp: {request.timestamp}")
        
        if request.context:
            print("\nContext:")
            for key, value in request.context.items():
                print(f"  {key}: {value}")
        
        print("\nCandidate Code:")
        print("-" * 40)
        print(request.candidate_code)
        print("-" * 40)
        
        print("\nCandidate Prompt:")
        print("-" * 40)
        print(request.candidate_prompt)
        print("-" * 40)
        
        # Get user input
        while True:
            print("\nReview Options:")
            print("1. Approve")
            print("2. Reject")
            print("3. Approve with feedback")
            print("4. Skip (auto-approve)")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                feedback = input("Optional feedback: ").strip()
                return HumanReviewResponse(
                    request_id=request.request_id,
                    approved=True,
                    feedback=feedback,
                    timestamp=datetime.now()
                )
            
            elif choice == "2":
                feedback = input("Rejection reason: ").strip()
                return HumanReviewResponse(
                    request_id=request.request_id,
                    approved=False,
                    feedback=feedback,
                    timestamp=datetime.now()
                )
            
            elif choice == "3":
                feedback = input("Feedback: ").strip()
                changes = input("Suggested changes (optional): ").strip()
                return HumanReviewResponse(
                    request_id=request.request_id,
                    approved=True,
                    feedback=feedback,
                    suggested_changes=changes if changes else None,
                    timestamp=datetime.now()
                )
            
            elif choice == "4":
                return HumanReviewResponse(
                    request_id=request.request_id,
                    approved=True,
                    feedback="Auto-approved (skipped)",
                    timestamp=datetime.now()
                )
            
            else:
                print("Invalid choice. Please enter 1-4.")


class WebCallbackInterface(BaseHumanInterface):
    """Web callback interface for human review."""
    
    def __init__(self, config: HumanInterfaceConfig):
        """
        Initialize web callback interface.
        
        Args:
            config: Human interface configuration
        """
        super().__init__(config)
        self.callback_handlers: Dict[str, Callable] = {}
    
    async def register_callback(self, request_id: str, callback: Callable) -> None:
        """
        Register callback for a request.
        
        Args:
            request_id: Request ID
            callback: Callback function
        """
        self.callback_handlers[request_id] = callback
    
    async def submit_for_review(self, request: HumanReviewRequest) -> str:
        """
        Submit candidate for review via web callback.
        
        Args:
            request: Review request
            
        Returns:
            Request ID
        """
        request_id = await super().submit_for_review(request)
        
        # Send webhook notification
        await self._send_webhook_notification(request)
        
        return request_id
    
    async def _send_webhook_notification(self, request: HumanReviewRequest) -> None:
        """
        Send webhook notification for review request.
        
        Args:
            request: Review request
        """
        if not self.config.webhook_url:
            logger.warning("No webhook URL configured")
            return
        
        try:
            import aiohttp
            
            payload = {
                'request_id': request.request_id,
                'candidate_id': request.candidate_id,
                'candidate_code': request.candidate_code,
                'candidate_prompt': request.candidate_prompt,
                'fitness_score': request.fitness_score,
                'generation': request.generation,
                'timestamp': request.timestamp.isoformat(),
                'context': request.context,
                'priority': request.priority
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for {request.request_id}")
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        
        except ImportError:
            logger.error("aiohttp not available for webhook notifications")
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def handle_webhook_response(self, request_id: str, response_data: Dict[str, Any]) -> None:
        """
        Handle webhook response from human review.
        
        Args:
            request_id: Request ID
            response_data: Response data
        """
        response = HumanReviewResponse(
            request_id=request_id,
            approved=response_data.get('approved', False),
            feedback=response_data.get('feedback', ''),
            suggested_changes=response_data.get('suggested_changes'),
            confidence=response_data.get('confidence', 1.0),
            timestamp=datetime.now()
        )
        
        # Add to response queue
        self.response_queue.put(response)
        self.completed_responses[request_id] = response
        
        # Call registered callback if exists
        if request_id in self.callback_handlers:
            try:
                await self.callback_handlers[request_id](response)
            except Exception as e:
                logger.error(f"Error in callback for {request_id}: {e}")


class MessageQueueInterface(BaseHumanInterface):
    """Message queue interface for human review."""
    
    def __init__(self, config: HumanInterfaceConfig):
        """
        Initialize message queue interface.
        
        Args:
            config: Human interface configuration
        """
        super().__init__(config)
        self.queue_name = f"human_review_{uuid.uuid4().hex[:8]}"
        self.consumer_thread = None
    
    def start(self) -> None:
        """Start message queue interface."""
        super().start()
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
    
    def _consumer_loop(self) -> None:
        """Message queue consumer loop."""
        while self._running:
            try:
                # Simulate message queue consumption
                # In practice, this would connect to Redis, RabbitMQ, etc.
                time.sleep(1.0)
                
                # Check for responses from external queue
                # This is a simplified implementation
                
            except Exception as e:
                logger.error(f"Error in message queue consumer: {e}")
    
    async def submit_for_review(self, request: HumanReviewRequest) -> str:
        """
        Submit candidate for review via message queue.
        
        Args:
            request: Review request
            
        Returns:
            Request ID
        """
        request_id = await super().submit_for_review(request)
        
        # Publish to message queue
        await self._publish_to_queue(request)
        
        return request_id
    
    async def _publish_to_queue(self, request: HumanReviewRequest) -> None:
        """
        Publish request to message queue.
        
        Args:
            request: Review request
        """
        # Simplified implementation
        # In practice, this would publish to Redis, RabbitMQ, etc.
        message = {
            'type': 'review_request',
            'data': asdict(request)
        }
        
        logger.info(f"Published review request {request.request_id} to queue {self.queue_name}")
        
        # Simulate external processing
        asyncio.create_task(self._simulate_external_processing(request))
    
    async def _simulate_external_processing(self, request: HumanReviewRequest) -> None:
        """
        Simulate external processing of review request.
        
        Args:
            request: Review request
        """
        # Simulate processing time
        await asyncio.sleep(5.0)
        
        # Simulate human response
        response = HumanReviewResponse(
            request_id=request.request_id,
            approved=True,  # Auto-approve for simulation
            feedback="Auto-processed by message queue",
            timestamp=datetime.now()
        )
        
        # Add to response queue
        self.response_queue.put(response)
        self.completed_responses[request.request_id] = response


class BatchHumanReviewManager:
    """
    Manages batch human review operations.
    """
    
    def __init__(self, interface: BaseHumanInterface):
        """
        Initialize batch review manager.
        
        Args:
            interface: Human interface
        """
        self.interface = interface
        self.batch_queue: List[HumanReviewRequest] = []
        self.batch_timeout = 60  # seconds
        self.last_batch_time = time.time()
    
    async def add_to_batch(self, request: HumanReviewRequest) -> str:
        """
        Add request to batch.
        
        Args:
            request: Review request
            
        Returns:
            Request ID
        """
        self.batch_queue.append(request)
        
        # Submit immediately if batch is full or timeout reached
        if (len(self.batch_queue) >= self.interface.config.batch_size or
            time.time() - self.last_batch_time >= self.batch_timeout):
            await self._submit_batch()
        
        return request.request_id
    
    async def _submit_batch(self) -> None:
        """Submit current batch for review."""
        if not self.batch_queue:
            return
        
        # Create batch request
        batch_request = HumanReviewRequest(
            request_id=f"batch_{uuid.uuid4().hex[:8]}",
            candidate_id="batch",
            candidate_code="\n---\n".join(req.candidate_code for req in self.batch_queue),
            candidate_prompt="\n---\n".join(req.candidate_prompt for req in self.batch_queue),
            fitness_score=sum(req.fitness_score for req in self.batch_queue) / len(self.batch_queue),
            generation=max(req.generation for req in self.batch_queue),
            timestamp=datetime.now(),
            context={
                'batch_size': len(self.batch_queue),
                'individual_requests': [req.request_id for req in self.batch_queue]
            }
        )
        
        # Submit batch
        await self.interface.submit_for_review(batch_request)
        
        # Clear batch
        self.batch_queue.clear()
        self.last_batch_time = time.time()
        
        logger.info(f"Submitted batch of {len(self.batch_queue)} requests")


class HumanInterfaceFactory:
    """Factory for creating human interfaces."""
    
    @staticmethod
    def create_interface(config: HumanInterfaceConfig) -> BaseHumanInterface:
        """
        Create human interface based on configuration.
        
        Args:
            config: Human interface configuration
            
        Returns:
            Human interface instance
        """
        if config.interface_type == InterfaceType.CLI:
            return CLIHumanInterface(config)
        elif config.interface_type == InterfaceType.WEB_CALLBACK:
            return WebCallbackInterface(config)
        elif config.interface_type == InterfaceType.MESSAGE_QUEUE:
            return MessageQueueInterface(config)
        else:
            raise ValueError(f"Unsupported interface type: {config.interface_type}")


class AsyncHumanReviewManager:
    """
    Asynchronous human review manager.
    """
    
    def __init__(self, interface: BaseHumanInterface):
        """
        Initialize async human review manager.
        
        Args:
            interface: Human interface
        """
        self.interface = interface
        self.pending_reviews: Dict[str, asyncio.Future] = {}
        self.review_stats = {
            'total_requests': 0,
            'approved_requests': 0,
            'rejected_requests': 0,
            'timeout_requests': 0
        }
    
    async def request_review(
        self, 
        candidate: Any, 
        context: Dict[str, Any] = None
    ) -> Optional[HumanReviewResponse]:
        """
        Request human review for a candidate.
        
        Args:
            candidate: Candidate to review
            context: Additional context
            
        Returns:
            Review response or None if timeout
        """
        # Create review request
        request = HumanReviewRequest(
            request_id=uuid.uuid4().hex,
            candidate_id=getattr(candidate, 'id', 'unknown'),
            candidate_code=getattr(candidate, 'code', ''),
            candidate_prompt=getattr(candidate, 'prompt', ''),
            fitness_score=getattr(candidate, 'fitness_score', 0.0),
            generation=getattr(candidate, 'generation', 0),
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Submit for review
        await self.interface.submit_for_review(request)
        
        # Wait for response
        response = await self.interface.get_review_response(request.request_id)
        
        # Update stats
        self.review_stats['total_requests'] += 1
        if response:
            if response.approved:
                self.review_stats['approved_requests'] += 1
            else:
                self.review_stats['rejected_requests'] += 1
        else:
            self.review_stats['timeout_requests'] += 1
        
        return response
    
    def get_review_stats(self) -> Dict[str, Any]:
        """
        Get review statistics.
        
        Returns:
            Review statistics
        """
        return self.review_stats.copy() 