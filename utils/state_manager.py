"""
State Management and Isolation Utilities

This module provides state isolation and cleanup mechanisms to prevent
shared state corruption from affecting multiple requests.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional
import copy
import traceback

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages backend state isolation and prevents corruption across requests
    """
    
    def __init__(self):
        self._state_lock = threading.Lock()
        self._request_states: Dict[str, Dict[str, Any]] = {}
        self._global_state_health = True
        
    @contextmanager
    def isolated_request_context(self, request_id: str):
        """
        Create an isolated context for a request to prevent state corruption
        
        Args:
            request_id: Unique identifier for this request
        """
        logger.info(f"Starting isolated context for request: {request_id}")
        
        try:
            # Initialize clean state for this request
            with self._state_lock:
                self._request_states[request_id] = {
                    'started': True,
                    'corrupted': False,
                    'error_count': 0
                }
            
            # Yield control to the request handler
            yield self._request_states[request_id]
            
        except Exception as e:
            logger.error(f"Exception in request {request_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Mark this request as corrupted but don't affect others
            with self._state_lock:
                if request_id in self._request_states:
                    self._request_states[request_id]['corrupted'] = True
                    self._request_states[request_id]['error_count'] += 1
            
            # Don't propagate the corruption - re-raise for handling
            raise
            
        finally:
            # Clean up request-specific state
            self._cleanup_request_state(request_id)
            
    def _cleanup_request_state(self, request_id: str):
        """
        Clean up state for a completed request
        
        Args:
            request_id: Request identifier to clean up
        """
        logger.info(f"Cleaning up state for request: {request_id}")
        
        try:
            with self._state_lock:
                if request_id in self._request_states:
                    request_state = self._request_states[request_id]
                    
                    if request_state.get('corrupted', False):
                        logger.warning(f"Request {request_id} was corrupted, performing deep cleanup")
                        # Perform additional cleanup for corrupted requests
                        self._perform_deep_cleanup()
                    
                    # Remove request state
                    del self._request_states[request_id]
                    
        except Exception as e:
            logger.error(f"Error during cleanup for request {request_id}: {str(e)}")
            
    def _perform_deep_cleanup(self):
        """
        Perform deep cleanup when corruption is detected
        """
        logger.warning("Performing deep state cleanup due to corruption")
        
        # This is where we can add specific cleanup logic for:
        # - Vector store client connections
        # - LLM client state
        # - Context enhancement pipeline
        # - Document retriever state
        
    def check_global_health(self) -> bool:
        """
        Check if the global state is healthy
        
        Returns:
            True if state is healthy, False if corrupted
        """
        with self._state_lock:
            # Count recent corrupted requests
            recent_corrupted = sum(
                1 for state in self._request_states.values() 
                if state.get('corrupted', False)
            )
            
            # If too many requests are corrupted, mark global state as unhealthy
            if recent_corrupted > 3:
                self._global_state_health = False
                logger.error("Global state marked as unhealthy due to multiple corrupted requests")
                
            return self._global_state_health
            
    def reset_global_health(self):
        """
        Reset global health status (typically after restart or cleanup)
        """
        with self._state_lock:
            self._global_state_health = True
            self._request_states.clear()
            logger.info("Global state health reset")

# Global state manager instance
state_manager = StateManager()


def isolated_request(request_id: str):
    """
    Decorator to wrap request handlers with state isolation
    
    Args:
        request_id: Unique identifier for the request
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with state_manager.isolated_request_context(request_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_shared_state() -> Dict[str, Any]:
    """
    Validate the health of shared state components
    
    Returns:
        Dictionary with health status of different components
    """
    health_status = {
        'global_health': state_manager.check_global_health(),
        'active_requests': len(state_manager._request_states),
        'timestamp': __import__('time').time()
    }
    
    logger.info(f"State health check: {health_status}")
    return health_status
