"""Retry utilities with exponential backoff for Med-STORM."""
import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

from ..config import settings

T = TypeVar('T')

class RetryError(Exception):
    """Custom exception for retry-related errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

def retry(
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    max_retries: Optional[int] = None,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    logger: Optional[Callable[[str], None]] = None,
):
    """
    Decorator that retries the wrapped function/method with exponential backoff.
    
    Args:
        exceptions: Exception(s) to catch and retry on. Can be a single exception
            class or a tuple of exception classes.
        max_retries: Maximum number of retry attempts. If None, uses settings.MAX_RETRIES.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier applied to delay between retries.
        jitter: If True, adds random jitter to delays to avoid thundering herd.
        logger: Optional logger function to log retry attempts.
    
    Returns:
        Decorated function that will retry on specified exceptions.
    """
    if max_retries is None:
        max_retries = settings.search_max_retries
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                retries = 0
                current_delay = initial_delay
                
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        
                        if retries > max_retries:
                            error_msg = (
                                f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                                f"Last error: {str(e)}"
                            )
                            if logger:
                                logger(error_msg)
                            raise RetryError(error_msg, e) from e
                        
                        # Calculate next delay with backoff
                        current_delay = min(
                            initial_delay * (backoff_factor ** (retries - 1)),
                            max_delay
                        )
                        
                        # Add jitter (up to 25% of current delay)
                        if jitter:
                            jitter_amount = random.uniform(0, current_delay * 0.25)
                            current_delay += jitter_amount
                        
                        # Log the retry
                        if logger:
                            logger(
                                f"Retry {retries}/{max_retries} for {func.__name__} "
                                f"after error: {str(e)}. Retrying in {current_delay:.2f}s..."
                            )
                        
                        # Wait before retrying
                        await asyncio.sleep(current_delay)
            
            return cast(Callable[..., T], async_wrapper)
        
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                retries = 0
                current_delay = initial_delay
                
                while True:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        
                        if retries > max_retries:
                            error_msg = (
                                f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                                f"Last error: {str(e)}"
                            )
                            if logger:
                                logger(error_msg)
                            raise RetryError(error_msg, e) from e
                        
                        # Calculate next delay with backoff
                        current_delay = min(
                            initial_delay * (backoff_factor ** (retries - 1)),
                            max_delay
                        )
                        
                        # Add jitter (up to 25% of current delay)
                        if jitter:
                            jitter_amount = random.uniform(0, current_delay * 0.25)
                            current_delay += jitter_amount
                        
                        # Log the retry
                        if logger:
                            logger(
                                f"Retry {retries}/{max_retries} for {func.__name__} "
                                f"after error: {str(e)}. Retrying in {current_delay:.2f}s..."
                            )
                        
                        # Wait before retrying
                        time.sleep(current_delay)
            
            return sync_wrapper
    
    return decorator

# Common retry decorators for different use cases
retry_network = retry(
    exceptions=(ConnectionError, TimeoutError, asyncio.TimeoutError),
    max_retries=settings.search_max_retries,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True
)

retry_api = retry(
    exceptions=(Exception,),
    max_retries=3,
    initial_delay=0.5,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
)
