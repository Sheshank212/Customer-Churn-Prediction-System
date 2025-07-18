"""
Rate Limiting Utilities for Customer Churn Prediction API
Implements rate limiting to prevent API abuse and ensure fair usage
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Add tokens based on elapsed time
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False


class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize sliding window rate limiter
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for identifier
        
        Args:
            identifier: Unique identifier (e.g., IP address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            request_times = self.requests[identifier]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check if we're under the limit
            if len(request_times) < self.max_requests:
                request_times.append(now)
                return True
            else:
                return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        request_times = self.requests[identifier]
        return max(0, self.max_requests - len(request_times))
    
    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get time when rate limit resets"""
        request_times = self.requests[identifier]
        if not request_times:
            return None
        
        oldest_request = request_times[0]
        reset_time = oldest_request + self.window_seconds
        return datetime.fromtimestamp(reset_time)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self, base_rate: int, max_rate: int, window_seconds: int):
        """
        Initialize adaptive rate limiter
        
        Args:
            base_rate: Base rate limit per window
            max_rate: Maximum rate limit per window
            window_seconds: Time window in seconds
        """
        self.base_rate = base_rate
        self.max_rate = max_rate
        self.current_rate = base_rate
        self.window_seconds = window_seconds
        self.limiter = SlidingWindowRateLimiter(base_rate, window_seconds)
        self.load_history = deque(maxlen=10)
        self.last_adjustment = time.time()
    
    async def is_allowed(self, identifier: str, current_load: float = 0.0) -> bool:
        """
        Check if request is allowed with adaptive rate limiting
        
        Args:
            identifier: Unique identifier
            current_load: Current system load (0.0 to 1.0)
            
        Returns:
            True if request is allowed, False otherwise
        """
        # Update load history
        self.load_history.append(current_load)
        
        # Adjust rate based on load every 60 seconds
        now = time.time()
        if now - self.last_adjustment > 60:
            await self._adjust_rate()
            self.last_adjustment = now
        
        return await self.limiter.is_allowed(identifier)
    
    async def _adjust_rate(self):
        """Adjust rate based on recent load"""
        if not self.load_history:
            return
        
        avg_load = sum(self.load_history) / len(self.load_history)
        
        if avg_load > 0.8:  # High load
            new_rate = max(self.base_rate // 2, self.base_rate)
        elif avg_load < 0.3:  # Low load
            new_rate = min(self.max_rate, int(self.current_rate * 1.2))
        else:  # Normal load
            new_rate = self.base_rate
        
        if new_rate != self.current_rate:
            self.current_rate = new_rate
            self.limiter = SlidingWindowRateLimiter(new_rate, self.window_seconds)
            logger.info(f"Adjusted rate limit to {new_rate} requests per {self.window_seconds}s")


class RateLimiterConfig:
    """Configuration for different rate limiters"""
    
    # Different tiers of rate limiting
    TIERS = {
        "free": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_capacity": 10
        },
        "premium": {
            "requests_per_minute": 300,
            "requests_per_hour": 10000,
            "burst_capacity": 50
        },
        "enterprise": {
            "requests_per_minute": 1000,
            "requests_per_hour": 50000,
            "burst_capacity": 200
        }
    }
    
    @classmethod
    def get_limiter_for_tier(cls, tier: str) -> Dict[str, SlidingWindowRateLimiter]:
        """Get rate limiters for a specific tier"""
        if tier not in cls.TIERS:
            tier = "free"
        
        config = cls.TIERS[tier]
        
        return {
            "per_minute": SlidingWindowRateLimiter(
                config["requests_per_minute"], 60
            ),
            "per_hour": SlidingWindowRateLimiter(
                config["requests_per_hour"], 3600
            ),
            "burst": TokenBucket(
                config["burst_capacity"], 
                config["requests_per_minute"] / 60.0
            )
        }


class RateLimitMiddleware:
    """Middleware for applying rate limits to FastAPI"""
    
    def __init__(self, app, default_tier: str = "free"):
        """
        Initialize rate limit middleware
        
        Args:
            app: FastAPI app instance
            default_tier: Default rate limit tier
        """
        self.app = app
        self.default_tier = default_tier
        self.limiters: Dict[str, Dict] = {}
        self.blocked_ips = set()
        
    def get_client_id(self, request) -> str:
        """Extract client identifier from request"""
        # Try to get real IP from headers (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def get_user_tier(self, request) -> str:
        """Determine user tier from request (e.g., API key, auth)"""
        # In production, this would check API keys, auth tokens, etc.
        api_key = request.headers.get("X-API-Key")
        
        if api_key == "enterprise_key":
            return "enterprise"
        elif api_key == "premium_key":
            return "premium"
        else:
            return "free"
    
    async def check_rate_limit(self, client_id: str, tier: str) -> tuple[bool, Dict]:
        """
        Check if request should be rate limited
        
        Returns:
            (allowed, headers) tuple
        """
        if client_id in self.blocked_ips:
            return False, {"X-RateLimit-Blocked": "true"}
        
        # Get or create limiters for this client
        if client_id not in self.limiters:
            self.limiters[client_id] = RateLimiterConfig.get_limiter_for_tier(tier)
        
        limiters = self.limiters[client_id]
        
        # Check all rate limits
        per_minute_ok = await limiters["per_minute"].is_allowed(client_id)
        per_hour_ok = await limiters["per_hour"].is_allowed(client_id)
        burst_ok = await limiters["burst"].consume(1)
        
        allowed = per_minute_ok and per_hour_ok and burst_ok
        
        # Prepare headers
        headers = {
            "X-RateLimit-Limit-Minute": str(RateLimiterConfig.TIERS[tier]["requests_per_minute"]),
            "X-RateLimit-Limit-Hour": str(RateLimiterConfig.TIERS[tier]["requests_per_hour"]),
            "X-RateLimit-Remaining-Minute": str(limiters["per_minute"].get_remaining_requests(client_id)),
            "X-RateLimit-Remaining-Hour": str(limiters["per_hour"].get_remaining_requests(client_id)),
        }
        
        if not allowed:
            reset_time = limiters["per_minute"].get_reset_time(client_id)
            if reset_time:
                headers["X-RateLimit-Reset"] = str(int(reset_time.timestamp()))
        
        return allowed, headers


# Global rate limiter instance
rate_limiter = RateLimitMiddleware(None)