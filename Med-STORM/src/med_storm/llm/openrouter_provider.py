"""
🚀 OPENROUTER PROVIDER - GEMINI 2.5 FLASH
Provider principal ultra-rápido para Med-STORM
"""

import asyncio
import logging
from typing import Optional
import openai
from ..core.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider with Gemini 2.5 Flash - ULTRA FAST"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        site_url: str = "https://med-storm.ai",
        site_name: str = "Med-STORM"
    ):
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.is_healthy = True
        
        # Performance tracking for Gemini 2.5 Flash
        self.total_requests = 0
        self.successful_requests = 0
        
        logger.info(f"🚀 OpenRouter provider initialized with {model}")
        
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Ultra-fast generation with Gemini 2.5 Flash
        """
        self.total_requests += 1
        
        try:
            # Optimized timeout for Gemini 2.5 Flash (ultra-fast)
            timeout = kwargs.get('timeout', 30)  # Reduced from 60s
            
            # Prepare headers for OpenRouter
            extra_headers = {
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name
            }
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers=extra_headers,
                    extra_body={}
                ),
                timeout=timeout
            )
            
            content = response.choices[0].message.content
            self.successful_requests += 1
            self.is_healthy = True
            
            logger.debug(f"✅ OpenRouter generation successful ({len(content)} chars)")
            return content or ""
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ OpenRouter timeout after {timeout}s")
            self.is_healthy = False
            raise
        except Exception as e:
            logger.error(f"❌ OpenRouter generation failed: {e}")
            self.is_healthy = False
            raise
    
    async def health_check(self) -> bool:
        """Quick health check optimized for Gemini 2.5 Flash"""
        try:
            test_response = await self.generate(
                "Hello", 
                max_tokens=5, 
                timeout=10
            )
            return len(test_response) > 0
        except:
            return False
    
    @property
    def success_rate(self) -> float:
        """Get success rate for monitoring"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class UltraFastLLMRouter:
    """
    Ultra-fast LLM router optimized for Gemini 2.5 Flash
    """
    
    def __init__(self, openrouter_api_key: str, deepseek_api_key: str = None):
        """Initialize with OpenRouter as primary, DeepSeek as fallback"""
        
        # Primary: OpenRouter with Gemini 2.5 Flash (ULTRA FAST)
        self.primary_provider = OpenRouterProvider(
            api_key=openrouter_api_key,
            model="google/gemini-2.5-flash"
        )
        
        # Fallback: DeepSeek (if provided)
        self.fallback_provider = None
        if deepseek_api_key:
            from .standard_provider import StandardDeepSeekProvider
            self.fallback_provider = StandardDeepSeekProvider(deepseek_api_key)
        
        logger.info("🚀 Ultra-fast LLM Router initialized (OpenRouter + DeepSeek)")
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate with ultra-fast primary provider + fallback
        """
        
        # Try OpenRouter (Gemini 2.5 Flash) first - ULTRA FAST
        try:
            return await self.primary_provider.generate(
                prompt, max_tokens, temperature, **kwargs
            )
        except Exception as e:
            logger.warning(f"⚠️ Primary provider (OpenRouter) failed: {e}")
        
        # Fallback to DeepSeek if available
        if self.fallback_provider:
            try:
                logger.info("🔄 Falling back to DeepSeek...")
                return await self.fallback_provider.generate(
                    prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                logger.error(f"❌ Fallback provider (DeepSeek) failed: {e}")
        
        raise RuntimeError("🚨 All LLM providers failed")
    
    async def health_check(self) -> bool:
        """Check if primary provider is healthy"""
        return await self.primary_provider.health_check()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "primary_provider": {
                "model": self.primary_provider.model,
                "success_rate": self.primary_provider.success_rate,
                "total_requests": self.primary_provider.total_requests,
                "is_healthy": self.primary_provider.is_healthy
            },
            "fallback_available": self.fallback_provider is not None
        } 