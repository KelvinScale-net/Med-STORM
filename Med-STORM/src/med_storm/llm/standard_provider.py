"""
🤖 STANDARD LLM PROVIDER
Implementación estándar que sigue las interfaces definidas
"""

import asyncio
import logging
from typing import Optional, List
import openai
from ..core.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class StandardDeepSeekProvider(LLMProvider):
    """Standard DeepSeek provider following interfaces"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "deepseek-chat"
        self.is_healthy = True
        
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Standard generate method - ONLY this method, no variants
        """
        try:
            # Standard timeout handling
            timeout = kwargs.get('timeout', 60)
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                ),
                timeout=timeout
            )
            
            content = response.choices[0].message.content
            self.is_healthy = True
            return content or ""
            
        except asyncio.TimeoutError:
            logger.error(f"DeepSeek timeout after {timeout}s")
            self.is_healthy = False
            raise
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            self.is_healthy = False
            raise
    
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        try:
            test_response = await self.generate(
                "Test", 
                max_tokens=10, 
                timeout=10
            )
            return len(test_response) > 0
        except:
            return False


class StandardLLMRouter:
    """Standard LLM router with failover"""
    
    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.primary_provider: Optional[LLMProvider] = None
        
    def add_provider(self, provider: LLMProvider, is_primary: bool = False):
        """Add a provider to the router"""
        self.providers.append(provider)
        if is_primary or not self.primary_provider:
            self.primary_provider = provider
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate with automatic failover
        """
        if not self.providers:
            raise ValueError("No LLM providers configured")
        
        # Try primary provider first
        if self.primary_provider:
            try:
                return await self.primary_provider.generate(
                    prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
        
        # Try other providers
        for provider in self.providers:
            if provider == self.primary_provider:
                continue
            try:
                return await provider.generate(
                    prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                logger.warning(f"Provider failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed")
    
    async def health_check(self) -> bool:
        """Check if any provider is healthy"""
        for provider in self.providers:
            if await provider.health_check():
                return True
        return False 