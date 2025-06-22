#!/usr/bin/env python3
"""
🚀 PRODUCTION-GRADE LLM ROUTER
Multi-provider fallback system for maximum reliability
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import httpx
from openai import AsyncOpenAI
import json
import os

from med_storm.llm.openrouter import OpenRouterLLM

# Allow flexible import regardless of current working directory modifications
try:
    from config.settings import settings  # Project-root config package
except ModuleNotFoundError:  # Fallback if 'config' not resolvable
    import importlib.util as _import_utils
    import sys as _sys, pathlib as _pl

    _project_root = _pl.Path(__file__).resolve().parents[3]  # ../../..
    if str(_project_root) not in _sys.path:
        _sys.path.append(str(_project_root))
    settings = getattr(__import__('config.settings', fromlist=['settings']), 'settings')

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 60
    priority: int = 1  # Lower = higher priority
    enabled: bool = True

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class LLMRouter:
    """
    Simplified LLM Router using only OpenRouter/Gemini 2.5 Flash
    Removed complex fallback system for maximum reliability
    """
    
    def __init__(self, openrouter_api_key: str, model: str = "google/gemini-2.5-flash"):
        """Initialize with single OpenRouter provider"""
        self.llm = OpenRouterLLM(api_key=openrouter_api_key, model=model)
        logger.info(f"🚀 LLM Router initialized with {model}")
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenRouter/Gemini 2.5 Flash"""
        try:
            response = await self.llm.generate(prompt, system_prompt)
            logger.info("✅ OpenRouter request successful")
            return response
        except Exception as e:
            logger.error(f"❌ OpenRouter request failed: {e}")
            raise
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Alias for generate method for backward compatibility"""
        return await self.generate(prompt, system_prompt, **kwargs)

# Global router instance
_router_instance: Optional[LLMRouter] = None

def get_llm_router() -> LLMRouter:
    """Get the global LLM router instance"""
    global _router_instance
    if _router_instance is None:
        api_key = settings.OPENROUTER_API_KEY.get_secret_value() if settings.OPENROUTER_API_KEY else os.getenv('OPENROUTER_API_KEY')
        _router_instance = LLMRouter(api_key)
    return _router_instance

async def generate_with_fallback(prompt: str, **kwargs) -> str:
    """Convenience function for generating responses with fallback"""
    router = get_llm_router()
    return await router.generate(prompt, **kwargs)

class ProductionLLMRouter:
    """Simplified production router returning a singleton OpenRouterLLM instance."""
    _instance = None

    @classmethod
    def get_llm(cls):
        if cls._instance is None:
            api_key = settings.OPENROUTER_API_KEY.get_secret_value() if settings.OPENROUTER_API_KEY else os.getenv('OPENROUTER_API_KEY')
            cls._instance = OpenRouterLLM(api_key=api_key)
        return cls._instance 