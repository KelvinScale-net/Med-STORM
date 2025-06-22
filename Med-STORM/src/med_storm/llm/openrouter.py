import asyncio
import aiohttp
import json
from typing import Optional

from med_storm.llm.base import LLMProvider
from med_storm.config import settings

class OpenRouterLLM(LLMProvider):
    """
    A language model provider for OpenRouter API (Gemini 2.5 Flash).
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates a response from the OpenRouter model (Gemini 2.5 Flash).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": getattr(settings, "HTTP_REFERER", "https://med-storm.ai"),
            "X-Title": getattr(settings, "SITE_NAME", "Med-STORM"),
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": getattr(settings, "MAX_TOKENS_RESPONSE", 4000),
            "temperature": getattr(settings, "TEMPERATURE", 0.3),
            "top_p": getattr(settings, "TOP_P", 0.95),
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                        
        except Exception as e:
            print(f"An error occurred while generating response from OpenRouter: {e}")
            return f"Error: Could not get a response from OpenRouter. {e}"

    # ------------------------------------------------------------------
    # Compatibility method: many modules expect `generate_response`.
    # We simply forward to `generate` to maintain interface consistency.
    # ------------------------------------------------------------------

    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Alias to `generate` for backward compatibility."""
        return await self.generate(prompt, system_prompt) 