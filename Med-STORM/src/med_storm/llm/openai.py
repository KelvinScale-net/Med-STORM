from openai import AsyncOpenAI
from typing import Optional

from .base import LLMProvider
from med_storm.config import settings

class OpenAILLM(LLMProvider):
    """LLM Provider for OpenAI models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Please set it in your .env file.")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generates a text completion using the OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Proper error handling is crucial here
            print(f"An error occurred while calling the OpenAI API: {e}")
            return "Error: Could not get a response from the model."
