import asyncio
from typing import Optional

from openai import AsyncOpenAI

from med_storm.llm.base import LLMProvider
from med_storm.config import settings

class DeepSeekLLM(LLMProvider):
    """
    A language model provider for the DeepSeek API.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates a response from the DeepSeek model.
        """
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
            print(f"An error occurred while generating response from DeepSeek: {e}")
            return f"Error: Could not get a response from DeepSeek. {e}"
