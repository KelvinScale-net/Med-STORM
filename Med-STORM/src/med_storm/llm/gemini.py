import asyncio
import google.generativeai as genai
from typing import Optional
import os
import re
from google.api_core import exceptions as google_exceptions

from .base import LLMProvider
from config.settings import settings

class GeminiLLM(LLMProvider):
    """LLM Provider for Google's Gemini models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash-latest"):
        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("Google API key is not set. Please set it in your .env file as GOOGLE_API_KEY.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generates a text completion using the Gemini API with adaptive retries."""
        max_retries = 3
        # Initial wait to avoid hitting the limit in the first place
        await asyncio.sleep(15)

        for attempt in range(max_retries):
            try:
                # Note: The Gemini API has a different way of handling system prompts.
                # We are passing it as part of the content for now.
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = await self.model.generate_content_async(full_prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                print(f"Gemini API quota exceeded. Attempt {attempt + 1} of {max_retries}.")
                # Try to extract retry_delay from the error
                retry_delay_match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(e))
                if retry_delay_match:
                    delay = int(retry_delay_match.group(1))
                    print(f"Retrying after {delay} seconds as suggested by the API.")
                    await asyncio.sleep(delay)
                else:
                    # Fallback if retry_delay is not found
                    delay = (2 ** attempt) * 5 # Exponential backoff
                    print(f"No retry_delay found. Retrying in {delay} seconds (exponential backoff).")
                    await asyncio.sleep(delay)
            except Exception as e:
                # In a real app, you'd have more robust logging/error handling
                print(f"An unexpected error occurred while calling the Gemini API: {e}")
                return f"Error: {e}"
        
        print("Max retries reached. Failed to get a response from Gemini API.")
        return "Error: Max retries exceeded for Gemini API. Could not get a response from the model."
