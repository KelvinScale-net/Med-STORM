import os
import hashlib
import pickle
from .base import LLMProvider

# Setup a cache directory
CACHE_DIR = "./.llm_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class CachingLLM(LLMProvider):
    """
    A wrapper for LLM providers that adds a file-based caching layer.
    This avoids issues with pickling complex objects like LLM clients.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def _get_cache_key(self, prompt: str, system_prompt: str) -> str:
        """
        Creates a unique cache key based on the LLM provider and the prompts.
        """
        hasher = hashlib.md5()
        hasher.update(self.llm_provider.__class__.__name__.encode('utf-8'))
        hasher.update(prompt.encode('utf-8'))
        if system_prompt:
            hasher.update(system_prompt.encode('utf-8'))
        return hasher.hexdigest()

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates a response, using a cached result if available.
        """
        cache_key = self._get_cache_key(prompt, system_prompt)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"--- Loading response from cache for key: {cache_key[:7]}... ---")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"--- Calling LLM and caching response for key: {cache_key[:7]}... ---")
        result = await self.llm_provider.generate(prompt, system_prompt)

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result
