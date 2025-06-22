from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract Base Class for all Large Language Model providers."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generates a text completion based on a prompt."""
        pass
