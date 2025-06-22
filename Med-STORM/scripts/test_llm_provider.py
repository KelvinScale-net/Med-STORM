import asyncio
import os
import sys

# Adjust path to import from the project's source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from med_storm.llm.openai import OpenAILLM
from config.settings import settings

async def main():
    """Main function to test the OpenAILLM provider."""
    print("Testing OpenAI LLM Provider...")

    if not settings.openai_api_key:
        print("\nERROR: OPENAI_API_KEY is not set in the .env file.")
        print("Please create a .env file in the project root and add your key.")
        print("Example .env content:")
        print('OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
        return

    try:
        llm = OpenAILLM()
        system_prompt = "You are a helpful assistant. Be concise."
        prompt = "Explain the concept of a 'p-value' in statistics in one sentence."

        print(f'\nSystem Prompt: "{system_prompt}"')
        print(f'User Prompt: "{prompt}"')
        print("\nWaiting for response from OpenAI...")

        response = await llm.generate(prompt, system_prompt=system_prompt)

        print("-" * 50)
        print(f"Model Response:\n{response}")
        print("-" * 50)
        print("\nLLM Provider test successful!")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")

if __name__ == "__main__":
    # This script requires the OPENAI_API_KEY to be set in a .env file
    # in the root of the project directory.
    asyncio.run(main())
