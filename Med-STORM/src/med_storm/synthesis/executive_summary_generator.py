from typing import Dict

from med_storm.llm.base import LLMProvider

class ExecutiveSummaryGenerator:
    """Generates an executive summary from a collection of report chapters."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def generate_summary(self, topic: str, chapters: Dict[str, str]) -> str:
        """
        Generates a high-level executive summary from the generated chapters.
        """
        if not chapters:
            return "No chapters were provided to generate an executive summary."

        formatted_chapters = ""
        for sub_topic, chapter_content in chapters.items():
            formatted_chapters += f"## Chapter: {sub_topic}\n\n{chapter_content}\n\n---\n\n"

        system_prompt = '''You are a distinguished medical editor and chief science officer.
Your task is to write a high-level executive summary for a comprehensive clinical review article on the provided topic.
The summary should be concise, impactful, and accessible to a clinical audience.

Your output MUST include:
1.  **Introduction**: A brief overview of the topic and the report's purpose.
2.  **Key Findings**: A synthesis of the most critical findings and conclusions from the chapters.
3.  **Consolidated Gaps in Evidence**: A summary of the most significant research gaps identified across all chapters.
4.  **Overall Conclusion**: A concluding statement on the clinical implications of the findings.

Do NOT include citations in the executive summary. It should be a standalone piece.'''

        user_prompt = f"""Please generate an executive summary for a clinical review on: '{topic}'

Use the following chapters as your source material:

---

{formatted_chapters}"""

        summary = await self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
        return summary
