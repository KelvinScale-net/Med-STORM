from typing import Dict

from med_storm.llm.base import LLMProvider

class ReportGenerator:
    """Generates a full report chapter from synthesized evidence sections."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def generate_chapter(self, sub_topic: str, reports: Dict[str, str]) -> str:
        """
        Generates a single, coherent chapter for a sub-topic by weaving together
        individual synthesized reports.
        """
        if not reports:
            return f"No reports were provided to generate a chapter for the topic: {sub_topic}."

        # Format the individual reports into a single block for the prompt
        formatted_reports = ""
        for question, report in reports.items():
            formatted_reports += f"### Question: {question}\n{report}\n\n---\n\n"

        system_prompt = '''You are a senior medical writer and editor. Your task is to synthesize a collection of individual Q&A-style reports into a single, cohesive, and well-structured chapter for a medical review article. The chapter should have a clear narrative flow.

Your output MUST include:

1. A brief, engaging introduction that sets the context for the chapter's topic.

2. A main body that logically connects the information from the provided reports. You must re-organize and re-phrase the content to avoid a simple Q&A format. Create a natural, flowing narrative.

3. A concise conclusion that summarizes the key takeaways of the chapter.

Maintain all original citations (e.g., [1], [2]) exactly as they appear in the source reports.

4. A dedicated section at the end titled 'Gaps in Evidence and Future Research Directions'. In this section, you must list the questions for which the reports explicitly state 'No evidence found to synthesize'. This highlights areas where research is lacking.'''

        user_prompt = f"""Please generate a comprehensive chapter for the following topic: '{sub_topic}'

Use the following synthesized reports as your source material:

---

{formatted_reports}"""

        final_chapter = await self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
        return final_chapter
