from med_storm.llm.base import LLMProvider

class QueryOptimizer:
    """
    Uses an LLM to refine search queries for specific medical knowledge bases.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def optimize_for_pubmed(self, question: str, context_topic: str) -> str:
        """
        Translates a natural language question into an expert-level PubMed query,
        using the context of the main topic.
        """
        system_prompt = (
            "You are a PubMed search expert. Your task is to convert a natural language question "
            "into a highly effective PubMed query. Use Medical Subject Headings (MeSH terms), "
            "boolean operators (AND, OR, NOT), and field tags (e.g., [Title/Abstract], [MeSH Terms]) "
            "to create a specific and efficient query. "
            f"The query MUST be focused on the context of: '{context_topic}'."
            "Provide only the final query string, without any explanations or introductory text."
        )

        prompt_with_context = f"Question: {question}\nContext Topic: {context_topic}"

        optimized_query = await self.llm.generate(
            prompt=prompt_with_context,
            system_prompt=system_prompt
        )
        
        # The LLM might add quotes around the query, let's strip them.
        return optimized_query.strip().strip('"')
