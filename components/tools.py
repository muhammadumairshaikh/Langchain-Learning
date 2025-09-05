# components/tools.py

from langchain.agents import Tool
from components.summarizer import build_summarizer


def build_text_summarizer_tool(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
    name: str = "TextSummarizer",
    sentences: int = 3,
    description: str | None = None,
) -> Tool:
    """
    Create a LangChain Tool that wraps the summarization chain.
    This tool can be used by an Agent to summarize text.
    """

    # Build summarizer chain (from Task 2)
    chain = build_summarizer(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
        sentences=sentences,
    )

    # Default description if not provided
    if description is None:
        description = (
            f"Summarizes input text into exactly {sentences} sentence(s). "
            "Input: plain text. Output: summary string."
        )

    # Internal wrapper function for the tool
    def _summarize(text: str) -> str:
        result = chain.invoke({"text": text})
        return result["text"] if isinstance(result, dict) else result

    return Tool(name=name, func=_summarize, description=description)
