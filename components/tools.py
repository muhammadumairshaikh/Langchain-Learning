# # components/tools.py

# components/tool.py
from langchain.tools import Tool
from components.summarizer import build_summarizer
from components.retriever import build_retriever

def create_text_summarizer_tool(
    deployment_name: str,
    name: str = "TextSummarizer",
    sentences: int = 3,
    description: str | None = None,
) -> Tool:
    """a LangChain Tool that summarizes text via the Task 2 chain."""
    chain = build_summarizer(deployment_name, sentences=sentences)

    if description is None:
        description = (
            f"Summarizes input text into exactly {sentences} sentence(s). "
            "Input: plain text. Output: summary as a string."
        )

    def _summarize(text: str) -> str:
        result = chain.invoke({"text": text})
        return result.content  

    return Tool(name=name, func=_summarize, description=description)

