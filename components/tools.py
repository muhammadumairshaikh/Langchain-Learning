# # components/tools.py

from langchain.agents import Tool
from components.summarizer import build_summarizer
from components.retriever import build_retriever


def build_text_summarizer_tool(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
    sentences: int = 3,
    name: str = "TextSummarizer",
    description: str | None = None,
) -> Tool:
    """Return a Tool that summarizes text via the Task 2 chain."""
    summarizer = build_summarizer(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
        sentences=sentences,
    )

    if description is None:
        description = (
            f"Summarizes input text into exactly {sentences} sentence(s). "
            "Input: plain text. Output: summary as a string."
        )

    def _summarize(text: str) -> str:
        return summarizer.invoke({"text": text})

    return Tool(name=name, func=_summarize, description=description)


def build_retriever_tool(
    file_path: str,
    api_key: str,
    endpoint: str,
    embedding_deployment: str,
    api_version: str,
    name: str = "TextRetriever",
    description: str = "Retrieve relevant text chunks from the AI intro document based on a query.",
) -> Tool:
    """Return a Tool that retrieves relevant chunks from a document."""
    retriever, _ = build_retriever(
        file_path=file_path,
        api_key=api_key,
        endpoint=endpoint,
        embedding_deployment=embedding_deployment,
        api_version=api_version,
    )

    def _retrieve(query: str) -> str:
        docs = retriever.invoke(query)
        return " ".join([doc.page_content for doc in docs])

    return Tool(name=name, func=_retrieve, description=description)


def build_word_count_tool(
    name: str = "WordCounter",
    description: str = "Counts the number of words in the input text.",
) -> Tool:
    """Return a simple Tool that counts words in the given text."""

    if description is None:
        description = "Counts the number of words in the given text string."

    def _count_words(text: str) -> str:
        return f"Word count: {len(text.split())}"

    return Tool(name=name, func=_count_words, description=description)
