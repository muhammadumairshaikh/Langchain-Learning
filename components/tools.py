# # components/tools.py

from langchain.tools import Tool
from components.summarizer import build_summarizer
from components.retriever import build_retriever
from datetime import date

def create_text_summarizer_tool(
    deployment_name: str,
    name: str = "TextSummarizer",
    sentences: int = 3,
    description: str | None = None,
) -> Tool:
    """Summarizer tool wrapping Task 2 chain."""
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


def create_retriever_tool(
    file_path: str,
    embedding_deployment: str,
    name: str = "TextRetriever",
    description: str | None = None,
) -> Tool:
    """Retriever tool wrapping Task 3 retriever."""
    retriever = build_retriever(file_path, embedding_deployment)

    if description is None:
        description = "Retrieve relevant text chunks from the document based on a query."

    def _retrieve(query: str) -> str:
        docs = retriever.invoke(query)  # modern API
        return "\n".join(doc.page_content for doc in docs)

    return Tool(name=name, func=_retrieve, description=description)


def create_word_count_tool(
    name: str = "WordCounter",
    description: str = "Counts the number of words in the given text.",
) -> Tool:
    """Simple word counting tool."""
    def _count_words(text: str) -> str:
        return f"Word count: {len(text.split())}"

    return Tool(name=name, func=_count_words, description=description)


def create_current_date_tool(
    name: str = "CurrentDate",
    description: str = "Fetches today's current date in YYYY-MM-DD format."
) -> Tool:
    """Tool to return today's date."""
    def _get_date(_: str = "") -> str:  
        return date.today().isoformat()

    return Tool(name=name, func=_get_date, description=description)

def create_mock_web_search_tool(
    name: str = "MockWebSearch",
    description: str = "Performs a fake web search and returns a static 50-word response."
) -> Tool:
    """Tool to simulate a web search by returning static content."""
    def _search(query: str) -> str:
        return (
            "This is a mock web search result. Artificial Intelligence (AI) is rapidly evolving, "
            "with major trends including generative AI, responsible AI practices, "
            "edge AI for IoT devices, and integration into business workflows. "
            "These updates reflect how AI is shaping industries in 2025."
        )
    return Tool(name=name, func=_search, description=description)