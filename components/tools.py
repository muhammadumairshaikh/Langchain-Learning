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