# Task 5 Script

# task_5/task5.py

from components.config_loader import load_config
from components.agent import build_agent
from components.tools import (
    build_text_summarizer_tool,
    build_retriever_tool,
    build_word_count_tool,
)
from langchain_openai import AzureChatOpenAI


def main():
    # Load config
    config = load_config()

    # Build the LLM directly here
    llm = AzureChatOpenAI(
        openai_api_key=config["api_key"],
        azure_endpoint=config["endpoint"],
        deployment_name=config["deployment"],
        openai_api_version=config["api_version"],
        temperature=0,
    )

    # Build tools
    retriever_tool = build_retriever_tool(
        file_path="task_3/ai_intro.txt",  # reuse the same file from Task 3
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        embedding_deployment="text-embedding-3-small",
        api_version=config["api_version"],
    )

    summarizer_tool = build_text_summarizer_tool(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        api_version=config["api_version"],
        sentences=3,
    )

    word_count_tool = build_word_count_tool()

    # Build the agent with all tools
    agent = build_agent(
        llm=llm,
        tools=[retriever_tool, summarizer_tool, word_count_tool],
    )

    # Query 1: Find and summarize AI breakthroughs
    query1 = "Find and summarize text about AI breakthroughs from the document."
    print("\n🔹 Query 1:", query1)
    response1 = agent.invoke(query1)
    print("\n✅ Agent Response:\n", response1["output"])

    # Query 2: Also include word count
    query2 = "Find and summarize text about AI breakthroughs from the document. Then count the words in the summary."
    print("\n🔹 Query 2:", query2)
    response2 = agent.invoke(query2)
    print("\n✅ Agent Response:\n", response2["output"])


if __name__ == "__main__":
    main()
