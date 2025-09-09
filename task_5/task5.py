from components.config_loader import load_config
from components.tools import (
    create_text_summarizer_tool,
    create_retriever_tool,
    create_word_count_tool,
)
from components.agent import build_agent

def main():
    config = load_config()

    # --- Build tools ---
    retriever_tool = create_retriever_tool(
        file_path="task_3/ai_intro.txt",
        embedding_deployment=config["embedding_deployment"]
    )
    summarizer_tool = create_text_summarizer_tool(
        deployment_name=config["chat_deployment"],
        sentences=1
    )
    word_count_tool = create_word_count_tool()

    tools = [retriever_tool, summarizer_tool, word_count_tool]

    # Build agent
    agent = build_agent(config["chat_deployment"], tools)

    # --- Test 1: AI breakthroughs ---
    print("\n--- Test 1: Retrieve + Summarize ---")
    query1 = "Find and summarize text about AI breakthroughs from the document."
    response1 = agent.invoke(query1)
    print("📝 Response:", response1["output"])

    # --- Test 2: Summarize + count words ---
    print("\n--- Test 2: Summarize and count words ---")
    query2 = "Summarize AI breakthroughs and then count the words in the summary."
    response2 = agent.invoke(query2)
    print("📝 Response:", response2["output"])


if __name__ == "__main__":
    main()