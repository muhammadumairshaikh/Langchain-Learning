# Task 11 Script

# task_11/task11.py
from components.config_loader import load_config
from components.agent import build_agent
from components.tools import (
    create_text_summarizer_tool,
    create_current_date_tool,
    create_mock_web_search_tool,
)

def main():
    config = load_config()

    # Build tools
    summarizer_tool = create_text_summarizer_tool(config["chat_deployment"], sentences=3)
    date_tool = create_current_date_tool()
    search_tool = create_mock_web_search_tool()

    # Create agent with all tools
    tools = [summarizer_tool, date_tool, search_tool]
    agent = build_agent(config["chat_deployment"], tools)

    # --- Test 1 ---
    print("\n=== Test 1: Summarize + Date ===")
    text = (
        "Artificial intelligence (AI) is a transformative technology that has rapidly "
        "evolved over the last few decades. It enables machines to perform tasks "
        "requiring human-like intelligence, such as problem-solving, decision-making, "
        "and natural language understanding. AI applications span industries like "
        "healthcare, finance, transportation, and education. With ongoing advances in "
        "machine learning and neural networks, AI continues to push the boundaries of "
        "automation, personalization, and efficiency. Ethical considerations around AI "
        "use, including fairness, transparency, and accountability, remain critical for "
        "its responsible development. AI represents both opportunities and challenges "
        "for the future of humanity."
    )

    query1 = f"Summarize this text: {text} And also tell me today's date."
    result1 = agent.invoke(query1)
    print(result1["output"])


    # --- Test 2 ---
    print("\n=== Test 2: Summarize + Search ===")
    query2 = "Summarize AI trends and search for recent updates."
    result2 = agent.invoke(query2)
    print(result2["output"])

if __name__ == "__main__":
    main()
