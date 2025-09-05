# -*- coding: utf-8 -*-
# task_4/task4.py

from components.config_loader import load_config
from components.agent import build_summarization_agent


def main():
    # Load credentials
    config = load_config()

    # Build the agent
    agent = build_summarization_agent(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        api_version=config["api_version"],
        sentences=3,
    )

    # Query 1: Healthcare text
    healthcare_text = """
    Artificial intelligence (AI) is transforming healthcare by improving diagnosis,
    treatment, and patient outcomes. Machine learning models are being trained on vast
    amounts of medical data, enabling doctors to detect diseases earlier and more
    accurately. AI-powered imaging systems can identify tumors or fractures with high
    precision, while predictive analytics help hospitals manage patient admissions and
    resources more effectively. Virtual assistants and chatbots provide basic medical
    advice and support, reducing strain on healthcare professionals. Although AI raises
    ethical concerns about data privacy and bias, its integration into healthcare
    continues to grow, promising improved efficiency and patient care worldwide.
    """
    print("\n🔹 Query 1: Summarize the impact of AI on healthcare")
    response1 = agent.run(f"Summarize the impact of AI on healthcare:\n\n{healthcare_text}")
    print("\n✅ Agent Response:\n", response1)

    # Query 2: Vague request
    print("\n🔹 Query 2: Summarize something interesting")
    response2 = agent.run("Summarize something interesting")
    print("\n✅ Agent Response:\n", response2)


if __name__ == "__main__":
    main()
