# task_4/task4.py

from components.config_loader import load_config
from components.tools import create_text_summarizer_tool
from components.agent import build_agent


def main():
    # Load config from .env
    config = load_config()

    # Create summarizer tool 
    summarizer_tool = create_text_summarizer_tool(
        deployment_name=config["chat_deployment"],
        sentences=3
    )

    # Build agent with the summarizer tool
    agent = build_agent(config["chat_deployment"], [summarizer_tool])

    # Test 1: Clear instruction
    text_healthcare = (
    """
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
    )

    print("\n--- Test 1: Summarize the impact of AI on healthcare ---")
    response1 = agent.invoke(f"Summarize the impact of AI on healthcare: {text_healthcare}")
    print("📝 Response:", response1["output"])

    # Test 2: Vague request
    print("\n--- Test 2: Summarize something interesting ---")
    response2 = agent.invoke("Summarize something interesting")
    print("📝 Response:", response2["output"])


if __name__ == "__main__":
    main()

