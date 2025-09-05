# Task-4 ScripT

from components.config_loader import load_config
from components.agent import build_agent

def main():
    config = load_config()

    # Build agent
    agent = build_agent(
        api_key=config["api_key"],
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        api_version=config["api_version"],
        sentences=3
    )

    # Clear query 
    healthcare_text = """
    Artificial intelligence (AI) is transforming healthcare by improving diagnosis, treatment, 
    and patient outcomes. Machine learning models are being trained on vast amounts of medical data, 
    enabling doctors to detect diseases earlier and more accurately. AI-powered imaging systems can 
    identify tumors or fractures with high precision, while predictive analytics help hospitals 
    manage patient admissions and resources more effectively. Virtual assistants and chatbots provide 
    basic medical advice and support, reducing strain on healthcare professionals. Although AI raises 
    ethical concerns about data privacy and bias, its integration into healthcare continues to grow, 
    promising improved efficiency and patient care worldwide.
    """

    print("\n🔹 Query 1: Summarize the impact of AI on healthcare")
    response1 = agent.invoke(f"Summarize the impact of AI on healthcare:\n\n{healthcare_text}")
    print("✅ Agent Response:\n", response1)

    # Example 2: Vague query
    print("\n🔹 Query 2: Summarize something interesting")
    response2 = agent.invoke("Summarize something interesting")
    print("✅ Agent Response:\n", response2)


if __name__ == "__main__":
    main()
