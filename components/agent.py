# components/agent.py

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import AzureChatOpenAI
from components.summarizer import build_summarizer


def build_agent(api_key, endpoint, deployment, api_version, sentences=3):
    """
    Build a zero-shot-react-description agent with a custom TextSummarizer tool.
    """

    # Build summarizer chain (reusing Task 2)
    summarizer = build_summarizer(api_key, endpoint, deployment, api_version, sentences)

    # Wrap summarizer into a Tool
    summarizer_tool = Tool(
        name="TextSummarizer",
        func=lambda text: summarizer.invoke({"text": text}),
        description="Use this tool to summarize text into a concise format."
    )

    # LLM for the agent (same as Task 2 model)
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        openai_api_version=api_version,
        temperature=0
    )

    # Build the agent
    agent = initialize_agent(
        tools=[summarizer_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False  # shows reasoning steps
    )

    return agent
