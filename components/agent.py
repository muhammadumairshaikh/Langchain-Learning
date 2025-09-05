# components/agents.py

from langchain.agents import initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from components.tools import build_text_summarizer_tool


def build_summarization_agent(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
    temperature: float = 0,
    sentences: int = 3,
):
    """
    Build a Zero-Shot-React agent with a TextSummarizer tool.
    """

    # LLM for reasoning & agent orchestration
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        openai_api_version=api_version,
        temperature=temperature,
    )

    # Add the summarizer tool
    summarizer_tool = build_text_summarizer_tool(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
        sentences=sentences,
    )

    # Initialize the agent
    agent = initialize_agent(
        tools=[summarizer_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # see reasoning steps
    )

    return agent
