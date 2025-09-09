# components/agents.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

def build_agent(deployment_name: str, tools: list[Tool]):
    """Build a zero-shot-react agent using the tool and Azure OpenAI model."""
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent
