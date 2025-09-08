from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


def build_summarizer_with_buffer_memory(api_key, endpoint, deployment, api_version):
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        openai_api_version=api_version,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Summarize the following text in **no more than 3 concise sentences**:
{text}
""",
    )

    memory = ConversationBufferMemory(memory_key="chat_history", k=3, return_messages=True)

    return LLMChain(llm=llm, prompt=prompt, memory=memory)


def build_summarizer_with_summary_memory(api_key, endpoint, deployment, api_version):
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        openai_api_version=api_version,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Summarize the following text in **no more than 3 concise sentences**:
{text}
""",
    )

    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    return LLMChain(llm=llm, prompt=prompt, memory=memory)
