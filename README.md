# LangChain Summarization ProjectA
Project to master LangChain through structured tasks. 
**Contributor:** 
- Muhammad Umair Shaikh


## Tasks Completed  



# RESULTS

# Task 1: Configuration and Environment Setup

Preparing the environment so that all later tasks could run smoothly with Azure OpenAI services. The key idea was to separate sensitive information such as API keys and deployment names from the code itself, and to organize reusable logic for future tasks.

- A .env file was created to store secrets like the Azure API key, endpoint, deployment name, and API version.

- These environment variables were then loaded into the project using a config loader built inside a new components folder.

- The components folder was introduced right from the start to hold reusable functions and modules, so that later tasks (summarizer, retriever, QA chain, etc.) could all share the same clean structure.


# Task 2: Building a Basic Summarization Chain

The second task introduced the first real use of LangChain’s LLMChain, focusing on summarization. 

✨ Approach

- Configured an Azure OpenAI model using environment variables set up in Task 1.

- Created a prompt template instructing the model to summarize any input text into exactly 3 sentences.

- Combined the model and prompt into an LLMChain, then tested it on a ~200-word paragraph about AI.

- Modified the prompt to limit the output to 1 sentence, allowing comparison between detailed and concise summarization.

📌 Results

***3-sentence summary:***
Artificial Intelligence (AI) simulates human intelligence in machines, enabling them to learn and solve problems like humans. It encompasses various subfields, leading to advancements in technologies such as voice assistants and autonomous vehicles, while also raising ethical concerns regarding bias, job displacement, and privacy. To address these issues, stakeholders are collaborating to create guidelines for the responsible development and use of AI.

***1-sentence summary:***
Artificial Intelligence (AI) simulates human intelligence in machines, enabling them to learn and solve problems, and while it has advanced various technologies, it also raises ethical concerns that necessitate responsible development and use.

🔎 Comparison

The 3-sentence version gave a richer breakdown, touching on AI’s capabilities, advancements, and ethical concerns separately.
The 1-sentence version compressed these points into a single line, which was concise but lost some clarity.


# Task 3: Exploring Retrievers with Summarization

This task combined the summarization chain from Task 2 with a document retriever. The goal was to break a long text file into chunks, embed them into a vector store, retrieve relevant pieces for a query, and then summarize them into a concise answer.

✨ Approach

- Created a 500-word text file (ai_intro.txt) describing the history of AI.

- Loaded the file with TextLoader, then split it into chunks of 200 characters with 20-character overlap using RecursiveCharacterTextSplitter.

- Built an in-memory FAISS vector store with Azure OpenAI embeddings.

- Queried the retriever with the prompt “AI milestones”, retrieving the most relevant chunks.

- Combined the retrieved chunks and passed them into the summarization chain (set to 2 sentences).

📌 Retrieval Output

✅ Total Chunks Created: 23

Top Retrieved Chunks (samples):

“countless business applications. Each milestone in AI’s history reflects not only technological progress but also humanity’s evolving relationship with intelligent machines…”

“AI systems to surpass human expertise in certain tasks. Progress in speech recognition, computer vision, and natural language processing during the early 2000s…”

“Researchers created early programs capable of tasks such as solving algebraic equations, playing checkers or chess, and proving logical theorems…”

## 📝 Final Summary

**The evolution of AI showcases significant technological advancements and humanity's changing interaction with intelligent machines, marked by milestones in areas like speech recognition and natural language processing. This journey reflects cycles of optimism and disillusionment, highlighting both early achievements and ongoing developments in AI capabilities.**

🔎 Observation

The retriever successfully narrowed down the document to the most relevant passages, and the summarizer captured both technological highlights and the cyclical nature of AI progress in just two sentences.


# Task 4: Creating an Agent for Summarization

This task introduced the concept of Agents in LangChain, which can dynamically decide which tools to use for a given query. Unlike fixed chains, an agent uses a reasoning process (thought → action → observation → final answer) to pick the right tool and handle both precise and vague queries.

✨ Approach

- Created a TextSummarizer tool from the summarizer chain built earlier.

- Integrated it into an AgentExecutor so the agent could decide when and how to call the summarizer.

- Tested the agent with:

- A clear and specific summarization query.

- A vague request to check whether the agent can interpret it and still provide a meaningful result.

📌 Results

Test 1: Clear Request – Impact of AI on Healthcare

Input: “Summarize the impact of AI on healthcare”

The agent identified the summarizer tool as needed, extracted key points, and provided:
Final Answer:
Artificial intelligence (AI) is revolutionizing healthcare by enhancing diagnosis, treatment, and patient outcomes through advanced machine learning models and imaging systems. These technologies enable earlier and more accurate disease detection, optimize hospital resource management, and offer support via virtual assistants and chatbots. Despite ethical concerns regarding data privacy and bias, the integration of AI in healthcare is expanding, leading to greater efficiency and improved patient care globally.

Test 2: Vague Request – “Summarize something interesting”

The agent interpreted the vague prompt, selected an example passage (discovery of penicillin), and summarized it.

Final Answer:
The discovery of penicillin by Alexander Fleming in 1928 was a pivotal moment in medical history, transforming the treatment of bacterial infections. It also laid the groundwork for the development of additional antibiotics. The profound impact of penicillin has saved countless lives and remains essential in modern medicine.

🔎 Observation

Test 1 confirmed that the agent works well with clear instructions.

Test 2 showed that the agent can handle vague instructions by reasoning its way to a meaningful task (choosing a passage and summarizing it).

**This demonstrated that agents provide flexibility and robustness compared to fixed chains.**


# Task 5: Combining Chains, Retrievers, and Agents

This task extended the use of agents by combining multiple tools to complete more complex tasks. Instead of relying on just summarization, the agent was given both a TextRetriever and TextSummarizer, and later an additional WordCounter tool. The goal was to test whether the agent could chain these tools together in the right order to produce meaningful results.

✨ Approach

- Built tools for retrieving relevant text chunks from the document, summarizing them, and counting words.

- Integrated all tools into an AgentExecutor, giving the agent freedom to decide which tool(s) to use depending on the query.

-Tested the agent with two scenarios:

1. Retrieve + Summarize a focused topic.

2. Retrieve + Summarize + Count Words, combining reasoning with multiple tool calls.

📌 Results

**Test 1: Retrieve + Summarize**

Query: “AI breakthroughs”

The agent first called TextRetriever to fetch relevant document chunks.

Then, it passed the retrieved text to the TextSummarizer tool.

Final Answer:
AI systems have advanced significantly since the late 1990s, particularly with deep learning breakthroughs in the 2010s, showcasing their ability to surpass human expertise in tasks like speech recognition, computer vision, and natural language processing.

**Test 2: Summarize and Count Words**

Query: “Summarize AI breakthroughs and count the words.”

The agent again called TextRetriever, then TextSummarizer, and finally used the WordCounter tool.

Final Answer:
The summary of AI breakthroughs is:
“AI has evolved significantly since the early 2000s, with breakthroughs in deep learning during the 2010s leading to superior performance in tasks like image recognition, highlighted by milestones such as IBM's Deep Blue defeating chess champion Garry Kasparov in 1997.”
Word count of summary: 40

🔎 Observation

- Test 1 showed that the agent could reason and use tools sequentially (retrieval → summarization).

- Test 2 demonstrated multi-tool chaining, where the agent not only summarized but also used the WordCounter tool to calculate word length.

- This highlighted the flexibility of agents in handling composite queries and producing structured outputs.

# Task 6: Using Memory to Improve Summarization

This task focused on experimenting with LangChain memory types to observe how they affect multi-turn interactions in summarization. Two kinds of memory were tested:

ConversationBufferMemory: Stores the full conversation history verbatim.

ConversationSummaryMemory: Stores a compressed summary of the conversation so far.

The aim was to check whether memory choice impacts summarization quality and continuity when handling related queries.

✨ Approach

- Implemented two summarizers, one with buffer memory and one with summary memory, inside the reusable summarizer component.

- Queried both with the same sequence of topics:

- Machine Learning, Deep Learning

- Compared whether memory type changed the flow, context retention, or final summaries.

📌 Results

🔹 Using ConversationBufferMemory

Machine Learning Summary:
Machine learning enables computers to learn from data and make autonomous predictions or decisions, relying on quality datasets and strong computation. Applications include spam detection, recommendations, fraud detection, and medical diagnosis.

Deep Learning Summary:
Deep learning, as a subset of ML, leverages multi-layered neural networks for complex data tasks like image recognition and natural language. It drives AI applications such as self-driving cars and medical image analysis, but requires massive labeled data and compute power.

🔹 Using ConversationSummaryMemory

Machine Learning Summary:
Machine learning, a subset of artificial intelligence, empowers computers to learn from data and make predictions or decisions autonomously by training algorithms on large datasets. Its effectiveness relies on data quality and computational resources, with applications spanning spam detection, recommendation systems, fraud detection, and medical diagnosis.

Deep Learning Summary:
Deep learning, a subset of machine learning, utilizes multi-layered artificial neural networks to effectively process complex data, excelling in tasks like image recognition and natural language understanding. While it drives significant advancements in AI, such as self-driving cars and medical image analysis, it demands substantial labeled data and computational resources to enhance performance.

🔎 Observation

- Buffer memory is more transparent, holding the raw dialogue, while summary memory becomes more useful as conversations grow longer, avoiding context overflow.

- This task demonstrated how memory influences multi-turn summarization, even though differences only emerge in extended conversations.

# Task 7: Leveraging Document Loaders for Diverse Sources

This task demonstrated how LangChain retrievers can be applied to different document sources — a local PDF file and an online webpage. The goal was to see how the retriever–summarizer pipeline behaves on structured academic text versus noisy web content.

📌 Results

🔹 PDF Retriever Results

Retrieved Text: Extracted coherent, domain-specific content on AI ethics.

Summary:
AI algorithms, while excelling in specific cognitive tasks traditionally performed by humans, do not introduce new ethical challenges despite their limitations in certain critical areas.

This shows that the retriever worked effectively on structured content, pulling out relevant ethical discussions.

🔹 Web Retriever Results

Retrieved Text: Instead of meaningful content, the retriever pulled in noisy material such as policies, copyright notices, and UI text from the site.

Summary:
The text outlines various policies and resources related to the Learning Lab, including user agreements, privacy, copyright, and customer support options.

This illustrates that web sources often require more aggressive cleaning and filtering, since raw HTML includes irrelevant metadata, menus, and disclaimers.

🔎 Observation

- The PDF retriever clearly outperformed the web retriever in this task, producing contextually relevant summaries aligned with the query.

- The web retriever struggled because of uncleaned boilerplate content, showing the importance of data preprocessing when working with live web sources.

- Overall, retrievers can be powerful, but content quality and formatting heavily impact results.


# Task 8: Customizing with Output Parsers

This task enhanced the retriever built earlier by extending it with an extra argument to support multi-query retrieval. Instead of fetching results from just one phrasing of the query, the retriever now generates and processes multiple variations of the query, ensuring more comprehensive coverage.
- The output was formatted in JSON, combining the summary with metadata for structured results.

✨ Approach

- Updated the retriever in the components folder to accept an extra argument for enabling multi-query retrieval.

- Retrieved information was passed to the summarizer component.

- Final response was returned as a JSON object containing both the summary and its metadata.

📌 Results

{
  "summary": "Artificial intelligence (AI) is a transformative technology impacting various industries, enhancing work, communication, and problem-solving. It offers significant benefits in healthcare, transportation, business, and education, while also raising concerns about privacy, bias, and job displacement. As AI adoption increases, it is essential to balance innovation with ethical governance to ensure its positive impact on society.",
  "length": "299"
}

🔎 Observation

By extending the retriever, the solution became flexible — the same retriever can handle both single-query and multi-query modes.

Multi-query mode improved coverage, pulling in richer context from the document store.

JSON output makes the result machine-readable, suitable for use in APIs or downstream apps.


# Task 9: Experimenting with Multi-Query Retrieval

This task focused on comparing the outputs of a Normal Retriever and an Extended Multi-Query Retriever (developed earlier by adding an extra argument). The goal was to evaluate how retrieval quality and coverage differ between the two approaches.

✨ Approach

- Used the Normal Retriever to fetch results from a single query.

- Used the Multi-Query Retriever with the same input, which expanded the query into multiple variations and retrieved a wider set of documents.

- Both results were passed into the summarizer component for comparison.

- The task highlighted how the multi-query strategy improves recall by covering different phrasings of the same query.

📌 Results

🔹 Normal Retriever:

✅ Total chunks created: 23

Retrieved 4 docs

Summary (Normal):
AI has evolved significantly since the early 2000s, becoming integral to various applications such as streaming recommendations, medical diagnostics, and autonomous vehicles, while reflecting both technological advancements and the changing dynamics of human interaction with intelligent machines.

🔹 Multi-Query Retriever:

✅ Total chunks created: 23

Retrieved 7 unique docs

Summary (Multi-Query):
The history of artificial intelligence, beginning in the mid-20th century with early explorations of machine mimicry of human intelligence, has evolved through significant technological milestones, particularly in the 2010s with deep learning advancements, leading to its pervasive integration in modern society across various applications.

🔎 Observation

- Normal Retriever was narrower in scope — focused mainly on modern applications and developments since the 2000s.

- Multi-Query Retriever provided broader coverage, pulling in historical context (1950s origins) alongside modern advancements.

- This demonstrates how the extra argument for multi-query retrieval increases both breadth and depth of retrieved information.


# Task 10: Building a Question-Answering Chain on Summaries

This task explored Question Answering (QA) using two different contexts:

A summarized version of the document.

The full original document.

The goal was to observe how context size impacts the quality and depth of the QA responses.

✨ Approach

- Implemented a QA system that could run queries against either:

1. The summary of the document (compressed information).

2. The full document (complete knowledge base).

- Asked the question: “What’s the key event mentioned?”

- Compared the reasoning and answers produced from both contexts.

📌 Results

🔹 QA on Summary

Answer: The Dartmouth Conference in 1956, considered the official start of AI as a field.

Rationale: Based on summarized context highlighting symbolic reasoning and problem-solving origins.

🔹 QA on Full Document

Answer: The Dartmouth Conference in 1956, regarded as the official birth of AI as a research field.

Rationale: Full context included more detail — such as the fact that it brought together pioneering computer scientists and coined the term “artificial intelligence”.

🔎 Observation

Both approaches identified the same key event: the Dartmouth Conference (1956).

Summary-based QA gave a concise but less detailed answer.

Full-document QA provided richer context, explaining why the event was pivotal and how it shaped AI’s origins.

**Conclusion:** Summaries make QA faster and focused, but full documents ensure completeness and richer detail.


# Task 11: Integrating External Tools with Agents

Extend the agent from Task 5 by:
- Adding a tool that fetches the current date using Python’s datetime module, Testing the agent and adding a second tool that performs a mock web search (returning a static 50-word response).

⚙️ Approach

- Extended tools.py:

1. Added create_date_tool() → gets today’s date from datetime.date.today().

2. Added create_mock_web_search_tool() → simulates a search by returning a static AI trends response.

**Results**

Test 1: Summarize + Date

Query:
“Summarize this 100-word text about AI and tell me today’s date.”

Agent Output:
Artificial intelligence (AI) has rapidly evolved into a transformative technology that allows machines to perform tasks requiring human-like intelligence across various industries, including healthcare and finance. Ongoing advancements in machine learning and neural networks enhance automation and efficiency, while ethical considerations such as fairness and accountability are crucial for its responsible development. As AI continues to advance, it presents both significant opportunities and challenges for the future of humanity. Today's date is 2025-09-11.

Test 2: Summarize + Search

Query:
“Summarize AI trends and search for recent updates.”

Agent Output:
Recent AI trends include generative AI, responsible AI practices, and edge AI for IoT devices, which are influencing business workflows. These advancements are expected to significantly impact various industries by 2025.

✅ Conclusion

- The agent was successfully extended with external tools while keeping agent.py reusable.

- The CurrentDate tool integrated real-time Python functionality.

- The MockWebSearch tool simulated external information access.

- Results show the agent can now combine summarization with external actions, making it more powerful and adaptable for future tasks.