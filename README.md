# LangChain Summarization ProjectA 
project to master LangChain through structured tasks. 
Contributor: - Muhammad Umair Shaikh


## Tasks Completed  

### Task 1: Setting Up LangChain Environment  
- Created a Python virtual environment.  
- Installed required packages: `langchain`, `openai`, `python-dotenv`.   


### Task 2: Building a Basic Summarization Chain  
- Configured **Azure OpenAI model** using environment variables.  
- Designed a **prompt template** to summarize text into **exactly 3 sentences**.  
- Built an `LLMChain` that combined the prompt with the model.  
- Tested it with a **200-word paragraph about Artificial Intelligence**.  
- Modified the prompt to summarize into **1 sentence** and compared outputs.  


### Task 3: Exploring Retrievers with Summarization
- Chunks and Embeddings Vectors
![Result](results/task3.1.png)

- Summary
![Summary](results/task3.2.png)


### Task 4: Creating an Agent for Summarization
- A LangChain Agent powered by a custom tool called TextSummarizer, which wraps the summarization chain from Task 2. The agent was tested with both a clear request and a vague request to demonstrate how it interprets and executes summarization tasks.

🔹 Query 1: Summarize the impact of AI on healthcare

Result:
Artificial intelligence (AI) is revolutionizing healthcare by enhancing diagnosis, treatment, and patient outcomes through advanced machine learning models and imaging systems. These technologies enable earlier and more accurate disease detection, optimize hospital resource management, and offer support via virtual assistants and chatbots. Despite ethical concerns regarding data privacy and bias, the integration of AI in healthcare is expanding, leading to greater efficiency and improved patient care globally.

🔹 Query 2: Summarize something interesting

Result:
The discovery of penicillin by Alexander Fleming in 1928 was a pivotal moment in medical history, revolutionizing the treatment of bacterial infections. This breakthrough led to the development of other antibiotics, significantly impacting healthcare practices. Fleming's contributions have saved countless lives and remain influential in modern medicine.



### Task 5: Combining Chains, Retrievers, and Agents

In this task, we built a multi-tool agent pipeline that integrates:

Retriever Tool – extracts relevant chunks of text from a document (Task 3).

Summarizer Tool – summarizes retrieved text into concise sentences (Task 2).

Word Counter Tool – counts words in the summary output.

This allows the agent to retrieve, summarize, and analyze text in a single workflow.

🔹 Query 1: Find and summarize text about AI breakthroughs from the document

Agent Response:
The 2010s marked significant advancements in AI, particularly with deep learning, as powerful GPUs and large datasets enabled deep neural networks to excel in tasks like image recognition and language modeling. Notable achievements included Google DeepMind's AlphaGo defeating the world champion Go player in 2016 and the emergence of generative models like GPT, which produced human-like text. Today, AI is integral to various aspects of society, influencing everything from recommendation systems to medical diagnostics, while its history reflects a cycle of optimism and renewed discovery in the field.

🔹 Query 2: Find and summarize text about AI breakthroughs, then count words

Summary:
The 2010s marked a significant turning point for AI with the advent of deep learning, leading to remarkable achievements such as AlphaGo's victory over the world champion Go player in 2016 and the emergence of generative models like GPT. Despite earlier setbacks, advancements in computing power and data availability revitalized AI research, with milestones like IBM's Deep Blue defeating Garry Kasparov in 1997 showcasing AI's potential to exceed human capabilities in specific tasks. Today, AI is integral to various aspects of society, influencing everything from recommendation systems to medical diagnostics, while reflecting the ongoing evolution of our relationship with intelligent machines.

Word Count: 101

✅ Outcome: Successfully demonstrated an agent workflow combining retrieval + summarization + word count into a seamless process.

- This shows how an agent can think autonomously and perform right set of actions.



### Task 6: Using Memory to Improve Summarization
🔹 Objective

We compared two memory types in LangChain:
ConversationBufferMemory → Stores the last N interactions verbatim. -> we kept last 3 records.
ConversationSummaryMemory → Stores a compressed summary of past interactions.


🔹 Input Texts

Machine Learning (Input):
Machine learning is a branch of artificial intelligence that focuses on enabling computers to learn patterns from data and make predictions or decisions without being explicitly programmed. It involves training algorithms on large datasets to identify trends, classify information, or forecast outcomes. Common applications include spam detection, recommendation systems, fraud detection, and medical diagnosis. Machine learning models can be supervised, unsupervised, or reinforcement-based depending on the type of data and problem. The success of machine learning largely depends on data quality and computational resources, making it a driving force behind modern AI advancements.

Deep Learning (Input):
Deep learning is a specialized area of machine learning that uses artificial neural networks with multiple layers to process complex data. It excels at handling tasks like image recognition, speech processing, and natural language understanding. By mimicking how the human brain processes information, deep learning can capture intricate patterns in data, leading to breakthroughs such as self-driving cars, advanced chatbots, and medical image analysis. However, deep learning requires massive amounts of labeled data and high computational power. Its ability to improve performance with more data makes it one of the most influential technologies in modern AI development.

🔹 Results
🔹 Using ConversationBufferMemory

Machine Learning Summary:
Machine learning is a subset of artificial intelligence that allows computers to learn from data and make predictions or decisions autonomously. Its effectiveness relies on the quality of data and computational resources, with applications in areas such as spam detection, recommendation systems, and medical diagnosis.


Deep Learning Summary:
Deep learning is a branch of machine learning that utilizes multi-layered artificial neural networks to analyze complex data, excelling in tasks like image recognition and natural language understanding. It demands significant labeled data and computational resources, but its capacity to enhance performance with increased data has made it a pivotal technology in AI advancements.


🔹 Using ConversationSummaryMemory

Machine Learning Summary:
Machine learning is a subset of artificial intelligence that allows computers to learn from data and make predictions or decisions autonomously. Its effectiveness relies on the quality of data and computational resources, with applications in areas such as spam detection, recommendation systems, and medical diagnosis.


Deep Learning Summary:
Deep learning is a branch of machine learning that utilizes multi-layered artificial neural networks to analyze complex data, excelling in tasks like image recognition and natural language understanding. It demands substantial labeled data and computational resources, but its capacity to enhance performance with increased data has made it a pivotal technology in AI advancements.


🔹 Key Insight

BufferMemory → keeps full raw history, so summaries are detailed but sometimes verbose.

SummaryMemory → keeps a compressed version of history, making outputs shorter and more focused.

✅ Purpose: This task shows how memory changes summarization across related inputs and helps decide whether an app should prioritize detail (BufferMemory) or conciseness (SummaryMemory)