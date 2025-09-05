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