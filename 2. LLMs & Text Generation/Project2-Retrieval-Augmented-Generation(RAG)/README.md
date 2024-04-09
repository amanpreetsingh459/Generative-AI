# Project: Build Your Own Custom Chatbot using RAG

In this project we use a custom dataset and use it with the OpenAI LLM gpt-3.5-turbo-instruct model.
We use Retrieval Augmented Generation(RAG) technique to let the model incorporate a custom dataset knowledge into its responses

### 1. Select a data source
`2023_fashion_trends.csv` - This file contains reports and quotes about fashion trends for 2023. Each row includes the source URL, article title, and text snippet.

### 2. Data Wrangling
Project dataset is loaded into a pandas dataframe containing at least 20 rows. Each row in the dataset contains a text sample in a column named "text"

### 3. Custom Query Completion
The project successfully sends a custom query with information from the project dataset to the OpenAI model and gets a response
- Generating Embeddings
- Create a Function that Finds Related Pieces of Text for a Given Question
- Create a Function that Composes a Text Prompt by incorporating the context from the custom dataset embeddings

### 4. Custom Performance Demonstration
- Create a Function that Answers a Question by making call to OpenAI model
- Comparisons of answers with and without context

### 5. Further improvements
1. Try using Vector databases (Open/Closed) for the embeddings purpose
2. Try using an open-weights LLM  for the task like [Gemma](https://huggingface.co/google/gemma-7b), [LLAMA2](https://huggingface.co/meta-llama/Llama-2-7b), [Mistral etc](https://huggingface.co/mistralai/Mistral-7B-v0.1).


## References
- Dataset - 2023_fashion_trends.csv: [udacity.com](https://www.udacity.com/course/generative-ai--nd608)