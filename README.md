# Generative-AI-Nanodegree
This repo contains a collection of project deliverables from my [Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608) course work.

## Educational Objectives
A graduate of this program will be able to:
  - Situate generative AI within the broader history, context, and
  applications of artificial intelligence and deep learning
  - Adapt generative foundation models to perform tasks in
  novel contexts
  - Use LLMs and prompt engineering to create a custom
  chatbot
  - Use image generation models such as Stable Diffusion to
  perform image inpainting
  - Build applications that use LLMs, implement semantic search
  with vector databases, and apply retrieval augmented
  generation techniques

## Projects
1. [Apply Lightweight Fine-Tuning to a Foundation Model](https://github.com/amanpreetsingh459/Generative-AI/tree/main/1.%20Generative%20AI%20Fundamentals/Project1-Lightweight%20Fine-Tuning(PEFT))
    1. Load a foundation model
    2. Identify and load a Hugging Face dataset for your particular task
    3. Utilize a state-of-the-art technique to adjust the foundation model's weights to meet the needs of your task, using a lightweight (AKA parameter-efficient) fine-tuning technique that improves performance faster and more efficiently

2. [Retrieval Augmented Generation(RAG)](https://github.com/amanpreetsingh459/Generative-AI/tree/main/2.%20LLMs%20%26%20Text%20Generation/Project2-Retrieval-Augmented-Generation(RAG))
    1. Find and prepare a dataset that augments a foundation model's knowledge, from a source such as APIs, web scraping, or documents on hand
    2. Create a semantic search pipeline by implementing a custom Python vector similarity search algorithm to match user questions to relevant parts of the custom dataset
    3. Compose a custom query by combining the semantic search results with the user's question and send it to the foundation model

3. [AI Photo Editing with Inpainting](https://github.com/amanpreetsingh459/Generative-AI/tree/main/3.%20Computer%20Vision%20and%20Generative%20AI/Project3-AI-Photo-Editing-with-Inpainting)
    1. Create a segmentation mask by differentiating between the subject and background of an image and create a matrix of pixels indicating the locations of these two components
    2. Given a text prompt and the pixel locations of the subject or background, replace part of the image with an AIgenerated image
    3. Connect your inpainting pipeline to a web interface that allows users to upload their own images and specify their own text prompts udacity

4. [Personalized Real Estate Agent](https://github.com/amanpreetsingh459/Generative-AI/tree/main/4.%20Building%20Generative%20AI%20Solutions/Project4-Personalized-Real-Estate-Agent)
    1. Generate synthetic data using LLMs
    2. Embed property listing data in a vector database
    3. Perform semantic search over property listings against user preferences
    4. Design prompts and use RAG techniques to deliver personalized recommendations
  
## Dependencies
[requirements.txt](requirements.txt)

## License
[License](LICENSE.txt)

## Credits
- [udacity.com](udacity.com)
- [CODEOWNERS](CODEOWNERS)
