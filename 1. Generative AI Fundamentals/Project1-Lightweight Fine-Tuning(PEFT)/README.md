# Project: Apply Lightweight Fine-Tuning to a Foundation Model

## 1. Prepare the Foundation Model
1. **Load a pretrained HF model:** Includes the relevant imports and loads a pretrained Hugging Face model that can be used for sequence classification

2. **Load and preprocess a dataset:** Includes the relevant imports and loads a Hugging Face dataset that can be used for sequence classification. Then includes relevant imports and loads a Hugging Face tokenizer that can be used to prepare the dataset. A subset of the full dataset may be used to reduce computational resources needed.

3. **Evaluate the pretrained model:** At least one classification metric is calculated using the dataset and pretrained model

## 2. Perform Lightweight Fine-Tuning
1. **Create a PEFT model:** Includes the relevant imports, initializes a Hugging Face PEFT config, and creates a PEFT model using that config

2. **Train the PEFT model:** The model is trained for at least one epoch using the PEFT model and dataset

3. **Save the PEFT model:** Fine-tuned parameters are saved to a separate directory. The saved weights directory should be in the same home directory as the notebook file.

## 3. Perform Inference Using the Fine-Tuned Model
1. **Load the saved PEFT model:** Includes the relevant imports then loads the saved PEFT model

2. **Evaluate the fine-tuned model:** Repeats the earlier evaluation process (same metric(s) and dataset) to compare the fine-tuned version to the original version of the model

## 4. Further improvements
1. Try using the bitsandbytes package (installed in the workspace) to combine quantization and LoRA. This is also known as QLoRA
2. Try training the model using different PEFT configurations and compare the results
3. Try training another popular LLM like [Gemma](https://huggingface.co/google/gemma-7b), [LLAMA2](https://huggingface.co/meta-llama/Llama-2-7b), [Mistral etc](https://huggingface.co/mistralai/Mistral-7B-v0.1).