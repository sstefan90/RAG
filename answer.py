import numpy as np
from vector_search import VectorSearch
import torch
import torch.nn.functional as f
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import dotenv
import os
from dotenv import load_dotenv

MODEL_ID = "google/gemma-2b-it"
load_dotenv()

def main(query):

    #initialize our vector search and create embeddings for the query
    vs = VectorSearch()
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.reshape(1, 768)
    query_embedding = f.normalize(query_embedding, p=2, dim=0)
    result = vs.find_top_N(query_embedding, 5, verbose=True)
    additional_context = "/n".join(result)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID, access_token=os.getenv("HF_TOKEN"))

    # load the LLM!
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_ID, 
                                                    torch_dtype=torch.float32, # datatype to use, we want float16
                                                    low_cpu_mem_usage=False)
    input_text = query

    #Add additional context to the input text
    input_text = input_text + ". Use the following content below to augment your answer. " + additional_context

    # Make the prompt
    dialogue_template = [
        {"role": "user",
        "content": input_text},
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                        tokenize=False,
                                        add_generation_prompt=True)
    print(f"\nPrompt (formatted):\n{prompt}")

    #inference our LLM
    tokenized_input = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**tokenized_input,
                             max_new_tokens=256)

    # Decode the generated output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    query = "What are healthy food groups?"
    print(main(query))