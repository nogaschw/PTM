import gc
import torch
import pickle
import pandas as pd
from Config import Config 
from itertools import chain
from transformers import AutoTokenizer, AutoModel

def process_in_batches(code_model, code_tokenizer, text_list, batch_size, device):
    """
    Processes a list of text inputs in batches and generates embeddings using a pre-trained model.

    Args:
        code_model (torch.nn.Module):  A pre-trained language model used to generate embeddings.
        code_tokenizer (Tokenizer):    A tokenizer that converts text inputs into tokenized format for `code_model`.
        text_list (list of str):       A list of text inputs to be processed.
        batch_size (int):              Number of text samples to process in each batch.
        device (str):                  The device on which computation is performed (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary where keys are text samples from `text_list` and values are the corresponding embeddings (lists of floats).
    """
    embedding_dict = {}
    
    for i in range(0, len(text_list), batch_size):
        if i + batch_size < len(text_list):
            batch_text = text_list[i:i + batch_size]
        else:
            batch_text = text_list[i:]
        if i % 1000 == 0:
            print(f"batch {i} / {len(text_list)}", flush=True)

        # Tokenize the batch
        encoding = code_tokenizer(batch_text, max_length=config.max_model_len, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            code_output = code_model(input_ids, attention_mask).last_hidden_state[:, 0, :]

        # Collect rows
        for j, coding in enumerate(batch_text):
            embedding_dict[coding] = code_output[j].tolist()

        # Clear memory
        del input_ids, attention_mask, code_output
        torch.cuda.empty_cache()
        gc.collect()
            
    return embedding_dict

# Initialize configuration
config = Config()

# Load preprocessed DataFrame
df = pd.read_pickle(config.path_tosave_codeworkout)

# Extract all unique code snippets
all_code = set(chain.from_iterable(df['prev_tasks'].apply(lambda x: set([item for sublist in x for item in sublist]))))
print(len(all_code))

# Set device for computation
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

# Load pre-trained tokenizer and model
code_tokenizer = AutoTokenizer.from_pretrained(config.code_model_name)
if code_tokenizer.pad_token is None:
    code_tokenizer.pad_token = code_tokenizer.eos_token
code_model = AutoModel.from_pretrained(config.code_model_name).to(device)

print(f"run with {device}, {config.code_model_name}")

# Process code snippets in batches and generate embeddings
rows = process_in_batches(code_model, code_tokenizer, list(all_code), 32, device)

# Save the embeddings to a pickle file
with open(config.save_code_embedding, 'wb') as f:
    pickle.dump(rows, f)