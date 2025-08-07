import torch
"""
This module provides classes and functions for handling and embedding data for a machine learning model.
Classes:
    Dataset_Embedding_Q: A custom PyTorch Dataset class for embedding questions and code snippets.
    SkillDataset: A subclass of Dataset_Embedding_Q that includes additional features and skill labels.
    LatentSkillDataset: A subclass of SkillDataset that includes student IDs.
Functions:
    load(all): Loads pickled dictionaries from files and combines them into a single dictionary.
Attributes:
    base_path (str): The base path for the embedding files.
    code_to_model_dict (dict): A dictionary mapping code snippets to their embeddings.
"""
import pickle
from torch.utils.data import Dataset

base_path = 'Datasets/'
code_to_model_dict = None

def load(all):
    global code_to_model_dict
    print(all)
    for i in all:
        dict_list = []
        with open(base_path + i + '.pkl', 'rb') as file:
            dict_list.append(pickle.load(file))
    code_to_model_dict = {k: v for d in dict_list for k, v in d.items()}

class DatasetOneLoss(Dataset):
    """
    A custom PyTorch Dataset class for embedding questions and code samples for the ablation and base for out PTM
    Attributes:
        df (pd.DataFrame): DataFrame containing the dataset.
        text_tokenizer (Tokenizer): Tokenizer for processing text data.
        max_len_code (int): Maximum length for code sequences. Default is 512 (size of the code embedding vec).
        padding_size_code (int): Padding size for code sequences. Default is 100 (limit attempts to each question).
        padding_size_q (int): Padding size for question sequences. Default is 200.
    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Retrieves the sample at the specified index.
            Args:
                idx (int): Index of the sample to retrieve.
            Returns:
                dict: A dictionary containing the following keys:
                    - 'text_input_ids' (torch.Tensor): Tokenized input IDs for the question.
                    - 'text_attention_mask' (torch.Tensor): Attention mask for the question.
                    - 'code_embedding' (torch.Tensor): Embedding tensor for the code samples.
                    - 'code_num' (torch.Tensor): Tensor containing the number of code snapshots.
                    - 'prev_struggling' (torch.Tensor): Tensor containing the previous struggling label.
                    - 'label' (torch.Tensor): Tensor containing the label for the sample.
    """
    def __init__(self, df, text_tokenizer, max_len_code=1024, padding_size_code=100, padding_size_q=30):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.max_len_text = text_tokenizer.model_max_length
        self.max_len_code = max_len_code
        self.padding_size_code = padding_size_code
        self.padding_size_q = padding_size_q

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['new_task']
        code_samples = row['prev_tasks']
        label = torch.tensor(row['Label'], dtype=torch.float)
        q_num = len(code_samples)
        
        # Tokenize curr coding text
        text_inputs = self.text_tokenizer(question, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')
        
        len_snapshots = [0 for i in range(q_num)]
        embedding = torch.zeros((self.padding_size_q, self.padding_size_code, self.max_len_code), dtype=torch.float)

        for q_idx, codes in enumerate(code_samples):
            len_snapshots[q_idx] = len(codes)
            for c_idx, code in enumerate(codes):
                embedding[q_idx, c_idx , :] = torch.tensor(code_to_model_dict[code])

        return {
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'code_embedding': embedding,
            'code_num': torch.tensor(len_snapshots),
            'prev_struggling': torch.tensor(row['prev_labels']),
            'label': label
        }
    
class DatasetPTM(DatasetOneLoss):
    """
    A dataset class for handling latent skill data and create the dataset for PTM, inheriting from Dataset_Embedding_Q.

    Attributes:
        df (pd.DataFrame): The dataframe containing the dataset.
        text_tokenizer (callable): A tokenizer function for processing text data.
        max_len_code (int): Maximum length of the code sequences. Default is 512.
        padding_size_code (int): Padding size for the code sequences. Default is 100.
        padding_size_q (int): Padding size for the question sequences. Default is 30.

    Methods:
        __getitem__(idx):
            Retrieves the item at the specified index, including additional features such as 
            'curr_comp_cons', 'skills_vec', and 'student_id_encoded'.
    """
    # For the PTM
    def __init__(self, df, text_tokenizer, max_len_code=1024, padding_size_code=2000, padding_size_q=200):
        super().__init__(df, text_tokenizer, max_len_code, padding_size_code, padding_size_q)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]
        dict['features'] = torch.tensor(row['curr_comp_cons'], dtype=torch.float32)     
        dict['skill_label'] = torch.tensor(row['skills_vec'], dtype=torch.float32)     
        dict['student_id'] = torch.tensor(row["student_id_encoded"], dtype=torch.float32)     
        return dict