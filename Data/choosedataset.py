import json
import numpy as np
import pandas as pd
from Config import Config

# Read the programming concepts from a JSON file
with open('programing_concepts.json', 'r') as file:
    programming_concepts= json.load(file)
config = Config()

def add_skills_vec(df, size=10, latents=False):
    """
    Create TBPP ground  to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'prev_comp_cons' and 'prev_score' columns.
        size (int): Size of the skills vector.
        latents (bool): Whether to include latent features.
    
    Returns:
        pd.DataFrame: DataFrame with added 'skills_vec' column.
    """
    def expand_lists(row):
        expanded = []
        for lst in row:  # Iterate over 30 lists
            expanded.append(lst + [1, 1, 1])  # Add three ones at the end
        return expanded
    # Apply the function to the column
    if latents:
        df['prev_comp_cons'] = df['prev_comp_cons'].apply(expand_lists)
    size = size + 3 if latents else size
    print(size, latents)
    df['score_vec'] = df['prev_scores'].apply(lambda x: [i[-1] for i in x])
    df['skills_vec'] = df.apply(lambda row: np.dot(
        np.array(row.prev_comp_cons).T,
        np.array(row.score_vec).reshape(30, 1)
        ).reshape(size), 
        axis=1)
    # Normalize skills_vec
    all_skills_vec = np.vstack(df['skills_vec'])  # Combine all vectors into a 2D array
    min_vec = all_skills_vec.min(axis=0)  # Compute min for each skill across all rows
    max_vec = all_skills_vec.max(axis=0)  # Compute max for each skill across all rows
    range_vec = max_vec - min_vec  # Compute the range
    range_vec[range_vec == 0] = 1
    df['skills_vec'] = df['skills_vec'].apply(lambda vec: (vec - min_vec) / range_vec)
    df['skills_vec'] = df['skills_vec'].apply(lambda vec: [v if v > 0 else vec.mean() for v in vec])
    return df 

def student_id(df):
    """
    Encode student IDs as integers.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'student_id' column.
    
    Returns:
        pd.DataFrame: DataFrame with added 'student_id_encoded' column.
    """
    student_id_mapping = {sid: idx for idx, sid in enumerate(np.unique(df.student_id.values))}
    df["student_id_encoded"] = df["student_id"].map(student_id_mapping)
    return df
    
class Falcon:
    """
    Class to process Falcon dataset.
    
    Attributes:
        name (str): Name of the dataset.
        embedded (list): The paths to the codes after made process to with LLM
        df (pd.DataFrame): Processed DataFrame.
    """
    def __init__(self, latents=False, change_features=True):
        """
        Initialize the Falcon class.
        
        Args:
            latents (bool): Whether to include latent features.
            change_features (bool): Whether to change features based on programming concepts.
        """
        self.name = 'falcon'
        self.embedded = ['falcon_to_model_output']
        path = config.path_saved_falcon
        self.df = pd.read_pickle(path)
        self.df.fillna(0, inplace=True)    
        num_col = self._prev_constracts(change_features)
        self.df = student_id(self.df)
        self.df = add_skills_vec(self.df, num_col, latents)
    
    def _prev_constracts(self, change_features):
        """
        Process previous constructs and update the DataFrame.
        """
        cleaned_file_path = config.falconcode_questions_path
        questions = pd.read_csv(cleaned_file_path, sep=',')
        questions = questions[questions.columns[1:]]
        
        q_with_compu = questions.copy()
        if change_features:
            rem_col = []
            for k in programming_concepts['falcon']:
                q_with_compu[k] = q_with_compu[programming_concepts['falcon'][k]].max(axis=1)
                rem_col.extend(programming_concepts['falcon'][k])
            q_with_compu = q_with_compu.drop(columns=rem_col)
        q_with_compu.fillna(0, inplace=True)
        cols_to_merge = q_with_compu.columns[7:]
        q_with_compu['comp_cons'] = q_with_compu[cols_to_merge].apply(lambda row: row.values.tolist(), axis=1)
        questions_df = q_with_compu.drop(columns=cols_to_merge)
        questions_df['id'] = questions_df['id'].apply(lambda x: x.lower())
        Comp_cons_dict = questions_df.set_index('id')['comp_cons'].to_dict()
        self.df['prev_comp_cons'] = self.df['prev_tasks_id'].apply(lambda x: [Comp_cons_dict[i] for i in x])
        self.df['curr_comp_cons'] = self.df['new_task_id'].apply(lambda x: Comp_cons_dict[x])
        self.df.rename(columns={'prompt': 'question'}, inplace=True)
        return len(cols_to_merge)

class Codeworkout:
    """
    Class to process Codeworkout dataset.
    
    Attributes:
        name (str): Name of the dataset.
        embedded (list):  The paths to the codes after made process to with LLM.
        df (pd.DataFrame): Processed DataFrame.
    """
    def __init__(self, latents=False, change_features=True):
        """
        Initialize the Codeworkout class.
        
        Args:
            latents (bool): Whether to include latent features.
            change_features (bool): Whether to change features based on programming concepts.
        """
        self.name = 'codeworkout'
        self.embedded = ['codeworkout_to_model_output']
        df = pd.read_pickle(config.path_saved_codeworkout)
        df.fillna(0, inplace=True)
        questions_df = pd.read_excel(config.codeworkout_questions_path)
        questions_df.fillna(0, inplace=True)
        questions_df = self._order_features(questions_df) if change_features else questions_df
        cols_to_merge = questions_df.columns.tolist()[3:]

        questions_df['comp_cons'] = questions_df[cols_to_merge].apply(lambda row: row.values.tolist(), axis=1)
        questions_df = questions_df.drop(columns=cols_to_merge)
        Comp_cons_dict = questions_df.set_index('ProblemID')['comp_cons'].to_dict()
        df['prev_comp_cons'] = df['prev_tasks_id'].apply(lambda x: [Comp_cons_dict[int(i)] for i in x])
        df['curr_comp_cons'] = df['new_task_id'].apply(lambda x: Comp_cons_dict[x])
        self.df = student_id(df)
        self.df = add_skills_vec(df, len(cols_to_merge), latents)

    def _order_features(self, df):
        """
        Order features based on programming concepts.
        """
        rem_col = []
        for k in programming_concepts['codeworkout']:
            df[k] = df[programming_concepts['codeworkout'][k]].max(axis=1)
            rem_col.extend(programming_concepts['codeworkout'][k])
        df = df.drop(columns=rem_col)
        df.fillna(0, inplace=True)
        return df