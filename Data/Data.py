import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def save_ids(train_ids, valid_ids, test_ids, filepath_prefix='Data/split_ids'):
    """
    Save train, validation, and test IDs to pickle files.
    
    Parameters:
    train_ids (list): List of training IDs.
    valid_ids (list): List of validation IDs.
    test_ids (list): List of test IDs.
    filepath_prefix (str): Prefix for the file paths to save the IDs.
    """
    with open(f'{filepath_prefix}_train_ids.pkl', 'wb') as f:
        pickle.dump(train_ids, f)
    with open(f'{filepath_prefix}_valid_ids.pkl', 'wb') as f:
        pickle.dump(valid_ids, f)
    with open(f'{filepath_prefix}_test_ids.pkl', 'wb') as f:
        pickle.dump(test_ids, f)

def load_ids(filepath_prefix='Data/split_ids'):
    """
    Load train, validation, and test IDs from pickle files.
    
    Parameters:
    filepath_prefix (str): Prefix for the file paths to load the IDs.
    
    Returns:
    tuple: A tuple containing lists of train, validation, and test IDs.
    """
    with open(f'{filepath_prefix}_train_ids.pkl', 'rb') as f:
        train_ids = pickle.load(f)
    with open(f'{filepath_prefix}_valid_ids.pkl', 'rb') as f:
        valid_ids = pickle.load(f)
    with open(f'{filepath_prefix}_test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    return train_ids, valid_ids, test_ids

def create_data_loader(df, dataset, text_tokenizer=None, batch_size=8, max_len_code=512,
                       padding_size_code=100, padding_size_q=30, create_split=True):
    """
    Create data loaders for training, validation, and test sets.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    dataset (Dataset): Dataset class to be used for creating data loaders.
    text_tokenizer (Tokenizer): Tokenizer for text data.
    batch_size (int): Batch size for data loaders.
    max_len_code (int): Maximum length for code sequences.
    ids_filepath_prefix (str): Prefix for the file paths to save/load the IDs.
    padding_size_code (int): Padding size for code sequences.
    padding_size_q (int): Padding size for question sequences.
    create_split (bool): Whether to create a new split or load existing split.
    
    Returns:
    tuple: A tuple containing train, validation, and test data loaders.
    """
    # Split the data to train and test by student ID
    if not create_split:
        print("Load existing splitting")
        train_ids, valid_ids, test_ids = load_ids(ids_filepath_prefix)
    else:
        student_id = df['student_id'].unique()
        id_to_struggle = df.groupby('student_id')['Label'].first()
        train_ids, test_ids = train_test_split(student_id, test_size=0.3, stratify=id_to_struggle[student_id])
        valid_ids, test_ids = train_test_split(test_ids, test_size=0.2/0.3, stratify=id_to_struggle[test_ids])

    train_df = df[df['student_id'].isin(train_ids)]
    valid_df = df[df['student_id'].isin(valid_ids)]
    test_df = df[df['student_id'].isin(test_ids)]

    # Tokenize
    train_dataset = dataset(train_df, text_tokenizer, max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q)
    valid_dataset = dataset(valid_df, text_tokenizer,max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q)
    test_dataset = dataset(test_df, text_tokenizer, max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q)

    # Dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)       
    return train_dataloader, valid_dataloader, test_dataloader

def create_data_loader_k_fold(df, dataset, text_tokenizer=None, batch_size=8, max_len_code=1024,
                                padding_size_code=2200, padding_size_q=200, k=5):
    """
    Create k-fold data loaders for cross-validation.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    dataset (Dataset): Dataset class to be used for creating data loaders.
    text_tokenizer (Tokenizer): Tokenizer for text data.
    batch_size (int): Batch size for data loaders.
    max_len_code (int): Maximum length for code sequences.
    padding_size_code (int): Padding size for code sequences.
    padding_size_q (int): Padding size for question sequences.
    k (int): Number of folds for cross-validation.
    
    Returns:
    list: A list of tuples containing train and test data loaders for each fold.
    """
    # Setup k-fold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    student_id = df['student_id'].unique()
    id_to_struggle = df.groupby('student_id')['Label'].first()
    data_loaders = []

    # Perform k-fold split
    for train_idx, test_idx in kf.split(student_id, id_to_struggle[student_id]):
        train_students = student_id[train_idx]
        test_students = student_id[test_idx]

        # Create train and test DataFrames
        train_df = df[df['student_id'].isin(train_students)]
        test_df = df[df['student_id'].isin(test_students)]

        # Tokenize
        train_dataset = dataset(train_df, text_tokenizer, max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q)
        test_dataset = dataset(test_df, text_tokenizer, max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q)

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Append to results
        data_loaders.append((train_dataloader, test_dataloader))
    return data_loaders