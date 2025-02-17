import os
import sys
import torch
from helper import *
from Models import *
import torch.nn as nn
sys.path.append(os.path.join(os.getcwd(), 'Thesis'))
from Data.Data import *
from Data.choosedataset import *
import Data.Data_Embedding as Data_Embedding
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer

# Set the configuration
config = Config()

# Determine if latents should be used based on the model type
latents = True if config.run_model == 'PTM' else False

# Load the dataset
data = [Codeworkout, Falcon][config.dataset](latents=latents)
df = data.df

# Load embeddings
Data_Embedding.load(data.embedded)

# Preprocess the data
df['prev_tasks'] = df['prev_tasks'].apply(lambda x: [i[-config.padding_size_code:] for i in x]) # n submissions padding_size_code snapshots

# Print label distribution
print(df['Label'].value_counts(), flush=True)

# Initialize the tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Set the device for computation
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

# Initialize the model, dataset, and loss functions based on the configuration
if config.run_model == 'PTM':
    dataset = Data_Embedding.DatasetPTM
    model = PTM
    caculate_func = caculate_2losses
    criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()] 
else:
    dataset = Data_Embedding.DatasetOneLoss
    model = AblationStudy
    caculate_func = caculate_1loss
    criterion = nn.BCEWithLogitsLoss()

# Print label distribution again (redundant)
print(df['Label'].value_counts(), flush=True)

# Initialize the tokenizer again (redundant)
text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Set the device for computation again (redundant)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

# Create data loaders for k-fold cross-validation
data_loaders = create_data_loader_k_fold(df, dataset, text_tokenizer,max_len_code=config.max_len_code, 
                                         padding_size_q=config.number_coding_tasks, padding_size_code=config.padding_size_code, 
                                         batch_size=config.batch_size)

# Function to print data loader statistics
def num_of(train_dataloader, test_dataloader):
    print(len(train_dataloader), len(test_dataloader))
    print(len(set(train_dataloader.dataset.df['student_id'])), len(set(test_dataloader.dataset.df['student_id'])))
    print(set(train_dataloader.dataset.df['student_id']).intersection(set(test_dataloader.dataset.df['student_id'])))
    print(train_dataloader.dataset.df.Label.value_counts(normalize=True))
    print(test_dataloader.dataset.df.Label.value_counts(normalize=True))

# Print statistics for each fold
for train, test in data_loaders:
    num_of(train, test)

# Initialize a dictionary to store fold results
fold_results = {'ROC-AUC' : [], 'f1' : [], 'recall': [], "precision": []}

# Perform k-fold cross-validation
for fold, (train_dataloader, test_dataloader) in enumerate(data_loaders):
    print(f"Fold {fold + 1}:")    # Prepare data for current fold
    if latents:
        m = model(config.text_model_name, config.max_len_code, len(np.unique(df['student_id_encoded'])))
    else:
        m = model(config.padding_size_code, config.text_model_name, config.max_len_code)
    optimizer = torch.optim.Adam(m.parameters(), lr=config.lr, weight_decay=1e-4)
   
    m = m.to(device)
    # Training Loop
    for epoch in range(config.epoch):
        total_loss = train_loop(m, train_dataloader, device, optimizer, criterion, caculate_func)

        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}: Loss = {total_loss / len(train_dataloader)}")

    # Evaluate the model on the test set for the current fold
    y_labels, y_probs = eval_loop(m, test_dataloader, device, caculate_func=caculate_func)
    y_prob = np.array(y_probs)
    y_true = np.array(y_labels)
    y_pred = np.where(y_prob > 0.25, 1, 0)

    # Store the results for the current fold
    fold_results['ROC-AUC'].append(roc_auc_score(y_true, y_prob))
    fold_results['precision'].append(precision_score(y_true, y_pred))
    fold_results['recall'].append(recall_score(y_true, y_pred))
    fold_results['f1'].append(f1_score(y_true, y_pred))

    # Print the results for the current fold
    for k, v in fold_results.items():
        print({f'fold{fold+1}_{k}': v[fold]})

# Aggregate and print the average results across all folds
for k, v in fold_results.items():
    print.log({f'{k}_avg': np.mean(v)})