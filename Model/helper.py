import copy
import torch
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

skills_list = []

def caculate_2losses(batch, model, device, criterion):
    """
    Calculate the combined loss for binary and skill predictions.
    
    Parameters:
    batch (dict): Batch of data.
    model (torch.nn.Module): Model to be used for predictions.
    device (str): Device to perform computations on.
    criterion (tuple): Tuple containing loss functions for binary and skill predictions.

    Returns:
    torch.Tensor: Combined loss.
    """
    global skills_list
    weight_binary = 0.5
    dict_batch = {k: v.to(device) for k, v in batch.items()}
    model_params = {k: v for k, v in dict_batch.items() if k != 'label' and k != 'skill_label'}
    binary_output, skill_output = model(*model_params.values())
    binary_labels = dict_batch['label'].unsqueeze(1).float()
    if not criterion:
        skills_list.append(skill_output)
        return binary_output, binary_labels
    loss_fn1, loss_fn2 = criterion
    skill_values = dict_batch['skill_label'].float()
    
    binary_loss = loss_fn1(binary_output, binary_labels)
    skill_loss = loss_fn2(skill_output, skill_values)
    return weight_binary * binary_loss +  (1 - weight_binary) * skill_loss
        
def caculate_1loss(batch, model, device, criterion):
    """
    Calculate the loss for binary predictions.
    
    Parameters:
    batch (dict): Batch of data.
    model (torch.nn.Module): Model to be used for predictions.
    device (str): Device to perform computations on.
    criterion (function): Loss function.
    
    Returns:
    torch.Tensor: Loss.
    """
    dict_batch = {k: v.to(device) for k, v in batch.items()}
    model_params = {k: v for k, v in dict_batch.items() if k != 'label'}
    logits = model(*model_params.values())
    label = dict_batch['label'].float()
    if not criterion:
        return logits, label
    return criterion(logits.squeeze(1), label)

def eval_loop(model, test_dataloader, device, criterion=None, caculate_func=caculate_2losses):
    """
    Evaluate the model on the test dataset.
    
    Parameters:
    model (torch.nn.Module): Model to be evaluated.
    test_dataloader (DataLoader): DataLoader for the test dataset.
    device (str): Device to perform computations on.
    criterion (function): Loss function.
    caculate_func (function): Function to calculate the loss.
    loss_fn (function): Additional loss function.
    
    Returns:
    tuple: True labels and predicted probabilities.
    """
    model.eval()
    test_loss = 0
    all_labels = []
    all_probs = []
    for i, batch in enumerate(test_dataloader):
        if i % 100 == 0:
            print(f"Test Batch {i} from {len(test_dataloader)}", flush=True)
        with torch.no_grad():
            output = caculate_func(batch, model, device, criterion)
        if criterion:
            test_loss += output.mean().item()
        else: 
            logits, label = output
            all_probs.append(torch.sigmoid(logits.cpu()))
            all_labels.append(label.cpu().numpy())
    # Concatenate all predictions and labels
    if criterion:
        return test_loss
    all_probs = torch.cat(all_probs, dim=0).numpy()  # Convert to numpy for consistency
    all_labels = np.concatenate(all_labels, axis=0)  # Use numpy concatenate
    return all_labels, all_probs

def train_loop(model, train_dataloader, device, optimizer, criterion, caculate_func=caculate_2losses):
    """
    Train the model on the training dataset.
    
    Parameters:
    model (torch.nn.Module): Model to be trained.
    train_dataloader (DataLoader): DataLoader for the training dataset.
    device (str): Device to perform computations on.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    criterion (function): Loss function.
    caculate_func (function): Function to calculate the loss.
    
    Returns:
    float: Total loss.
    """
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f"Batch {i} from {len(train_dataloader)}", flush=True)
        loss = caculate_func(batch, model, device, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def training_loop(model, train_dataloader, test_dataloader, optimizer, criterion, device, name, caculate_func=caculate_2losses):
    """
    Perform the training loop with early stopping.
    
    Parameters:
    model (torch.nn.Module): Model to be trained.
    train_dataloader (DataLoader): DataLoader for the training dataset.
    test_dataloader (DataLoader): DataLoader for the test dataset.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    criterion (function): Loss function.
    device (str): Device to perform computations on.
    name (str): Name for saving the best model.
    caculate_func (function): Function to calculate the loss.
    loss_fn (function): Additional loss function.
    
    Returns:
    torch.nn.Module: Trained model with the best weights.
    """
    # Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5

    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)
    print(len(train_dataloader), len(test_dataloader))

    for epoch in range(100):
        print(f"Epoch: {epoch}", flush=True)
        total_loss = train_loop(model, train_dataloader, device, optimizer, criterion, caculate_func)
        test_loss = eval_loop(model, test_dataloader, device, criterion=criterion, caculate_func=caculate_func)
        avg_loss_train = total_loss / len(train_dataloader)
        avg_loss_valid = test_loss / len(test_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}], LR: {current_lr:.6f}, Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_valid:.4f}, patience: {patience}', flush=True)

        # Early stopping
        if avg_loss_valid < best_loss:
            best_loss = avg_loss_valid 
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            print("success deep copy")
            patience = 5  # Reset patience counter
            torch.save(best_model_weights, name)
            print("success save in", name)
        else:
            patience -= 1
            if patience == 0:
                break
         
    torch.save(best_model_weights, name)
    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)
    model.load_state_dict(best_model_weights)
    print("Loaded best model weights.")
    return model

def results(threshold, y_true, y_prob):
    """
    Calculate and log evaluation metrics.
    
    Parameters:
    threshold (float): Threshold for binary classification.
    y_true (np.array): True labels.
    y_prob (np.array): Predicted probabilities.
    
    Returns:
    None
    """
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = np.where(y_prob > threshold, 1, 0)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    best = "best"
    if threshold == 0.5:
        best = "0.5"
    #  df = pd.concat([pd.DataFrame([[model_name, threshold, roc_auc, accuracy, precision, recall, f1]], columns=df.columns), df], ignore_index=True)
    print({"threshold": threshold, "roc_auc": roc_auc, "accuracy": accuracy, f"precision_{best}": precision, f"recall_{best}": recall, f"f1_{best}": f1})
    cm = confusion_matrix(y_true, y_pred)
    print({f"confusion_matrix_{best}": cm})