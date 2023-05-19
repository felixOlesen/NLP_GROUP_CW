import pandas as pd
import torch
import copy
import numpy as np
import random
import pickle
import collections

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

PARENT_LABEL_HIERARCHY = {
    "annoyance" : "anger",
    "grief" : "sadness",
    "disappointment" : "sadness",
    "disapproval" : "sadness",
    "embarrassment" : "sadness",
    "approval" : "admiration",
    "love" : "gratitude",
    "pride" : "gratitude",
    "caring" : "gratitude",
    "nervousness" : "fear",
    "realization" : "surprise",
    "optimism" : "joy",
    "excitement" : "joy",
    "amusement" : "joy",
}

# Preprocessing helpers ################################################
def select_label(sample):
    label_len = len(sample['labels'])
    rand = random.randint(0, label_len-1)
    label = sample['labels'][rand]

    sample['labels'] = label
    return sample

def get_parent_labels(parent_label_hierarchy, labels):
    parent_labels = {}
    # replace string label with int value
    for (key, value) in parent_label_hierarchy.items():
        key_idx = labels.index(key)

        parent_labels.update({key_idx : labels.index(value)})

    return parent_labels

# merge grouped labels under their parent label
def merge_labels(df_train, df_val, original_label_list):
    parent_labels = get_parent_labels(PARENT_LABEL_HIERARCHY, original_label_list)
  
    # merge labels under their defined parent label from PARENT_LABEL_HIERARCHY
    df_train = df_train.replace(parent_labels)
    df_val = df_val.replace(parent_labels)

    # get label mappings after merging)
    # these will not be in consecutive order which will cause issues for transformers
    old_label_mappings = {}
    old_labels = df_train.labels.unique()
    old_labels.sort()
    for label in (old_labels):
        str_val = original_label_list[label]
        old_label_mappings[str_val] = label

    # new label mappings that are consectuive
    count = 0
    NEW_LABEL_MAPPINGS = {}
    for (key,value) in old_label_mappings.items():
        NEW_LABEL_MAPPINGS[key] = count
        count+=1

    # maps old merged labels to new consecutive labels
    replace_map = {}
    for (key,value) in old_label_mappings.items():
        replace_map[old_label_mappings[key]] = NEW_LABEL_MAPPINGS[key]

    # replace the labels in dataframe
    df_train = df_train.replace(replace_map)
    df_val = df_val.replace(replace_map)
    
    return df_train, df_val, NEW_LABEL_MAPPINGS

def get_dataloader(input_ids, attention_masks, labels, batch_size, shuffle):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
        
def tokenize(tokenizer, df, padding):
    encoded_data = tokenizer.batch_encode_plus(
        df.text, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=padding, 
        max_length=256, 
        return_tensors='pt',
        truncation=True
    )
        
    return encoded_data

def tokenize_and_get_dataloader(tokenizer, df_train, df_val, batch_size, padding):
    encoded_train = tokenize(tokenizer, df_train, padding)
    encoded_val = tokenize(tokenizer, df_val, padding)

    dataloader_train = get_dataloader(encoded_train['input_ids'],
                                    encoded_train['attention_mask'],
                                    torch.tensor(list(df_train.labels)),
                                    batch_size,
                                    shuffle=True)
    dataloader_val = get_dataloader(encoded_val['input_ids'],
                                    encoded_val['attention_mask'],
                                    torch.tensor(list(df_val.labels)),
                                    batch_size,
                                    shuffle=True)
  
    return dataloader_train, dataloader_val 


# Training helpers ################################################

# calculate class weights to be used for loss and sampling
def get_class_weights(df):
    label_count = df['labels'].value_counts()
    labels = label_count.index
    label_count = label_count.to_dict()
    samples_per_class = list(collections.OrderedDict(sorted(label_count.items())).values())

    class_weights = []
    total = sum(samples_per_class)

    for i in samples_per_class:
        weight = round((1 - (i/total)),4)
        class_weights.append(weight)

    return class_weights

# methods for training and evaluation of transformers
def train(model, dataloader, optim, criterion, fine_tune, device):
    model.train()
    print('training...')
    running_loss = 0.0
    running_correct_preds = 0
    count = 0
    for i, (input_id, mask, label)  in tqdm(enumerate(dataloader), total=len(dataloader)):
        count += 1

        # clear gradients
        optim.zero_grad()

        # move everything to device
        input_id, mask, label = input_id.squeeze(1).to(device), mask.to(device), label.to(device) 

        if fine_tune:
            output = model(input_id, mask)
        else:
            output = model(input_id).squeeze(1)
    
        loss = criterion(output, label)
        running_loss += loss.item()
    
        # accuracy
        _, preds = torch.max(output.data, 1)
        running_correct_preds += (preds == label).sum().item()
    
        # backward pass
        loss.backward()
        optim.step()

    # calculate loss and acc for epoch
    epoch_loss = running_loss / count
    epoch_acc = 100. * (running_correct_preds / len(dataloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, fine_tune, device):
    model.eval()
    print('validating...')
    running_loss = 0.0
    running_correct_preds = 0
    count = 0
    with torch.no_grad():
        for i, (input_id, mask, label)  in tqdm(enumerate(dataloader), total=len(dataloader)):
            count += 1

            # move everything to device
            input_id, mask, label = input_id.squeeze(1).to(device), mask.to(device), label.to(device)

            if fine_tune:
                output = model(input_id, mask)
            else:
                output = model(input_id).squeeze(1)

            loss = criterion(output, label)
            running_loss += loss.item()
      
            # accuracy
            _, preds = torch.max(output.data, 1)
            running_correct_preds += (preds == label).sum().item()

    # calculate loss and acc for epoch
    epoch_loss = running_loss / count
    epoch_acc = 100. * (running_correct_preds / len(dataloader.dataset))
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, epochs, optim, criterion, fine_tune, device, run_name):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    best_model = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"[Epoch {epoch+1} of {epochs}]")

        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optim, criterion, fine_tune, device)
        val_epoch_loss, val_epoch_acc = validate(model, val_loader,  
                                                criterion, fine_tune, device)
    
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

        print(f"training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"validation loss: {val_epoch_loss:.3f}, validation acc: {val_epoch_acc:.3f}")

        avg_train_loss = np.array(train_loss).mean()
        avg_val_loss = np.array(val_loss).mean()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
        
    return best_model

# Model trace helpers ################################################

# method adapated from official HuggingFace documentation: https://huggingface.co/docs/transformers/torchscript
def get_dummy_input(tokenizer, device):
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,1]
    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    dummy_input = [tokens_tensor, segments_tensors]
    
    return dummy_input