import torch
import torchtext
import json
from datasets import load_dataset
from pandas import DataFrame
from helpers import select_label
from helpers import merge_labels
from helpers import tokenize_and_get_dataloader
from helpers import train_model
from helpers import get_class_weights
from helpers import get_dummy_input
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AdamW
import argparse

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'distilbert-base-cased'
BATCH_SIZE = 32
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# model implementation adapted from: https://towardsdatascience.com/transformers-can-you-rate-the-complexity-of-reading-passages-17c76da3403
# author: Peggy Chang
class DistilBertConcat(nn.Module):  
    def __init__(self):        
        super().__init__() 
        self.distilbert_model = DistilBertModel.from_pretrained(MODEL_NAME, torchscript=True)  
        self.regressor = nn.Linear(768*4, 14)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.distilbert_model(input_ids, attention_mask, 
                                        return_dict=True, output_hidden_states=True)        
        hidden_states = raw_output["hidden_states"] 
        hidden_states = torch.stack(hidden_states) 
        concat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1) 
        first_token = concat[:, 0, :]        
        output = self.regressor(first_token)    
        return output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Run training only on a few training samples for testing purposes of entire pipeline')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of epochs to train for')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='learning rate for training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size for training')

    
    args = parser.parse_args()
    if args.debug:
        print("Running in DEBUG")
    
    print("Preprocessing data...")
    dataset = load_dataset("go_emotions", "simplified")
    #go through and select only one label for each sample
    dataset_train = dataset['train'].map(select_label)
    dataset_val = dataset['validation'].map(select_label)
    #merge all labels
    df_train, df_val, LABELS = merge_labels(DataFrame.from_dict(dataset_train), 
                                             DataFrame.from_dict(dataset_val), 
                                             dataset['train'].features['labels'].feature.names)
    #save remapped labels as json
    LABELS = {v: k for k, v in LABELS.items()}
    with open('labelMap.json', 'w') as fp:
        labelMap = json.dumps(LABELS)
        json.dump(labelMap, fp)
    
    #set up tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, use_fast=True, 
                                             do_lower_case=True)
    
    if args.debug:
        #for debugging purposes only train model on subset of dataset
        df_train = df_train.groupby('labels').apply(lambda s: s.sample(5))
        df_val = df_val.groupby('labels').apply(lambda s: s.sample(5))
    
    #set up dataloaders
    dataloader_train, dataloader_val = tokenize_and_get_dataloader(tokenizer,
                                                                    df_train,
                                                                    df_val,
                                                                    args.batch_size,
                                                                    padding=True)
    
    print("Setting up hyper-parameters...")
    #setup hyper-parameters
    model = DistilBertConcat().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(get_class_weights(df_train)).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    #train model
    trained_model = train_model(model, dataloader_train, dataloader_val, args.epochs, optimizer, 
                                    criterion, fine_tune=True, device=DEVICE, run_name='distilbert_trained')
    print("Saving model...")
    #save the trace of the model
    dummy_input = get_dummy_input(tokenizer, DEVICE)
    trained_model.eval()
    #creating trace
    traced_model = torch.jit.trace(trained_model, dummy_input)
    #save
    torch.jit.save(traced_model, "traced_distilbert.pt") 
    print("Trained model succesfully saved.")