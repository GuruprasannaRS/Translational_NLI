import os
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

class Dataloader:
    def __init__(self, word_vectors, max_sent_length, X_train, y_train, X_val, y_val, X_test, y_test,premise_head,hypo_head):
        self.word_vectors = word_vectors
        self.max_sent_length = max_sent_length
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val 
        self.y_val = y_val
        self.X_test = X_test 
        self.y_test = y_test 
        self.premise_head = premise_head
        self.hypo_head = hypo_head

    def tokens2indices(self,tokens):
            sos_id = self.word_vectors.key_to_index["<SOS>"]
            eos_id = self.word_vectors.key_to_index["<EOS>"]
            pad_id = self.word_vectors.key_to_index["<PAD>"]
            unk_id = self.word_vectors.key_to_index["<UNK>"]
            indices = []
            # print('token_size', len(tokens))
            for sentence in tokens:
                sentence_tokens =  sentence[:self.max_sent_length - 2]
                sentence_indices = [sos_id]
                for token in sentence_tokens:
                        token_index = self.word_vectors.key_to_index.get(token, unk_id)
                        sentence_indices.append(token_index)
                sentence_indices.append(eos_id)
                if len(sentence_indices) < self.max_sent_length:     
                    pad_count = self.max_sent_length - len(sentence_indices) 
                    padded = [pad_id] * pad_count 
                    padded_indices = sentence_indices + padded
                    indices.append(padded_indices)
                else:
                    indices.append(sentence_indices)    
            return indices
    
    def indices_processing(self,df):
        prem = df[self.premise_head].tolist()
        hypo = df[self.hypo_head].tolist()
        # print(prem[:1])
        premise_indices = self.tokens2indices(prem)
        lengths = [len(lst) for lst in premise_indices]
        # print(lengths[:10])
        # print(premise_indices[:1])
        hypothesis_indices = self.tokens2indices(hypo)
        return premise_indices,hypothesis_indices
    
    def get_train_loader(self):
            train_data = self.X_train
            train_labels = self.y_train
            train_labels = train_labels.tolist()
            premise_indices, hypothesis_indices = self.indices_processing(train_data)
            combined_indices = [premise + hypothesis for premise, hypothesis in zip(premise_indices, hypothesis_indices)]
            combined_dataset = TensorDataset(torch.tensor(combined_indices), torch.tensor(train_labels))
            # lengths = [len(lst) for lst in combined_dataset]
            train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
            return train_loader
    
    def get_val_loader(self):
            
            val_data = self.X_val
            val_labels = self.y_val
            val_labels = val_labels.tolist()
            premise_indices, hypothesis_indices = self.indices_processing(val_data)
            combined_indices = [premise + hypothesis for premise, hypothesis in zip(premise_indices, hypothesis_indices)]
            combined_dataset = TensorDataset(torch.tensor(combined_indices), torch.tensor(val_labels))
            val_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
            return val_loader
    
    def get_test_loader(self):
            test_data = self.X_test
            test_labels = self.y_test
            test_labels = test_labels.tolist()
            premise_indices, hypothesis_indices = self.indices_processing(test_data)
            combined_indices = [premise + hypothesis for premise, hypothesis in zip(premise_indices, hypothesis_indices)]
            combined_dataset = TensorDataset(torch.tensor(combined_indices), torch.tensor(test_labels))
            test_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
            return test_loader
           
    
    
   


