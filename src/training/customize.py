'''
Custom classes
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define a custom dataset class
class CustomLoss(nn.Module):
    def __init__(self, weight):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, labels):
        # Custom loss computation
        loss = torch.mean((outputs - labels) ** 2) * self.weight
        return loss
    
class StoryDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for story in stories:
            encodings_dict = tokenizer('<|startoftext|>'+ story['input'] + '<|endoftext|>', '<|startoftext|>' + story['target'] + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(torch.tensor(encodings_dict['input_ids']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

# Define the transformer-based model
class StoryGenerator(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(StoryGenerator, self).__init__()
        self.transformer = pretrained_model

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs