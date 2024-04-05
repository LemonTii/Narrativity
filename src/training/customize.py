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
        self.stories = stories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        input_encodings = self.tokenizer(story['input'], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        target_encodings = self.tokenizer(story['target'], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        input_ids = input_encodings['input_ids'].squeeze()  # Remove batch dimension
        attn_masks = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()
        return input_ids, attn_masks, labels

# Define the transformer-based model
class StoryGenerator(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(StoryGenerator, self).__init__()
        self.transformer = pretrained_model

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs