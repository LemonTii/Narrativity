'''
Custom classes
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from transformers import SFTTrainer

def custom_collate(batch):
    # Separate input_ids, attention_masks, and labels
    input_ids = [item[0] for item in batch]
    attn_masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Pad sequences so they are all the same length as the longest sequence
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attn_masks_padded = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) 

    return input_ids_padded, attn_masks_padded, labels_padded

# Define a custom dataset class
class CustomLoss(nn.Module):
    def __init__(self, weight):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, labels):
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
        input_ids = input_encodings['input_ids'].squeeze()
        attn_masks = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()
        return input_ids, attn_masks, labels

# Define the transformer-based model
class StoryGenerator(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(StoryGenerator, self).__init__()
        self.transformer = pretrained_model
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
class MyCustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, neural_loss_model=None, loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_loss_model = neural_loss_model
        self.loss_weight = loss_weight
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute the default model loss
        if "labels" in inputs:
            labels = inputs["labels"]
            loss_default = outputs.loss if outputs.loss is not None else model.compute_loss(outputs, labels)
        else:
            loss_default = 0

        # Compute the loss from the neural network
        with torch.no_grad():
            loss_neural = self.neural_loss_model(logits, labels).mean()

        # Combine the losses
        combined_loss = (1 - self.loss_weight) * loss_default + self.loss_weight * loss_neural

        return (combined_loss, outputs) if return_outputs else combined_loss