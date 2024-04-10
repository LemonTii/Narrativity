import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import os
from dotenv import load_dotenv

class StoryRatingModel(nn.Module):
    def __init__(self, embedding_dim=256000, projection_dim=512, hidden_dim=256, output_dim=1):
        super(StoryRatingModel, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.lstm = nn.LSTM(projection_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        projected = F.relu(self.projection(x))
        _, (hidden, _) = self.lstm(projected)
        output = self.fc(hidden.squeeze(0))
        rating = torch.sigmoid(output)
        return rating

class StoriesDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }, self.labels[idx]

def tokenize_texts(texts, tokenizer):
    tokenized_stories = tokenizer(texts, padding=True, return_tensors="pt")
    return tokenized_stories

def load_dataset(stories_file, labels_file, tokenizer):
    with open(stories_file, 'r', encoding='utf-8') as file:
        stories = [story.strip() for story in file.readlines()]

    labels_df = pd.read_csv(labels_file)
    filtered_labels_df = labels_df[labels_df['hits'] != 0]
    filtered_stories = [stories[i] for i in filtered_labels_df.index]
    
    assert len(filtered_stories) == len(filtered_labels_df), "Stories and labels length mismatch"
    
    tokenized_stories = tokenize_texts(filtered_stories, tokenizer)
    labels = (filtered_labels_df['kudos'] / filtered_labels_df['hits']).values
    
    return tokenized_stories, labels

def split_dataset(tokenized_stories, labels, test_size=0.2, random_state=42):
    input_ids_train, input_ids_val, labels_train, labels_val = train_test_split(
        tokenized_stories['input_ids'], 
        labels,
        test_size=test_size,
        random_state=random_state
    )
    attention_mask_train, attention_mask_val, _, _ = train_test_split(
        tokenized_stories['attention_mask'], 
        labels,
        test_size=test_size,
        random_state=random_state
    )
    train_dataset = StoriesDataset(input_ids_train, attention_mask_train, labels_train)
    val_dataset = StoriesDataset(input_ids_val, attention_mask_val, labels_val)
    return train_dataset, val_dataset

def main():
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv('TOKEN')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=os.environ['HF_TOKEN'])
    tokenizer.padding_side = 'right'
    criterion = nn.MSELoss()
    
    stories_path = os.path.join("data", "loss_net", "cleaned_stories.csv")
    labels_path = os.path.join("data", "loss_net", "cleaned_kudos.csv")
    model_save_path = os.path.join("..", "models", 'loss_net.pth')

    stories, labels = load_dataset(stories_path, labels_path, tokenizer)

    train_dataset, val_dataset = split_dataset(stories, labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = StoryRatingModel(embedding_dim=len(tokenizer.vocab.keys()), projection_dim=512, hidden_dim=256, output_dim=1).to('cuda')
    optimizer = AdamW(model.parameters(), lr=3e-5)

    best_val_loss = float('inf')
    for epoch in range(1, 100):
        model.train()
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to('cuda').float()
            attention_mask = inputs['attention_mask'].to('cuda')
            labels = labels.to('cuda')
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        
if __name__ == '__main__':
    main()