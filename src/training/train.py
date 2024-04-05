import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from customize import StoryDataset, StoryGenerator  # Make sure these are defined in your `customize.py`
from data_process import parallel_process_file  # Make sure this is defined
from preprocess_data import preprocess_file
from torch.cuda.amp import GradScaler, autocast
import time
from sklearn.model_selection import KFold
import numpy as np

scaler = GradScaler()

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, attn_masks, labels in dataloader:
        input_ids, attn_masks, labels = input_ids.to(device), attn_masks.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attn_masks, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn_masks, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attn_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # stories 1
    # data_path = os.path.join("data", "stories", "train.csv")
    # val_data_path = os.path.join("data", "stories", "validation.csv")

    save_path = os.path.join("..", "models")
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists

    data_path = os.path.join("data", "stories3", "data.txt")
    stories = preprocess_file(data_path)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = StoryGenerator(GPT2LMHeadModel.from_pretrained('gpt2'))

    max_length = 300
    batch_size = 8
    epoch = 5
    k = 5  # Number of folds

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    stories_array = np.array(stories)

    # # Pre-load and pre-process data once
    # print("Loading and processing training data...")
    # stories = parallel_process_file(data_path)
    # print("Loading and processing validation data...")
    # val_stories = parallel_process_file(val_data_path)

    # Convert to datasets
    # dataset = StoryDataset(stories, tokenizer, max_length=max_length)
    # val_dataset = StoryDataset(val_stories, tokenizer, max_length=max_length)

    # for e in range(epoch):
    #     start_time = time.time()
    #     print('epoch: ', e)

    #     # Training phase
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    #     train_loss = train(model, dataloader, optimizer, device)
    #     print(f'Epoch {e+1}, Training Loss: {train_loss:.4f}')

    #     # Validation phase
    #     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    #     val_loss = validate(model, val_dataloader, device)
    #     print(f'Epoch {e+1}, Validation Loss: {val_loss:.4f}')
    #     print(f'took {time.time() - start_time} seconds')

    #     # Save model
    #     model_save_path = os.path.join(save_path, f'story_generator_epoch_{e+1}.pth')
    #     torch.save(model.state_dict(), model_save_path)
    #     print('saved model')

    # print("Training and validation completed.")

    # k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(stories_array)):
        print(f"Starting fold {fold + 1}/{k}")

        # Split the data for the current fold
        train_stories = stories_array[train_idx].tolist()
        val_stories = stories_array[val_idx].tolist()

        # Prepare datasets and dataloaders
        train_dataset = StoryDataset(train_stories, tokenizer, max_length=max_length)
        val_dataset = StoryDataset(val_stories, tokenizer, max_length=max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # Reset model and optimizer for each fold
        model = StoryGenerator(GPT2LMHeadModel.from_pretrained('gpt2')).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Your existing training and validation loop here
        for e in range(epoch):
            print(f'fold: {fold+1}, epoch: {e+1}')
            # Train and validate the model
            train_loss = train(model, train_dataloader, optimizer, device)
            val_loss = validate(model, val_dataloader, device)
            print(f'Fold {fold+1}, Epoch {e+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            epoch_save_path = os.path.join(save_path, f'fold_{fold+1}_epoch_{e+1}.pth')
            torch.save(model.state_dict(), epoch_save_path)
            print(f'Model saved to {epoch_save_path}')