'''
For training models
'''

import torch
import torch.nn as nn
import torch.optim as optim
from customize import CustomLoss, StoryDataset, StoryGenerator
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attn_masks, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_masks, labels=labels)
        loss = outputs.loss
        print('loss', loss)

        loss.backward()
        print('bkwd', loss)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Sample usage
if __name__ == "__main__":
    # Sample story dataset
    stories = [
        {'input': "Once upon a time,", 'target': "there was a princess."},
        {'input': "In a faraway land,", 'target': "there lived a brave knight."},
        {'input': "Once upon a time, little red lived in a house.", 'target': "In the house there was a hidden treasure room filled with goods."}
    ]
    model_path = None
    save_path = None

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    model = StoryGenerator(GPT2LMHeadModel.from_pretrained('gpt2'))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Prepare dataset and dataloader
    dataset = StoryDataset(stories, tokenizer, max_length=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # criterion = CustomLoss(weight=0.5)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(5):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        if save_path is not None:
            torch.save(model.state_dict(), f'{save_path}/story_generator_epoch_{epoch+1}.pth')