import torch
from transformers import MobileBertForSequenceClassification

# Assuming you have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased')
model.to(device)
model.eval()

# Use torch.cuda.memory_allocated() to check memory usage on the GPU
print(f"Memory allocated for the model: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")