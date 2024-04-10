import torch
from transformers import MobileBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased')
model.to(device)
model.eval()

print(f"Memory allocated for the model: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")