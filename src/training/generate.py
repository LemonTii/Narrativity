'''
This file is to use the model
'''

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from customize import StoryGenerator

prompt = 'Once upon a time, there was a man'
model_path = None

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
if model_path is not None:
        model.load_state_dict(torch.load(model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

output_sequence = model.generate(input_ids, max_length=50, num_return_sequences=1)
decoded_output = tokenizer.decode(torch.reshape(output_sequence, (-1,)), skip_special_tokens=True)

print(decoded_output)