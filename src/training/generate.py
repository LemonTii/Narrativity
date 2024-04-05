import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from customize import StoryGenerator  # Assuming this imports your custom class
import os

prompt = "There was a happy couple"
model_path = os.path.join('..', 'models', 'fold_2_epoch_5.pth')

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize and load the StoryGenerator model
# Assuming the StoryGenerator wraps a GPT2LMHeadModel
pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
model = StoryGenerator(pretrained_model)
model.load_state_dict(torch.load(model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

# Prepare the prompt
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

# Generating text
# Assuming you need to call the generate method on the underlying transformer model
output_sequences = model.transformer.generate(
    input_ids=input_ids,
    max_length=200,  # Example max_length
    # Add other generation parameters as necessary
)

decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(decoded_output)