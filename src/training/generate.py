import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from customize import StoryGenerator  # Assuming this imports your custom class
import os

prompt = "Caroline. I learned her name through the phonebook, my shaking fingers carefully caressing its pages as I searched for the address I'd seen her at so many times."
model_path = os.path.join('..', 'models', 'fold_2_epoch_100.pth')
# model_path = os.path.join('..', 'models', 'story_generator_epoch_5.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
# Initialize and load the StoryGenerator model
# Assuming the StoryGenerator wraps a GPT2LMHeadModel
pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
model = StoryGenerator(pretrained_model).to(device)

# model = StoryGenerator(AutoModelForCausalLM.from_pretrained(model_id,
#                                                             quantization_config=bnb_config,
#                                                             device_map={"":0},
#                                                             token=os.environ['HF_TOKEN'])).to(device)
model.load_state_dict(torch.load(model_path))
# model.to(device)
model.eval()  # Set the model to evaluation mode
max_length = 200

# Prepare the prompt
encoded_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Generating text
# Assuming you need to call the generate method on the underlying transformer model
output_sequences = model.transformer.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # Pass the attention mask
    max_length=max_length,
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id if necessary
    do_sample=True,
    temperature=0.7,  # Adjust based on your needs
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=3,
    no_repeat_ngram_size=2
)

decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(decoded_output)