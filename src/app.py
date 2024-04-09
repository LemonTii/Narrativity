from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from training.customize import StoryGenerator
from flask_session import Session

app = Flask(__name__)

# Assuming the setup from your message, initialize your model and tokenizer here
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token

# # Load the pretrained model
# pretrained_model_path = 'gpt2'  # Use the appropriate path or identifier
# pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)

# # Load your StoryGenerator model
# model_path = 'path/to/your/model/fold_2_epoch_100.pth'  # Adjust the path
# model = StoryGenerator(pretrained_model).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
max_length = 200

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/', methods=['GET'])
def home():
    if 'messages' not in session:
        session['messages'] = []
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    
    # generated_response = generate_response(prompt)

    session['messages'].append({'text': prompt, 'type': 'user-message'})
    session['messages'].append({'text': "Second testing", 'type': 'bot-message'})
    session.modified = True  # Ensure the session is marked as modified
    
    return redirect(url_for('home'))

def generate_response(prompt):
    # Encode user prompt
    encoded_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate text using your model
    # Adjust generation parameters as necessary
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
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    app.run(debug=True)
