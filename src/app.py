from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from training.generate_gemma import GemmaStoryModel
import os
from flask_session import Session

app = Flask(__name__)

model_path = os.path.join('models', 'fold_2_epoch_100_gemma.pth')
generator = GemmaStoryModel(model_path)
max_length = 200

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/', methods=['GET'])
def home():
    if 'messages' not in session:
        session['messages'] = []
    return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def submit_prompt():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received'}), 400
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt received'}), 400
    
    # Process your prompt here...
    print("Received prompt:", prompt)
    prompt = request.json.get('prompt')
    session['prompt'] = prompt
    session['messages'].append({'text': prompt, 'type': 'user-message'})

    return jsonify({'message': 'Prompt received', 'prompt': prompt})

@app.route('/generate', methods=['POST'])
def generate_text():

    prompt = session.get('prompt', '')
    generated_response = generator.generate_story(prompt, max_length)

    print("generated_response: ", generated_response)
    session['messages'].append({'text': generated_response, 'type': 'bot-message'})

    return jsonify({'generatedResponse': generated_response})

@app.route('/latest-messages', methods=['GET'])
def latest_messages():
    return jsonify(messages=session.get('messages', []))

if __name__ == '__main__':
    app.run(debug=True)
