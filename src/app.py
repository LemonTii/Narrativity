from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from training.generate_gemma import GemmaStoryModel
import os
# from training.customize import StoryGenerator
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

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    session['messages'].append({'text': prompt, 'type': 'user-message'})
    session.modified = True
    redirect(url_for('home'))
    generated_response = generator.generate_story(prompt, max_length)
    session['messages'].append({'text': generated_response, 'type': 'bot-message'})
    session.modified = True
    
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
