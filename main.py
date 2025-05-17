from dotenv import load_dotenv
load_dotenv()
import logging
import os
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from openai import OpenAI

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'superstructure'
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = OpenAI(api_key=api_key)

stop_words = set(stopwords.words('english')).difference(
    {'how', 'what', 'who', 'where', 'when', 'why', 'you', 'about', 'me', 'are', 'is'}
)

USE_MOCK = False

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def generate_gpt_response(prompt):
    if USE_MOCK:
        logging.info("Using mock response instead of OpenAI API")
        return "This is a placeholder, real response from OpenAI would be here."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}", exc_info=True)
        return f"OpenAI API error: {str(e)}"

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    logging.info(f"Received data: {data}")
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'response': 'Please send a message!'}), 400

    processed_input = preprocess_text(user_input)
    if not processed_input:
        return jsonify({'response': 'Sorry, I didn’t understand that.'})

    response_text = generate_gpt_response(user_input)
    session['last_message'] = user_input
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
