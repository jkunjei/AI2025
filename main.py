from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

from collections import Counter
import logging
import random

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

training_data = []
raw_patterns = [
    ("hello|hi|hey", "greeting", "Hello!"),
    ("bye|goodbye", "farewell", "Bye!"),
    ("thank you|thanks", "thanks", "You're welcome!"),
    ("joke|funny", "joke", "Why did the chicken cross the road?"),
    ("who are you|what are you", "identity", "I'm a chatbot."),
    ("help|what can you do", "help", "I can chat with you!"),
    ("time|what time is it", "time", "I don't have a watch, sorry."),
    ("news|headlines", "news", "Here's the latest news."),
    ("fact|interesting", "fact", "Octopuses have 3 hearts!"),
    ("music|play a song", "music", "Bohemian Rhapsody is great!"),
    ("movie|recommend a movie", "movie", "Watch Inception!"),
    ("food|what to eat", "food", "Try pizza!"),
]

for pattern, intent, response in raw_patterns:
    for p in pattern.split('|'):
        training_data.append((p, intent, response))

stop_words = set(stopwords.words('english')) - {'how', 'what', 'who', 'where', 'when', 'why', 'you', 'about', 'me', 'are', 'is'}

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

X_train = [preprocess_text(p) for p, _, _ in training_data]
y_train = [intent for _, intent, _ in training_data]
responses = {intent: [] for intent in set(y_train)}
for _, intent, response in training_data:
    if response not in responses[intent]:
        responses[intent].append(response)

vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True)
X_vec = vectorizer.fit_transform(X_train)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "SVC": SVC(kernel='linear', probability=True),
    "LinearSVC": LinearSVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SGD": SGDClassifier(),
    "Ridge": RidgeClassifier(),
    "Perceptron": Perceptron(),
    "PassiveAggressive": PassiveAggressiveClassifier()
}

for name, clf in models.items():
    clf.fit(X_vec, y_train)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        model_name = data.get('model', 'LogisticRegression')

        if model_name not in models:
            return jsonify({'response': f'Unknown model: {model_name}'}), 400

        if not user_input:
            return jsonify({'response': 'Please send a message!'}), 400

        processed = preprocess_text(user_input)
        if not processed:
            return jsonify({'response': 'Sorry, I didn’t understand that.'})

        input_vec = vectorizer.transform([processed])
        model = models[model_name]
        pred = model.predict(input_vec)[0]
        response = random.choice(responses.get(pred, ["Sorry, I didn’t understand that."]))

        session['last_intent'] = pred
        return jsonify({'response': response, 'intent': pred})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'response': f'Error: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)