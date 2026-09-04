from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer (from Part 1)
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "Fake News Detection API is running. POST to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data.get('title', '')
    content = data.get('content', '')

    if not title and not content:
        return jsonify({'error': 'Please provide a title or content'}), 400

    full_text = title + " " + content
    text_vector = vectorizer.transform([full_text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    label = "Credible" if prediction == 1 else "Fake"
    confidence = round(float(probability[prediction]) * 100, 2)

    return jsonify({
        'label': label,
        'confidence': confidence,
        'fake_probability': round(float(probability[0]) * 100, 2),
        'credible_probability': round(float(probability[1]) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
