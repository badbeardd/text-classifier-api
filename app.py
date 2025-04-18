from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load all models at startup
spam_model = joblib.load('spam_model.pkl')
phishing_model = joblib.load('phishing_model.pkl')
sentiment_model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form', methods=['POST'])
def predict_form():
    text = request.form['text']
    task = request.form.get('task', 'spam')

    if task == 'spam':
        prediction = spam_model.predict([text])[0]
        label = 'Spam' if prediction == 1 else 'Ham'
    elif task == 'sentiment':
        prediction = sentiment_model.predict([text])[0]
        label = 'Positive' if prediction == 1 else 'Negative'
    else:
        return render_template('index.html', prediction="Phishing detection not supported via form")

    return render_template('index.html', prediction=label)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    task = data.get('task')
    
    if task == 'spam':
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        prediction = spam_model.predict([text])[0]
        label = 'spam' if prediction == 1 else 'ham'
        return jsonify({'task': 'spam', 'prediction': label})

    elif task == 'sentiment':
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        prediction = sentiment_model.predict([text])[0]
        label = 'positive' if prediction == 1 else 'negative'
        return jsonify({'task': 'sentiment', 'prediction': label})

    elif task == 'phishing':
        features = data.get('features')
        if not features:
            return jsonify({'error': 'No phishing features provided'}), 400
        try:
            input_df = pd.DataFrame([features])
            prediction = phishing_model.predict(input_df)[0]
            label = 'phishing' if prediction == 1 else 'legitimate'
            return jsonify({'task': 'phishing', 'prediction': label})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid task specified'}), 400

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
