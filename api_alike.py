import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from flask import Flask, request, jsonify# IMPORTING LIBRARY TO IMPLEMENT TF-IDF ALGORITHM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from celery import Celery
import time
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def preprocess(data):
    # Remove non-alphabet characters and convert to lowercase
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    data_cleaned = re.sub('[^a-zA-Z]', ' ', data)
    data_cleaned = data_cleaned.lower()
    
    # Tokenize the cleaned text
    words = word_tokenize(data_cleaned)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Lemmatize and stem each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    
    # Join the stemmed words back into a single string
    processed_text = ' '.join(stemmed_words)
    
    return processed_text

def depForm(request_data):
    # intialize needed variables
    pos_score = 0
    neg_score = 0
    neg_score += int(request_data.get('always', ''))
    neg_score += int(request_data.get('usually', ''))*.67
    neg_score += int(request_data.get('sometimes', ''))*.33
    pos_score += int(request_data.get('never', ''))

    return 1 if neg_score > pos_score else 0


sia = SentimentIntensityAnalyzer()

def vadar_sentiment(text):

    sentiment = sia.polarity_scores(text)
    return 1 if sentiment['compound'] < 0 else 0





app = Flask(__name__)


# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


def long_task(duration):
    print(f'Starting long task...')
    time.sleep(duration)
    print(f'Long task completed after {duration} seconds')

# Flask route to trigger the background task
@app.route('/')
def index():
    # Trigger the background task (sleep for 10 seconds in this example)
    long_task.delay(10)
    return 'Background task started!'

# Load Logistic Regression model
with open('models/hs_model_lgr.pkl', 'rb') as model_file:
    lgr_hs = pickle.load(model_file)

# Load Logistic Regression model
with open('models/hs_model_svc.pkl', 'rb') as model_file:
    svc_hs = pickle.load(model_file)

with open('models/hs_model_dt.pkl', 'rb') as model_file:
    dt_hs = pickle.load(model_file)

# Load Vectorizer
with open('models/hs_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer_hs = pickle.load(vectorizer_file)


# for the dep model 

with open('models/dp_model_lgr.pkl', 'rb') as model_file:
    lgr_dp = pickle.load(model_file)

# Load Logistic Regression model
with open('models/dp_model_svc.pkl', 'rb') as model_file:
    svc_dp = pickle.load(model_file)

with open('models/dp_model_dt.pkl', 'rb') as model_file:
    dt_dp = pickle.load(model_file)

# Load Vectorizer
with open('models/dp_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer_dp = pickle.load(vectorizer_file)

# with open('models/dp_vectorizer.pkl', 'rb') as vectorizer_file:
#     roberta = pickle.load(open('models/roberta_sentiment_model.pkl', 'rb'))

# with open('models/dp_vectorizer.pkl', 'rb') as vectorizer_file:
#     vader = pickle.load(open('models/vader_sentiment_model.pkl', 'rb'))


@app.route('/hello')
def home():
    return "hello fesu"


@app.route('/predict_HS', methods=['POST'])
def predict_HS():
    try:

        request_data = request.get_json()
        text = request_data.get('text', '')

        # Your existing preprocessing code
        a = preprocess(text)

        example_counts = vectorizer_hs.transform([a])

        # Predict using Logistic Regression
        lgr_prediction = lgr_hs.predict(example_counts)

        # Predict using SVC
        svc_prediction = svc_hs.predict(example_counts)

        #predict using Decision Tree
        dt_prediction = dt_hs.predict(example_counts)

        prediction = dt_prediction[0]+lgr_prediction[0]+dt_prediction[0]

        if prediction >1:
            # If either model predicts 1, return a response indicating an issue
            return jsonify({'prediction': 'Hate speech detected'})

        return jsonify({'prediction': 'Normal'})

    except Exception as e:
        return jsonify({'error': str(e)})


# Depression Model End Point 
@app.route('/predict_DP', methods=['POST'])
def predict_DP():
    try:
        # Get JSON data from the request
        request_data = request.get_json()
        
        # Ensure request_data is a dictionary
        if not isinstance(request_data, dict):
            return jsonify({'error': 'Invalid input, expected a dictionary'}), 400

        # Access the dictionary fields as needed
        # Example: Assuming the dictionary has fields 'field1' and 'field2'
        text = request_data.get('text', '')

        count=0
        count+=vadar_sentiment(text)
        count+=depForm(request_data)

        if(count>0) :

            # Your existing preprocessing code
            a = preprocess(text)

            example_counts = vectorizer_dp.transform([a])

            # Predict using Logistic Regression
            lgr_prediction = lgr_dp.predict(example_counts)

            # Predict using SVC
            svc_prediction = svc_dp.predict(example_counts)

            # Predict using Decision Tree
            dt_prediction = dt_dp.predict(example_counts)

            prediction = dt_prediction[0]+lgr_prediction[0]+dt_prediction[0]

            if prediction > 1:
                # If either model predicts 1, return a response indicating an issue
                return jsonify({'prediction': 'Depression detected'})
            
            return jsonify({'prediction': 'Negative'})

            
        # Otherwise, predict normally
        return jsonify({'prediction': 'Normal'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    #app.run(ssl_context=("cert.pem", "key.pem"))

