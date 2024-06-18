import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
from sklearn.tree import DecisionTreeClassifier


#Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    # Remove non-alphabet characters and convert to lowercase
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text_cleaned = re.sub('[^a-zA-Z]', ' ', text)
    text_cleaned = text_cleaned.lower()
    
    # Tokenize the cleaned text
    words = word_tokenize(text_cleaned)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Lemmatize and stem each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    
    # Join the stemmed words back into a single string
    processed_text = ' '.join(stemmed_words)
    
    return processed_text

def hate_speech():
    # Load and preprocess data
    data = pd.read_csv("hate_speech.csv")
    data = data.rename(columns={"text": "Text", "label": "Category"})
    data=data[["Text","Category"]]

    data = data.sample(frac=1)
    data["Text"] = data["Text"].str.lower()
    data["Text"] = data["Text"].apply(preprocess)

    # Split data into features (X) and target labels (y)
    X = data['Text']
    y = data['Category']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer()
    Text_tf = vectorizer.fit_transform(x_train)
    x_test_vect = vectorizer.transform(x_test)


    # Train Logistic Regression model
    lgr = LogisticRegression()
    lgr.fit(Text_tf, y_train)

    # Train LinearSVC model
    svc = LinearSVC(dual=False)
    svc.fit(Text_tf, y_train)

    # Train Multinomial Decission Tree model
    dt = DecisionTreeClassifier()
    dt.fit(Text_tf, y_train)

    # Save trained models and vectorizer using pickle
    with open('models/hs_model_lgr.pkl', 'wb') as model_file:
        pickle.dump(lgr, model_file)

    with open('models/hs_model_svc.pkl', 'wb') as model_file:
        pickle.dump(svc, model_file)

    with open('models/hs_model_dt.pkl', 'wb') as model_file:
        pickle.dump(dt, model_file)

    with open('models/hs_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model training and saving completed successfully.")


def depression():
     # Load and preprocess data
    data = pd.read_csv('sentiment_tweets3.csv')
    data = data.rename(columns = {"message to examine":"Text", "label (depression result)" :"Category" })
    data=data[["Text","Category"]]

    data = data.sample(frac=1)
    data["Text"] = data["Text"].str.lower()
    data["Text"] = data["Text"].apply(preprocess)

    # Split data into features (X) and target labels (y)
    X = data['Text']
    y = data['Category']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer()
    Text_tf = vectorizer.fit_transform(x_train)
    x_test_vect = vectorizer.transform(x_test)

    # Train Logistic Regression model
    lgr = LogisticRegression()
    lgr.fit(Text_tf, y_train)

    # Train LinearSVC model
    svc = LinearSVC(dual=False)
    svc.fit(Text_tf, y_train)

    # Train Multinomial Naive Bayes model
    dt = DecisionTreeClassifier()
    dt.fit(Text_tf, y_train)

    # Save trained models and vectorizer using pickle
    with open('models/dp_model_lgr.pkl', 'wb') as model_file:
        pickle.dump(lgr, model_file)

    with open('models/dp_model_svc.pkl', 'wb') as model_file:
        pickle.dump(svc, model_file)

    with open('models/dp_model_dt.pkl', 'wb') as model_file:
        pickle.dump(dt, model_file)

    with open('models/dp_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model training and saving completed successfully.")


depression()
print("dep done")
hate_speech()
print("hs done")
