import string

import joblib
import nltk
import numpy as np
import streamlit as st


def lowercase(text):
    return text.lower()


def punctuation_removal(text):
    translator = str.maketrans("", "", string.punctuation)

    return text.translate(translator)


def tokenize(text):
    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words("english")

    return [token for token in tokens if token not in stop_words]


def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return tokens


def create_dictionary(messages):
    dictionary = []

    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


def predict(text, model, dictionary):
    features = create_features(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls


# Constants for model and dictionary paths
MODEL_PATH = "models/spam_classifier.joblib"
DICTIONARY_PATH = "models/dictionary.joblib"
LE_PATH = "models/label_encoder.joblib"


# Load the model and other necessary files
model = joblib.load(MODEL_PATH)
dictionary = joblib.load(DICTIONARY_PATH)
le = joblib.load(LE_PATH)

# Streamlit application
st.title("Text Classification App")
st.write("Enter a sentence to classify its sentiment or category.")

# Text input
test_input = st.text_input("Input text:")

# Perform prediction and display the result
if st.button("Predict"):
    if test_input:
        prediction_cls = predict(test_input, model, dictionary)
        st.write(f"Prediction: {prediction_cls}")
    else:
        st.write("Please enter some text to get a prediction.")
