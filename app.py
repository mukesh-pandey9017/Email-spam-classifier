import joblib
import streamlit as st
import string
import nltk
import regex as re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

tfidf = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

ps = PorterStemmer()
stop_words = stopwords.words("english")
def transform_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ",text)
    text = nltk.word_tokenize(text)
    text = [ps.stem(x) for x in text if x not in stop_words]
    text = " ".join(text)
    return text


if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("NOT spam")

