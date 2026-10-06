import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

model = joblib.load('emotion_model (1).pkl')
emotion_names = joblib.load('emotion_names (1).pkl')

def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    new = ""
    for ch in txt:
        if ch.isalpha() or ch == " ":
            new = new + ch
    words = []
    for word in new.split():
        if word not in stop_words:
            word = lemmatizer.lemmatize(word, pos='v')
            word = lemmatizer.lemmatize(word)
            words.append(word)
    return " ".join(words)

st.set_page_config(page_title="EmotionAI")

st.title("EmotionAI")
st.write("Type a sentence and the model will predict the emotion inside it.")

user_text = st.text_area("Type a sentence", height=100)

if st.button("Predict"):
    if user_text:
        cleaned = clean_text(user_text)
        prediction = model.predict([cleaned])
        emotion = emotion_names[prediction[0]]
        st.success("The emotion is " + emotion)
    else:
        st.warning("Please type a sentence")

with st.expander("Example sentences"):
    st.write("i feel so alone today")
    st.write("i am very happy today")
    st.write("i feel so angry right now")

st.sidebar.title("About")
st.sidebar.write("Dataset is the emotions text dataset from Kaggle")
st.sidebar.write("Model is Logistic Regression with bag of words")
st.sidebar.write("Test accuracy is 0.866")