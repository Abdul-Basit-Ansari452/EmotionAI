EmotionAI
This project detects the emotion in a text sentence. The model reads a sentence like "i feel so alone today" and predicts one of six emotions, sadness, anger, joy, love, fear or surprise.

Dataset
I used the Emotions dataset for NLP from Kaggle.

https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp-analysis

The dataset has train.txt with sentences and one emotion label for each sentence.

What I did
Loaded train.txt into a pandas dataframe
Cleaned the text with lower casing, punctuation removal, number removal, emoji removal, stopword removal, spelling correction with pyspellchecker and lemmatization with NLTK
Turned the emotion labels into numbers
Split the data into 80 percent train and 20 percent test
Turned the text into numbers with bag of words and TF-IDF
Trained Naive Bayes and Logistic Regression with sklearn pipelines
Compared all four results and picked the best one
Saved the final model with joblib
Built a web app with Streamlit
Libraries used
pandas
nltk
pyspellchecker
scikit-learn
joblib
streamlit
Final result
Logistic Regression with bag of words gave the best result with 0.866 test accuracy.

How to run
First run the notebook to train the model and save the pkl files. Then start the app with the command below.

streamlit run app.py
