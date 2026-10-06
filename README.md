# EmotionAI

EmotionAI detects the emotion in a text sentence. The model reads a sentence like `"i feel so alone today"` and predicts one of six emotions:

**sadness, anger, joy, love, fear, surprise**

## Dataset

I used the **Emotions dataset for NLP** from Kaggle:

https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp-analysis

The dataset has `train.txt` with sentences and one emotion label for each sentence.

## What I Did

1. Loaded `train.txt` into a pandas dataframe
2. Cleaned the text with:
   - lower casing
   - punctuation removal
   - number removal
   - emoji removal
   - stopword removal
   - spelling correction with `pyspellchecker`
   - lemmatization with NLTK
3. Turned the emotion labels into numbers
4. Split the data into 80 percent train and 20 percent test
5. Turned the text into numbers with Bag of Words and TF-IDF
6. Trained Naive Bayes and Logistic Regression with sklearn pipelines
7. Compared all four results and picked the best one
8. Saved the final model with joblib
9. Built a web app with Streamlit

## Libraries Used

- pandas
- nltk
- pyspellchecker
- scikit-learn
- joblib
- streamlit

Install them with:

```bash
pip install pandas nltk pyspellchecker scikit-learn joblib streamlit
```

## Models Compared

| Vectorizer | Naive Bayes | Logistic Regression |
|------------|:-----------:|:-------------------:|
| Bag of Words | ✓ | ✓ (best) |
| TF-IDF | ✓ | ✓ |

## Final Result

**Logistic Regression with Bag of Words** gave the best result with **0.866 test accuracy**.

## How to Run

First, run the notebook to train the model and save the `.pkl` files. Then start the app with the command below:

```bash
streamlit run app.py
```

