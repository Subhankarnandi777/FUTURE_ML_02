import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../data/processed/cleaned_tickets.csv")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['cleaned_text'])

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("Vectorizer Saved Successfully")