import pandas as pd
import joblib

from sklearn.metrics import classification_report

df = pd.read_csv("../data/processed/cleaned_tickets.csv")

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

model = joblib.load("../models/ticket_classifier.pkl")

X = vectorizer.transform(df['cleaned_text'])

y = df['ticket_type']

y_pred = model.predict(X)

print(classification_report(y, y_pred))