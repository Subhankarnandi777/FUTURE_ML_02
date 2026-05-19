import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/processed/cleaned_tickets.csv")

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

X = vectorizer.transform(df['cleaned_text'])

y = df['ticket_type']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(model, "../models/ticket_classifier.pkl")

print("Model Saved Successfully")