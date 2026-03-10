import pandas as pd
import joblib

from sklearn.metrics import classification_report

from sentiment_analysis.config import DATA_PATH, MODEL_PATH
from sentiment_analysis.preprocessing import clean_text


def evaluate():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["clean_text"] = df["Text"].apply(clean_text)

    print("Loading model...")
    model, vectorizer = joblib.load(MODEL_PATH)

    X = vectorizer.transform(df["clean_text"])
    y = df["Sentiment"]

    preds = model.predict(X)

    print("\nClassification Report\n")
    print(classification_report(y, preds))


if __name__ == "__main__":
    evaluate()