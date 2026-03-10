import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sentiment_analysis.config import DATA_PATH, MODEL_PATH
from sentiment_analysis.preprocessing import clean_text


def train():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    # normalize labels
    df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
    
    positive_words = [
        "positive","joy","excited","excitement","happy","happiness",
        "contentment","relief","gratitude","love","optimism","pride",
        "admiration","enthusiasm","hope"
    ]
    
    negative_words = [
        "negative","anger","sad","sadness","fear","disgust",
        "boredom","frustration","annoyance","disappointment",
        "indifference"
    ]
    
    def map_sentiment(label):
    
        if label in positive_words:
            return "Positive"
    
        elif label in negative_words:
            return "Negative"
    
        else:
            return "Neutral"
    
    df["Sentiment"] = df["Sentiment"].apply(map_sentiment)
    # keep only required columns
    df = df[["Text", "Sentiment"]]
    print(df["Sentiment"].value_counts())
    # remove missing values
    df = df.dropna()
    print(df["Sentiment"].value_counts())
    
    print("Cleaning text...")
    df["clean_text"] = df["Text"].apply(clean_text)

    X = df["clean_text"]
    y = df["Sentiment"]

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training model...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    
    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5],
        "solver": ["liblinear", "lbfgs"]
    }
    
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    
    print("Best parameters:", grid.best_params_)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(classification_report(y_test, preds))
    
    print(f"Model Accuracy: {acc:.3f}")

    print("Saving model...")
    joblib.dump((model, vectorizer), MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()