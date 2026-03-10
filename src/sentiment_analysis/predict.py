import joblib

from sentiment_analysis.preprocessing import clean_text
from sentiment_analysis.config import MODEL_PATH


def predict(text: str):

    model, vectorizer = joblib.load(MODEL_PATH)

    text = clean_text(text)

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)

    return prediction[0]


if __name__ == "__main__":

    while True:

        text = input("\nEnter text (or type exit): ")

        if text.lower() == "exit":
            break

        result = predict(text)

        print("Predicted Sentiment:", result)