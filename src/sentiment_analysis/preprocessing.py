import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:

    text = str(text).lower()

    # remove links
    text = re.sub(r"http\S+", "", text)

    # remove hashtags symbol
    text = re.sub(r"#", "", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)

    words = [w for w in words if w not in stop_words]

    return " ".join(words)