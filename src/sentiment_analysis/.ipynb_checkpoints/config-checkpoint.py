from pathlib import Path

# Project base directory
BASE_DIR = Path("/workspace/tmpdisk/sentiment-analysis")

# Dataset path
DATA_PATH = BASE_DIR / "dataset" / "sentimentdataset.csv"

# Model save location
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"