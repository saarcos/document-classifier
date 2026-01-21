import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import re 
DATA_PATH = "data/company-document-text.csv"
MODEL_PATH = "artifacts/document_classifier.joblib"

LABEL_KEYWORDS = [
    "invoice",
    "purchase orders",
    "report",
    "order id"
]
def clean_text(text: str) -> str:
    text = text.lower()

    # Remove label-leaking keywords
    for kw in LABEL_KEYWORDS:
        text = text.replace(kw, "")

    # Remove extra whitespace and non-alphanumeric chars
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_data(path: str):
    df = pd.read_csv(path)
    df = df[["text", "label"]]
    df["text"] = df["text"].astype(str).apply(clean_text)
    return df


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=3
        )),
        ("clf", LinearSVC())
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
