"""Download the UCI Sentiment Labelled Sentences dataset and build SQLite.
If the download is unavailable, this uses the included starter_demo dataset instead.
"""
from io import BytesIO
from pathlib import Path
import sqlite3
import urllib.request
import zipfile
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "sentiment.db"
CSV_PATH = DATA_DIR / "reviews.csv"
SAMPLE_PATH = DATA_DIR / "sample_reviews.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"


def load_uci_dataset() -> pd.DataFrame:
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "RNN-Sentiment-Learning-Project/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    dataframes = []
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        for filename, source in [
            ("amazon_cells_labelled.txt", "amazon"),
            ("imdb_labelled.txt", "imdb"),
            ("yelp_labelled.txt", "yelp"),
        ]:
            matches = [name for name in archive.namelist() if name.endswith(filename)]
            if not matches:
                raise FileNotFoundError(f"{filename} was not found in the UCI archive.")
            with archive.open(matches[0]) as file:
                frame = pd.read_csv(
                    file, sep="\t", header=None, names=["text", "label"],
                    encoding="utf-8", on_bad_lines="error"
                )
            frame["source"] = source
            dataframes.append(frame)

    df = pd.concat(dataframes, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""].drop_duplicates(subset=["text"]).reset_index(drop=True)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Dataset should contain both labels 0 and 1.")
    return df[["text", "label", "source"]]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = load_uci_dataset()
        print(f"Downloaded official UCI dataset: {len(df)} rows.")
    except Exception as exc:
        print("Could not download the UCI dataset:", exc)
        print("Using the bundled starter sample instead (small; not suitable for serious accuracy claims).")
        df = pd.read_csv(SAMPLE_PATH)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("reviews", conn, if_exists="replace", index=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                predicted_sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    print(f"CSV saved: {CSV_PATH}")
    print(f"SQLite database ready: {DB_PATH}")
    print("Label mapping: 0 = negative, 1 = positive")
    print(df["label"].value_counts().sort_index().rename(index={0: "negative", 1: "positive"}))


if __name__ == "__main__":
    main()
