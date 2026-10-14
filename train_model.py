"""Train a SimpleRNN sentiment classifier using reviews stored in SQLite."""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "sentiment.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_rnn.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"
MAX_WORDS = 10000
MAX_LEN = 100
SEED = 42


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError("Database is missing. Run: python prepare_database.py")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT text, label FROM reviews", conn)

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""].drop_duplicates(subset=["text"])
    if len(df) < 20 or df["label"].nunique() != 2:
        raise ValueError("Need at least 20 reviews and both sentiment labels to train.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values.astype("float32"),
        test_size=0.20,
        random_state=SEED,
        stratify=df["label"].values,
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    train_seq = tokenizer.texts_to_sequences(x_train)
    test_seq = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    model = Sequential([
        Input(shape=(MAX_LEN,), dtype="int32"),
        Embedding(input_dim=MAX_WORDS, output_dim=64),
        SimpleRNN(32),
        Dropout(0.25),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        x_train_pad,
        y_train,
        validation_split=0.15,
        epochs=12,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    probabilities = model.predict(x_test_pad, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test.astype(int), predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    (MODEL_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    config = {"max_words": MAX_WORDS, "max_len": MAX_LEN, "test_accuracy": accuracy, "training_rows": int(len(x_train)), "test_rows": int(len(x_test)), "seed": SEED}
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nTEST ACCURACY:", f"{accuracy:.3f}")
    print("\nCLASSIFICATION REPORT:\n", classification_report(y_test.astype(int), predictions, target_names=["negative", "positive"], zero_division=0))
    print("CONFUSION MATRIX:\n", confusion_matrix(y_test.astype(int), predictions))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_PATH}")
    print(f"Metrics saved to: {CONFIG_PATH}")
    if len(df) < 1000:
        print("WARNING: Training data is small. Run prepare_database.py with internet access to fetch the full UCI dataset.")


if __name__ == "__main__":
    main()
