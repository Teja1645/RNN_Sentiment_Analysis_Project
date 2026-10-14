"""Streamlit interface for the RNN sentiment analysis project."""
from pathlib import Path
import json
import sqlite3
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "sentiment.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_rnn.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"

st.set_page_config(page_title="RNN Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 RNN Sentiment Analyzer")
st.caption("A beginner-friendly Recurrent Neural Network project with a SQLite review database.")

if not all(p.exists() for p in [MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH]):
    st.error("The trained model files are missing. Run the setup and training commands locally, then push the generated files to GitHub.")
    st.code("python prepare_database.py\npython train_model.py", language="powershell")
    st.stop()

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return model, tokenizer, config


def ensure_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
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


def save_prediction(text, label, confidence):
    ensure_database()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (review_text, predicted_sentiment, confidence) VALUES (?, ?, ?)",
            (text, label, float(confidence)),
        )
        conn.commit()


model, tokenizer, config = load_assets()
ensure_database()

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Analyze a review")
    review_text = st.text_area(
        "Enter a product, movie, restaurant, or service review",
        placeholder="Example: The service was friendly and the food was delicious.",
        height=150,
        max_chars=5000,
    )
    predict_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        cleaned = review_text.strip()
        if not cleaned:
            st.warning("Please enter a review first.")
        else:
            sequences = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequences, maxlen=int(config["max_len"]), padding="post", truncating="post")
            prob_positive = float(model.predict(padded, verbose=0)[0][0])
            label = "Positive" if prob_positive >= 0.5 else "Negative"
            confidence = prob_positive if label == "Positive" else 1.0 - prob_positive
            save_prediction(cleaned, label, confidence)
            st.session_state["last_result"] = {
                "text": cleaned,
                "label": label,
                "confidence": confidence,
                "prob_positive": prob_positive,
            }

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        if result["label"] == "Positive":
            st.success(f"Predicted sentiment: {result['label']}")
        else:
            st.warning(f"Predicted sentiment: {result['label']}")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.progress(max(0.0, min(1.0, result["prob_positive"])))
        st.caption(f"Model estimated positive probability: {result['prob_positive']:.1%}")

with right:
    st.subheader("Project information")
    st.markdown("""
    **Model:** Embedding + SimpleRNN + Dense layers  
    **Dataset:** UCI Sentiment Labelled Sentences  
    **Database:** SQLite (`data/sentiment.db`)  
    **Labels:** `0 = Negative`, `1 = Positive`
    """)
    st.metric("Held-out test accuracy", f"{float(config.get('test_accuracy', 0)):.1%}")
    st.caption("This is a learning demo. Predictions can be wrong, especially for sarcasm, mixed opinions, or text unlike the training data.")

st.divider()
st.subheader("Database dashboard")
with sqlite3.connect(DB_PATH) as conn:
    try:
        counts = pd.read_sql_query("SELECT label, COUNT(*) AS total FROM reviews GROUP BY label ORDER BY label", conn)
    except Exception:
        counts = pd.DataFrame(columns=["label", "total"])
    history = pd.read_sql_query(
        "SELECT created_at, review_text, predicted_sentiment, confidence FROM predictions ORDER BY id DESC LIMIT 20",
        conn,
    )

if not counts.empty:
    counts["sentiment"] = counts["label"].map({0: "Negative", 1: "Positive"})
    cols = st.columns(3)
    cols[0].metric("Training reviews in database", int(counts["total"].sum()))
    cols[1].metric("Negative examples", int(counts.loc[counts["label"] == 0, "total"].sum()))
    cols[2].metric("Positive examples", int(counts.loc[counts["label"] == 1, "total"].sum()))
else:
    st.info("Review dataset not found yet. Run python prepare_database.py")

with st.expander("View the latest saved predictions"):
    if history.empty:
        st.write("No predictions saved yet. Analyze a review above.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)

st.caption("Dataset citation: Kotzias et al. (2015), UCI Machine Learning Repository, Sentiment Labelled Sentences, DOI: 10.24432/C57604, CC BY 4.0.")
