import streamlit as st
import torch
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import random
import math
import os

from database import create_database, save_entry, load_user_data
create_database()

from auth import create_user_table, register_user, login_user
create_user_table()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Aware Diary",
    page_icon="🧠",
    layout="wide"
)

# ---------------- UI CSS ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
}

/* TITLE */
.title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 6px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

/* TEXT AREA */
textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(to right, #ff4b2b, #ff416c);
}

/* SUCCESS BOX */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_all():
    model = RobertaForSequenceClassification.from_pretrained("uma0826/emotion-diary-model")
    tokenizer = RobertaTokenizer.from_pretrained("uma0826/emotion-diary-model")

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, label_encoder, metrics, device

model, tokenizer, label_encoder, metrics, device = load_all()

# ---------------- SESSION STATE ----------------
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# ---------------- SA FUNCTION ----------------
emotion_states = {
    "anger": 0,
    "sadness": 1,
    "fear": 2,
    "neutral": 3,
    "hopeful": 4,
    "joy": 5,
    "love": 5
}

affirmation_levels = {
    0: [
        "I know things feel overwhelming right now. Take one slow breath.",
        "It’s okay to feel angry — your emotions are valid."
    ],
    1: [
        "You are stronger than this moment.",
        "Bad days do not define your life."
    ],
    2: [
        "You are safe. This feeling will pass.",
        "Courage grows quietly inside you."
    ],
    3: [
        "You are doing enough. Keep moving forward.",
        "Stay calm. Things will settle."
    ],
    4: [
        "Good things are finding their way to you.",
        "Believe in the progress you are making."
    ],
    5: [
        "Your energy is beautiful — keep shining!",
        "Hold onto this happiness!"
    ]
}

def simulated_annealing_affirmation(emotion, eiv):
    current_state = emotion_states.get(emotion, 3)
    temperature = max(1, eiv)

    for _ in range(3):
        neighbor = min(5, current_state + random.choice([0,1]))
        delta = neighbor - current_state

        if delta > 0:
            current_state = neighbor
        else:
            probability = math.exp(delta / temperature)
            if random.random() < probability:
                current_state = neighbor

        temperature *= 0.5

    return random.choice(affirmation_levels[current_state])

# ---------------- PREDICTION ----------------
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    emotion = label_encoder.inverse_transform([pred.item()])[0]
    confidence_pct = round(confidence.item() * 100, 2)

    if confidence_pct < 60:
        emotion = "neutral"

    emotion_base = {
        "anger": 9,
        "fear": 8,
        "sadness": 7,
        "disgust": 6,
        "surprise": 5,
        "neutral": 4,
        "joy": 2,
        "love": 1
    }

    base = emotion_base.get(emotion, 5)
    eiv = round(base + (confidence.item()*2 - 1), 2)
    eiv = max(1, min(eiv, 10))

    affirmation = simulated_annealing_affirmation(emotion, eiv)

    st.session_state.emotion_history.append((emotion, eiv))

    return emotion, confidence_pct, eiv, affirmation

# ---------- Registration / Login ----------
menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if menu == "Register":

    st.title("📝 Create Account")

    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=10, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    address = st.text_input("Address")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        success = register_user(name, age, gender, address, email, password)

        if success:
            st.success("Account created successfully! Go to Login.")
        else:
            st.error("User already exists!")

elif menu == "Login":

    st.title("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(email, password)

        if user:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

# ---------------- MAIN UI ----------------
if st.session_state.get("logged_in"):

    st.markdown('<div class="title">🧠 Emotion Aware Diary</div>', unsafe_allow_html=True)

    user_id = st.session_state.get("user_email")

    if user_id:
        df = load_user_data(user_id)
    else:
        df = pd.DataFrame()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    text = st.text_area("Write your thoughts", height=150)

    if st.button("✨ Analyze Emotion"):

        if not st.session_state.get("logged_in"):
            st.warning("Login required")

        elif text.strip() == "":
            st.warning("Write something")

        else:
            emotion, confidence, eiv, affirmation = predict(text)

            save_entry(user_id, text, emotion, confidence, eiv, affirmation)

            st.markdown(f"""
            <div class="card">
                <h3>Emotion: {emotion.upper()}</h3>
                <p>Confidence: {confidence}%</p>
                <p>EIV: {eiv}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence/100)

            st.markdown(f"""
            <div class="card" style="background:#2ecc71;">
                {affirmation}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- ACCURACY GRAPH --------
    st.subheader("📈 Model Accuracy over Epochs")

    if st.button("Show Accuracy vs Epoch Graph"):

        if os.path.exists("training_history.json"):
            with open("training_history.json", "r") as f:
                history_data = json.load(f)

            epochs = history_data["epochs"]
            val_acc = history_data["val_accuracy"]

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(epochs, val_acc, marker="o")

            best_idx = np.argmax(val_acc)
            ax.scatter(epochs[best_idx], val_acc[best_idx])

            st.pyplot(fig)
        else:
            st.warning("training_history.json not found")

    # -----------------load user data-------------------
    user_id = st.session_state.get("user_email", "")

    if user_id.strip() != "":
        df = load_user_data(user_id)
    else:
        df = pd.DataFrame()

    # ----------- COOLING CURVE GRAPH --------------
    st.subheader("🔥 Cooling Curve (Emotional Recovery)")

    if st.button("Show Cooling Curve"):

        if df.empty or "eiv" not in df.columns:
            st.info("No data available.")
        else:
            days = list(range(1, len(df)+1))
            eiv_values = df["eiv"].tolist()

            fig, ax = plt.subplots(figsize=(7,4))

            ax.plot(days, eiv_values, marker="o")
            ax.set_xlabel("Days")
            ax.set_ylabel("Emotional Intensity")
            ax.set_title("Cooling Curve (Simulated Annealing)")
            ax.grid(True)

            st.pyplot(fig)

            if len(eiv_values) > 1:
                if eiv_values[-1] < eiv_values[0]:
                    st.success("Your emotional intensity is improving 📉")
                else:
                    st.warning("Emotional intensity increased — keep reflecting 💭")

    # ---------------- LOG BOOK -----------------
    st.subheader("📘 Daily Emotional Logbook")

    if st.button("Show Logbook"):

        if df.empty:
            st.info("No entries yet.")
        else:
            required_cols = ["created_at", "predicted_emotion", "eiv", "affirmation"]

            if all(col in df.columns for col in required_cols):
                display_df = df[required_cols].copy()
                display_df.columns = ["Date", "Emotion", "EIV", "Coping Strategy"]
                st.dataframe(display_df)
            else:
                st.error("Missing columns in database.")

    # ---------------- EIV GRAPH ----------------
    st.subheader("📈 Emotion Intensity Trends")

    if st.button("Show EIV Trend"):

        if df.empty or "predicted_emotion" not in df.columns:
            st.info("No entries yet.")
        else:
            fig, ax = plt.subplots(figsize=(8,4))

            df.groupby("predicted_emotion")["eiv"].mean().plot(
                kind="line",
                marker="o",
                ax=ax
            )

            ax.set_xlabel("Emotion")
            ax.set_ylabel("Average EIV")
            ax.set_title("Emotion vs Emotion Intensity Value")
            ax.grid(True)

            st.pyplot(fig)

    # ---------------- TIME BASED ANALYSIS ----------------
    st.subheader("📊 Emotional Progress Analysis")

    analysis_type = st.selectbox(
        "Choose analysis type",
        ["Daily", "Weekly", "Monthly"]
    )

    if st.button("Generate Analysis"):

        if df.empty or "created_at" not in df.columns:
            st.info("No data available.")
        
        else:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            df = df.dropna(subset=["created_at"])

            if analysis_type == "Daily":

                df["date"] = df["created_at"].dt.date
                daily = df.groupby("date")["eiv"].mean()

                if len(daily) <= 1:
                    st.info("Add more diary entries to see trend.")

                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(daily.index, daily.values, marker="o")

                plt.xticks(rotation=45)

                ax.set_xlabel("Date")
                ax.set_ylabel("EIV")
                ax.set_title("Daily Emotional Trend")
                ax.grid(True)

                st.pyplot(fig)

            elif analysis_type == "Weekly":

                df["week"] = df["created_at"].dt.to_period("W")
                weekly = df.groupby("week")["eiv"].mean()

                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(weekly.index.astype(str), weekly.values, marker="o")

                plt.xticks(rotation=45)

                ax.set_xlabel("Week")
                ax.set_ylabel("EIV")
                ax.set_title("Weekly Emotional Trend")
                ax.grid(True)

                st.pyplot(fig)

            else:

                df["month"] = df["created_at"].dt.to_period("M")
                monthly = df.groupby("month")["eiv"].mean()

                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(monthly.index.astype(str), monthly.values, marker="o")

                plt.xticks(rotation=45)

                ax.set_xlabel("Month")
                ax.set_ylabel("EIV")
                ax.set_title("Monthly Emotional Trend")
                ax.grid(True)

                st.pyplot(fig)
