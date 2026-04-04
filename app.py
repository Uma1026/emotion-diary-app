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

# ---------------- AFFIRMATIONS ----------------
affirmations = {
    "anger": "Pause. Breathe. You are in control 🌿",
    "fear": "You are stronger than this moment 💪",
    "sadness": "Your feelings are valid. Healing takes time 💙",
    "love": "Love is your strength ❤️",
    "surprise": "Unexpected moments bring growth 🌸"
}

emotion_colors = {
    "anger": "#FF4B4B",
    "fear": "#A66DD4",
    "sadness": "#4A90E2",
    "love": "#FF69B4",
    "surprise": "#FFA500"
}

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
    "joy": 5
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
        "You are doing enough. Keep moving forward."
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
    # LOW CONFIDENCE CONTROL
    if confidence_pct < 60:
        emotion = "neutral"
    # -------- Emotion based EIV calculation --------
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
    # keep EIV inside 1–10
    eiv = max(1, min(eiv, 10))

    # --- SA INTEGRATION ---
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

# ---------------- UI ----------------
st.success(f"Logged in as: {st.session_state.get('user_email')}")
if st.session_state.get("logged_in"):

    # ---------- PREMIUM HEADER ----------
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to right, #1e3c72, #2a5298);
            color: white;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🧠 Emotion Aware Diary</div>', unsafe_allow_html=True)

    # ---------- INPUT CARD ----------
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        user_id = st.session_state.get("user_email")

        text = st.text_area("Write your thoughts", height=150)

        if st.button("✨ Analyze Emotion"):

            if not user_id:
                st.warning("Login required")

            elif text.strip() == "":
                st.warning("Write something")

            else:
                emotion, confidence, eiv, affirmation = predict(text)

                save_entry(user_id, text, emotion, confidence, eiv, affirmation)

                # RESULT CARD
                st.markdown(f"""
                <div class="card">
                    <h3>🎭 Emotion: {emotion.upper()}</h3>
                    <p>Confidence: {confidence}%</p>
                    <p>EIV: {eiv}</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(confidence/100)

                # AFFIRMATION CARD
                st.markdown(f"""
                <div class="card" style="background-color:#2ecc71; color:white;">
                    {affirmation}
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
# --------------- accuracy vs epoch ------------
# Load user data
user_id = st.session_state.get("user_email")

if user_id:
    df = load_user_data(user_id)
else:
    df = pd.DataFrame()

st.subheader("📈 Model Accuracy over Epochs")

if st.button("Show Accuracy vs Epoch Graph"):

    # 🟢 NEW → load real training data (from training_history.json)
    with open("training_history.json", "r") as f:
        history_data = json.load(f)

    epochs = history_data["epochs"]
    val_acc = history_data["val_accuracy"]

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(epochs, val_acc, marker="o")

    # 🟢 NEW → auto best epoch detection
    best_idx = np.argmax(val_acc)
    best_epoch = epochs[best_idx]
    best_acc = val_acc[best_idx]

    ax.scatter(best_epoch, best_acc)
    ax.text(best_epoch, best_acc, " Best", fontsize=10)

# ----------- COOLING CURVE GRAPH --------------
st.subheader("🔥 Cooling Curve (Emotional Recovery)")

if st.button("Show Cooling Curve"):

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
        display_df = df[[
            "created_at",
            "predicted_emotion",
            "eiv",
            "affirmation"
        ]].copy()

        display_df.columns = ["Date", "Emotion", "EIV", "Coping Strategy"]

        st.dataframe(display_df)


# -----------------load user data-------------------       
if user_id.strip() != "":
    df = load_user_data(user_id)
else:
    df = pd.DataFrame()
# ---------------- EIV GRAPH (EMOTION VS EIV) ----------------
st.subheader("📈 Emotion Intensity Trends")

if st.button("Show EIV Trend"):

    if df.empty:
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

    if df.empty:
        st.info("No data available.")
    
    else:

        df["created_at"] = pd.to_datetime(df["created_at"])

        if analysis_type == "Daily":

            df["date"] = df["created_at"].dt.date
            daily = df.groupby("date")["eiv"].mean()
            # -----single entry check-----
            if len(daily) == 1:
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

