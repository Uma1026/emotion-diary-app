import sqlite3
import pandas as pd

# CREATE DATABASE
def create_database():

    conn = sqlite3.connect("emotion_diary.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diary_entries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        diary_text TEXT,
        predicted_emotion TEXT,
        confidence REAL,
        eiv REAL,
        affirmation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# SAVE ENTRY
def save_entry(user_id, text, emotion, confidence, eiv, affirmation):

    conn = sqlite3.connect("emotion_diary.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO diary_entries
    (user_id, diary_text, predicted_emotion, confidence, eiv, affirmation)
    VALUES (?, ?, ?, ?, ?, ?)
    """,(user_id,text,emotion,confidence,eiv,affirmation))

    conn.commit()
    conn.close()


# LOAD USER DATA
def load_user_data(user_id):

    conn = sqlite3.connect("emotion_diary.db")

    query = """
    SELECT * FROM diary_entries
    WHERE user_id = ?
    ORDER BY datetime(created_at) 
    """

    df = pd.read_sql_query(query,conn,params=(user_id,))
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    return df