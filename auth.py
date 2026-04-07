import sqlite3
import hashlib

# ---------------- PASSWORD HASH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------- CREATE TABLE ----------------
def create_user_table():
    conn = sqlite3.connect("emotion_diary.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        address TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


# ---------------- REGISTER ----------------
def register_user(name, age, gender, address, email, password):
    conn = sqlite3.connect("emotion_diary.db")
    c = conn.cursor()

    hashed_password = hash_password(password)

    try:
        c.execute("""
        INSERT INTO users (name, age, gender, address, email, password)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (name, age, gender, address, email, hashed_password))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# ---------------- LOGIN ----------------
def login_user(email, password):
    conn = sqlite3.connect("emotion_diary.db")
    c = conn.cursor()

    hashed_password = hash_password(password)

    c.execute("""
    SELECT * FROM users 
    WHERE email=? AND password=?
    """, (email, hashed_password))

    user = c.fetchone()
    conn.close()

    return user