import sqlite3

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


def register_user(name, age, gender, address, email, password):
    conn = sqlite3.connect("emotion_diary.db")
    c = conn.cursor()

    try:
        c.execute("""
        INSERT INTO users (name, age, gender, address, email, password)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (name, age, gender, address, email, password))

        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def login_user(email, password):
    conn = sqlite3.connect("emotion_diary.db")
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))

    user = c.fetchone()
    conn.close()

    return user