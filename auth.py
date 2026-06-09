import bcrypt
from db import get_connection


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_user(username, email, password):
    conn = get_connection()
    cur = conn.cursor()

    hashed = hash_password(password).decode()

    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, hashed)
    )

    conn.commit()
    conn.close()


def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    row = cur.fetchone()
    conn.close()

    if row:
        user = dict(row)

        if check_password(password, user["password"]):
            return user

    return None


def reset_password(username, email, new_password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=? AND email=?",
        (username, email)
    )

    user = cur.fetchone()

    if not user:
        conn.close()
        return False

    hashed = hash_password(new_password).decode()

    cur.execute(
        "UPDATE users SET password=? WHERE id=?",
        (hashed, user["id"])
    )

    conn.commit()
    conn.close()

    return True
