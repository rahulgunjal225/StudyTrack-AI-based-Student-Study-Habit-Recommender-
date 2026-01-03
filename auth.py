import bcrypt
from db import get_connection

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_user(username, email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (%s,%s,%s)",
        (username, email, hash_password(password))
    )
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cur.fetchone()
    conn.close()

    if user and check_password(password, user["password"]):
        return user
    return None

# 🔑 FORGOT PASSWORD / RESET PASSWORD FUNCTION
def reset_password(username, email, new_password):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # user verify (username + email)
    cur.execute(
        "SELECT * FROM users WHERE username=%s AND email=%s",
        (username, email)
    )
    user = cur.fetchone()

    if not user:
        conn.close()
        return False   # user not found

    # new password hash
    hashed = hash_password(new_password).decode()

    # update password
    cur.execute(
        "UPDATE users SET password=%s WHERE id=%s",
        (hashed, user["id"])
    )

    conn.commit()
    conn.close()
    return True
