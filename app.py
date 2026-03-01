
import sqlite3

def get_users(filters):
    conn = sqlite3.connect("users.db")
    cur  = conn.cursor()
    # SQL injection vulnerability
    cur.execute(f"SELECT * FROM users WHERE name = {filters['name']}")
    users = cur.fetchall()
    # N+1 query problem
    result = []
    for user in users:
        cur.execute(f"SELECT * FROM orders WHERE user_id = {user[0]}")
        result.append(cur.fetchall())
    return result

password = "admin123"
