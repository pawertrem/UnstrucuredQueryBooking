from flask import Flask, redirect, session, request
import uuid
import random
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'c9f651cdcd7893a508869adbca17ccb9222aad0d259540c11d1a539b7deb5dd9'
app.config['SESSION_COOKIE_NAME'] = "session"
app.config['SESSION_COOKIE_SECURE'] = False


def init_db():
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_variants (
            user_id TEXT PRIMARY KEY,
            variant TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_user_action(user_id, action):
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO user_logs (user_id, action) VALUES (?, ?)',
        (user_id, action)
    )
    conn.commit()
    conn.close()

@app.route('/')
def route_user():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_id = session['user_id']

    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT variant FROM user_variants WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()

    if row:
        variant = row[0]
    else:
        variant = random.choice(['A', 'B'])
        cursor.execute('INSERT INTO user_variants (user_id, variant) VALUES (?, ?)', (user_id, variant))
        conn.commit()

    conn.close()

    session['variant'] = variant
    log_user_action(user_id, f"Зашел на сайт, вариант {variant}")

    if variant == 'A':
        return redirect('http://158.160.149.234:5001/')
    else:
        return redirect('http://158.160.149.234:5002/')

@app.route('/debug')
def debug():
    user_id = session.get('user_id')
    variant = session.get('variant')
    return f"""
        user_id: {user_id}<br>
        variant: {variant}<br>
        session: {dict(session)}
    """

if __name__ == '__main__':
    app.run(port=5000, debug=True)