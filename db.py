import sqlite3
import os

def create_db():
    db_file = 'predictions.db'
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT,
                predicted_class TEXT,
                probability REAL,
                timestamp TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                last_updated TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        ''')

        conn.commit()
        conn.close()
    else:
        print("База данных уже существует")
        
def add_user(name, email):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()

def add_prediction(user_id, image_path, predicted_class, probability, timestamp):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_id, image_path, predicted_class, probability, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, image_path, predicted_class, probability, timestamp))
    conn.commit()
    conn.close()

create_db()
