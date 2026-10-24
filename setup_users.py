import sqlite3
from werkzeug.security import generate_password_hash

print("1. Connecting to database...")
conn = sqlite3.connect('data/nurturenode.db')
cursor = conn.cursor()

print("2. Creating users table...")
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
''')

# Clear existing users to avoid duplicates if run multiple times
cursor.execute('DELETE FROM users')

print("3. Generating secure passwords and inserting accounts...")
users = [
    ('admin', generate_password_hash('admin123'), 'Director'),
    ('staff', generate_password_hash('staff123'), 'Caregiver')
]
cursor.executemany('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', users)

conn.commit()
conn.close()
print("4. Success: Created accounts 'admin' (pass: admin123) and 'staff' (pass: staff123).")