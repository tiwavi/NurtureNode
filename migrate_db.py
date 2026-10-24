import pandas as pd
import sqlite3
import os

print("1. Initializing SQLite Database connection...")
# Create a new database file inside the data folder
db_path = os.path.join('data', 'nurturenode.db')
conn = sqlite3.connect(db_path)

print("2. Loading CSV datasets...")
inv_df = pd.read_csv('data/supply_chain_data.csv')
acad_df = pd.read_csv('data/Student_performance_data _.csv')
vol_df = pd.read_csv('data/cleaned_volunteer.csv')

print("3. Migrating data to relational tables...")
# The to_sql function automatically creates the table schema based on the dataframe
inv_df.to_sql('inventory', conn, if_exists='replace', index=False)
acad_df.to_sql('students', conn, if_exists='replace', index=False)
vol_df.to_sql('volunteers', conn, if_exists='replace', index=False)

conn.close()
print(f"4. Success: Database migrated and saved to {db_path}")