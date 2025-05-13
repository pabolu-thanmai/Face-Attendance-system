import sqlite3

# Database name
db_name = "databases/attendance.db"

# Table names
table_names = ["Y22CDA", "Y22CDB", "Y22CDC"]

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create tables
for table in table_names:
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        roll_number TEXT PRIMARY KEY,
        embedding BLOB
    );
    """
    cursor.execute(create_table_query)
    print(f"Table {table} created successfully.")

# Commit and close connection
conn.commit()
conn.close()
print("Database setup completed.")
