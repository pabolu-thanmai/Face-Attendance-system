import sqlite3
import os

# Folder to store databases
DATABASE_FOLDER = "databases"

# List of branch databases
branches = ["EEE", "IT", "CSM", "CSBS"]

# Ensure the database folder exists
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

# Function to create a table in the specified database
def create_table(database_name, table_name):
    db_path = os.path.join(DATABASE_FOLDER, f"{database_name}.db")  # Store inside 'databases' folder
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            roll_number TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Table '{table_name}' created successfully in {database_name}.db")

# Creating databases and default tables

create_table(branches[2], "Y22ECC")  # Creating a default table

print("All databases and tables created successfully inside 'databases' folder.")
