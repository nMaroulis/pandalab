import sqlite3
from settings.settings import TRAINING_REPORT_DB_PATH


def init_training_db():
    conn = sqlite3.connect(TRAINING_REPORT_DB_PATH)  # Connect to the database, create it if it doesn't exist
    c = conn.cursor()  # Create a cursor object
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_report
        (pid INTEGER PRIMARY KEY,
        training_status TEXT);
    ''')
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def create_new_training(pid):
    conn = sqlite3.connect(TRAINING_REPORT_DB_PATH)  # Create a cursor object
    c = conn.cursor()
    # Insert a new row of data
    c.execute("INSERT INTO training_report (pid, training_status) VALUES (?, ?)", (pid,'Initializing new Training Process...'))
    conn.commit()
    conn.close()
    return


def get_training_progress(pid):
    conn = sqlite3.connect(TRAINING_REPORT_DB_PATH)  # Create a cursor object
    c = conn.cursor()
    c.execute("SELECT training_status FROM training_report WHERE pid=?", (pid,))
    result = c.fetchone()
    print(result)
    conn.close()
    if result:
        return result[0]
    else:
        return None


def update_training_status(status_str, pid):
    conn = sqlite3.connect(TRAINING_REPORT_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE training_report SET training_status = ? WHERE pid = ?",(status_str, pid))
    conn.commit()
    conn.close()
    return

def fetch_all():
    conn = sqlite3.connect(TRAINING_REPORT_DB_PATH)  # Create a cursor object
    c = conn.cursor()
    c.execute("SELECT * FROM training_report")
    result = c.fetchall()
    conn.close()
    return result


