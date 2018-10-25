import sqlite3
from datetime import datetime

def create_table():
	conn = sqlite3.connect("Database.db")
	cur = conn.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS Database(ID INTEGER PRIMARY KEY, Entry_Time timestamp, Exit_Time timestamp)")
	conn.commit()
	conn.close()

def insert_data(Entry_Time, Exit_Time):
	conn = sqlite3.connect("Database.db")
	cur = conn.cursor()
	cur.execute("INSERT INTO Database VALUES (NULL, ?, ?)", (Entry_Time, Exit_Time))
	conn.commit()
	conn.close()

def update(id, Entry_Time, Exit_Time):
	conn = sqlite3.connect("Database.db")
	cur = conn.cursor()
	cur.execute("UPDATE Database SET Entry_Time = ?, Exit_Time = ? where id = ?", (Entry_Time, Exit_Time, id))
	conn.commit()
	conn.close()

def get_id():
	conn = sqlite3.connect("Database.db")
	cur = conn.cursor()
	cur.execute("SELECT max(id) as id FROM Database")
	rows = cur.fetchall()
	conn.close()
	return rows[0][0]
