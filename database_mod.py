import credentials
import pyodbc

def create_table1():
	conn = pyodbc.connect('DRIVER='+credentials.driver+';SERVER='+credentials.server
		+';DATABASE='+credentials.database+';UID='+credentials.username+';PWD='+credentials.password)
	cur = conn.cursor()
	if cur.tables(table='TimesNoted', tableType='TABLE').fetchone():
		print("",end= "") # exists 
	else:
		cur.execute("CREATE TABLE TimesNoted(id int NOT NULL IDENTITY(1,1) primary key, entry_time varchar(100), exit_time varchar(100))" ) 	
	conn.commit()
	conn.close()

def insert_data1(Entry_Time, Exit_Time):
	conn = pyodbc.connect('DRIVER='+credentials.driver+';SERVER='+credentials.server
		+';DATABASE='+credentials.database+';UID='+credentials.username+';PWD='+credentials.password)
	cur = conn.cursor()
	cur.execute("INSERT INTO TimesNoted(entry_time,exit_time) VALUES (?, ?)", (Entry_Time, Exit_Time))
	conn.commit()
	conn.close()