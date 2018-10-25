import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
conn = sqlite3.connect("Database.db")
dataframe = pd.read_sql_query("SELECT * from Database", conn, index_col = 'ID')
conn.close()

# print(dataframe)
entry_time = []
exit_time = []
time_difference = []
entries = []
time2 = []
entry_time_temp = dataframe['Entry_Time']
exit_time_temp = dataframe['Exit_Time']

print(len(entry_time_temp))
for i in range(0, len(entry_time_temp)):
    entry_time.append(datetime.strptime(entry_time_temp.iloc[i], '%Y-%m-%d %H:%M:%S.%f'))
    exit_time.append(datetime.strptime(exit_time_temp.iloc[i], '%Y-%m-%d %H:%M:%S.%f'))

def time_diff(entry_time, exit_time):
    elapsedTime = exit_time - entry_time
    elapsedTime = divmod(elapsedTime.total_seconds(), 60)
    # print(elapsedTime)
    return elapsedTime

for i in range(0, len(entry_time)):
    entries.append(i)
    time_difference.append(time_diff(entry_time[i], exit_time[i]))
    time2.append(time_difference[i][1])

plt.title('Daily Analysis')
plt.xlabel('Instance')
plt.ylabel('Time Period of Animal Detection')
plt.scatter(entries, time2)
plt.show()
# print(type(time_difference[0]))
