import sqlite3
import pandas as pd
import pymysql
import mariadb
import numpy as np


def make_batches(list_of_something, batch_size=1000):
    structured_list = []
    num_of_sublists = int(round(len(list_of_something) / batch_size))
    for i in np.array_split(list_of_something, num_of_sublists):
        structured_list.append(list(i))
    return structured_list


with sqlite3.connect('data/ships.db') as con:
    df = pd.read_sql_query(f"select * from ships_details", con)
df["net_tonnage"] = df["net_tonnage"].str.replace(',', '.')
data = df.to_dict("records")
# print(data[:5])
columns = (", ".join(data[0].keys()))
strings = [f"({str(list(string.values()))[1:-1].replace('nan', 'NULL').replace('None', 'NULL')})" for string in data]
strings = [string.replace("''", "NULL") for string in strings]
print(strings[:3])
# str_values = [f"({', '.join([str(value) for value in string.values()])})" for string in data]
# print(', '.join(str_values[:3]))
# data_values = [f"({', '.join(row.values())})" for row in data]
# print(data_values[0])

string_batches = make_batches(strings)
# print(string_batches[30][585])


sql_queries = [f"INSERT INTO ships_details ({columns})\n VALUES {', '.join(string)}" for string in string_batches]


with mariadb.connect(host="ships-db.ceb9xxeumyfk.eu-central-1.rds.amazonaws.com", user="admin", password="ca78lo91ps23ck", port=3306, database="ships") as remote_con:
    print("connected")
    cur = remote_con.cursor()
    _ = 0
    for sql_query in sql_queries:
        _ += 1
        cur.execute(sql_query)
        print(f"{_} queries sent")
    cur.execute("COMMIT")
