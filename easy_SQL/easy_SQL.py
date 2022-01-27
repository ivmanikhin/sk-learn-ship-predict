import sqlite3
import pandas as pd
import re
import os

DATA_TYPES = {
    "int": "Integer",
    "float": "Real",
    "str": "Text",
    "object": "Text"
}

SEP = os.sep

def new_column_to_sql(con, df, table_name):
    cur = con.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table_name})")
        db_columns = pd.DataFrame(cur.fetchall())[1].to_list()
        for col_name in list(df.columns.values):
            if col_name not in db_columns:
                print(f"New column: \"{col_name}\"")
                data_type_py = re.search("([a-zA-Z]+)", str(df.dtypes[col_name])).group(1)
                print(f"Python data type: \"{data_type_py}\"")
                data_type_sql = DATA_TYPES[data_type_py]
                print(f"SQL data type: \"{data_type_sql}\"")
                cur.execute(f"PRAGMA table_info({table_name})")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN `{col_name}` {data_type_sql}")
    except:
        pass


def extract_table_from_sql(table_name, sql_column_names, df_column_names, additional_parameters):
    try:
        with sqlite3.connect(f'data{SEP}ships.db') as con:
            df = pd.read_sql_query(f"select {sql_column_names} from {table_name} {additional_parameters}", con)
        df.columns = df_column_names
        return df
    except:
        print("Failed to read SQL")


def write_to_sql(df, table_name, if_exists="append"):
    con = sqlite3.connect(f'data{SEP}ships.db')
    new_column_to_sql(con, df, table_name)
    df.to_sql(name=table_name, con=con, if_exists=if_exists, index=False)
    con.close()

