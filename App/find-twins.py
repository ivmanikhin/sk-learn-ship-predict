import mariadb
from os import sep
from constants import *


def extract_table_from_sql(additional_parameters):
    try:
        with mariadb.connect(host=HOST, user=USER, password=PASSWORD, port=PORT, database=DATABASE) as con:
            cur = con.cursor()
            cur.execute(f"SELECT {COL_NAMES} FROM {TABLE_NAME} WHERE ")
            df = pd.read_sql_query(f"select {COL_NAMES} from {TABLE_NAME} {additional_parameters}", con)
        df.columns = df_column_names
        return df
    except:
        print("Failed to read SQL")