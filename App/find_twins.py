import mariadb
from App.constants import *


def extract_table_from_sql(where="imo_no in (9832717, 9864332)"):
    try:
        with mariadb.connect(host=HOST, user=USER, password=PASSWORD, port=PORT, database=DATABASE) as con:
            cur = con.cursor()
            cur.execute(f"SELECT {COL_NAMES} FROM {TABLE_NAME} WHERE {where}")
            raw_data = cur.fetchall()
            values = [dict(zip(DF_COL_NAMES, values)) for values in raw_data]


        return values
    except:
        print("Failed to read SQL")


