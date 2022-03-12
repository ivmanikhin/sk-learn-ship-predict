import pymysql
from App.constants import *
from App.some_text import *


def extract_table_from_sql(where):
    try:
        with pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=PORT, database=DATABASE) as con:
            cur = con.cursor()
            cur.execute(f"SELECT {COL_NAMES} FROM {TABLE_NAME} WHERE {where}")
            raw_data = cur.fetchall()
            values = [dict(zip(DF_COL_NAMES, values)) for values in raw_data]
        return values
    except:
        print("Failed to read SQL")


def get_twins(params, delta=.05):
    where = f"ice_class = {1 if params['ice_class'] == 'on' else 0} and " \
            f"dynpos = {1 if params['dynpos'] == 'on' else 0} and uni_type = '{params['uni_type']}' " \
            f"and deadweight between {int(params['deadweight']) * (1 - delta)} and {int(params['deadweight']) * (1 + delta)} " \
            f"and speed between {float(params['speed']) * (1 - delta)} and {float(params['speed']) * (1 + delta)} "
    # for key in params.keys():
    #     where += f"{key} between {(1 - delta) * params[key]} and {(1 + delta) * params[key]} and "
    # where = where[:-4]
    print(where)
    return extract_table_from_sql(where=where)
