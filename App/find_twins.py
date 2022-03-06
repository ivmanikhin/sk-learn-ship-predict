import sqlite3

def extract_table_from_sql(table_name, sql_column_names, df_column_names, additional_parameters):
    try:
        with sqlite3.connect('/content/drive/MyDrive/Colab Notebooks/ships.db') as con:
            df = pd.read_sql_query(f"select {sql_column_names} from {table_name} {additional_parameters}", con)
        df.columns = df_column_names
        return df
    except:
        print("Failed to read SQL")