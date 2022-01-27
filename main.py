from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from easy_SQL.easy_SQL import *
from scipy import stats

SEP = os.sep

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', None)


dataset = extract_table_from_sql(table_name="main_ships_data_raw",
                                 sql_column_names="type, loa, boa, draft, speed, power, year",
                                 df_column_names=["type", "loa", "boa", "draft", "speed", "power", "year"],
                                 additional_parameters="where year > 1985 and loa > 30 and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 5 and 100 and power between 10 and 80000")
# print(tabulate(dataset.head(50), headers='keys', tablefmt='psql'))
# dataset = dataset.sample(frac=0.1)
print(dataset.shape)
dataset["loa"] = dataset["loa"].astype("float64")
dataset["boa"] = dataset["boa"].astype("float64")
dataset["draft"] = dataset["draft"].astype("float64")
dataset["power"] = dataset["power"].astype("float64")
dataset["speed"] = dataset["speed"].astype("float64")
dataset["year"] = dataset["year"].astype("int32")
dataset["vol"] = dataset["loa"] * dataset["boa"] * dataset["draft"]
dataset["fatness"] = dataset["loa"] / dataset["boa"]
good_rows = (np.abs(stats.zscore(dataset["fatness"])) < 3)
dataset = dataset[good_rows]
print(dataset.shape)




# plt.rcParams["figure.figsize"] = (8,6)
# ax = plt.axes(projection='3d')
# ax.scatter(dataset["vol"], dataset["speed"], dataset["power"], s=3, c=(dataset["fatness"]))
# ax.set_xlabel("vol")
# ax.set_ylabel("speed")
# ax.set_zlabel("power")
# plt.show()



