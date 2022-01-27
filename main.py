from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from easy_SQL.easy_SQL import *
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

SEP = os.sep

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', None)


# Getting data from SQLite
df = extract_table_from_sql(table_name="main_ships_data_raw",
                            sql_column_names="type, loa, boa, draft, speed, power, year",
                            df_column_names=["type", "loa", "boa", "draft", "speed", "power", "year"],
                            additional_parameters="where year > 1985 and loa > 30 and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 5 and 100 and power between 10 and 80000")
# print(tabulate(df.head(50), headers='keys', tablefmt='psql'))
# df = df.sample(frac=0.1)
print(df.shape)

# Converting to numbers
df["loa"] = df["loa"].astype("float64")
df["boa"] = df["boa"].astype("float64")
df["draft"] = df["draft"].astype("float64")
df["power"] = df["power"].astype("float64")
df["speed"] = df["speed"].astype("float64")
df["year"] = df["year"].astype("int32")

# Removing outliers
df["fatness"] = df["loa"] / df["boa"]
good_rows = (np.abs(stats.zscore(df["fatness"])) < 3)
df = df[good_rows]
print(df.shape)


# making dataset
y = df.pop("power").values
X = df[["loa", "boa", "draft", "speed"]].values.tolist()


# splitting to train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)


plt.rcParams["figure.figsize"] = (8,6)
ax = plt.axes(projection='3d')
vol = [a[0] * a[1] * a[2] for a in X_test]
speed = [a[3] for a in X_test]
print(vol)

ax.scatter(vol, speed, y_test, s=3, c="b")
ax.scatter(vol, speed, pred, s=3, c="r")
ax.set_xlabel("vol")
ax.set_ylabel("speed")
ax.set_zlabel("power")
plt.show()



