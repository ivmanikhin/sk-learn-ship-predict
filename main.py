from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from easy_SQL.easy_SQL import *
from scipy import stats
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
import time


SEP = os.sep

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', None)


# Getting data from SQLite
df = extract_table_from_sql(table_name="main_ships_data_raw",
                            sql_column_names="uni_type, ice_class, loa, boa, draft, speed, power, year",
                            df_column_names=["type", "ice", "loa", "boa", "draft", "speed", "power", "year"],
                            additional_parameters="where year > 1985 and loa > 30 and uni_type not NULL and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 5 and 100 and power between 10 and 80000")
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
type = pd.get_dummies(df["type"], prefix="type")
ice = pd.get_dummies(df["ice"], prefix="ice")
df_one_hot = pd.concat([df.drop(["type", "ice"], axis=1), type, ice], axis=1)
# Removing outliers
df_one_hot["fatness"] = df_one_hot["loa"] / df_one_hot["boa"]
good_rows = (np.abs(stats.zscore(df_one_hot["fatness"])) < 2)
df_one_hot = df_one_hot[good_rows]
print(df_one_hot.shape)
time.sleep(3)

######################################################
#
# plt.rcParams["figure.figsize"] = (8,6)
# ax = plt.axes(projection='3d')
# vol = [a[0] * a[1] * a[2] for a in dataset]
# speed = [a[3] for a in dataset]
# power = [a[4] for a in dataset]
#
# ax.scatter(vol, speed, power, s=3, c="b")
# # ax.scatter(vol, speed, pred, s=3, c="r")
# ax.set_xlabel("vol")
# ax.set_ylabel("speed")
# ax.set_zlabel("power")
# plt.show()
#

#
y = df_one_hot["power"]
X = df_one_hot.drop(["power"], axis=1)


# splitting to train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


tf.random.set_seed(42)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.002)
model = Sequential([
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(1)
])
model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=1000)


model.evaluate(X_test, y_test)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

'''
tf.random.set_seed(42)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.002)
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=1200)

Epoch 199/200
19/19 [==============================] - 1s 60ms/step - loss: 8.5021 - mean_absolute_percentage_error: 8.5021
Epoch 200/200
19/19 [==============================] - 1s 60ms/step - loss: 8.1533 - mean_absolute_percentage_error: 8.1533
78/78 [==============================] - 0s 4ms/step - loss: 10.8341 - mean_absolute_percentage_error: 10.8341


==============================================================================================================

tf.random.set_seed(42)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.001)
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(500, activation=leaky_relu),
                             tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=1200)

Epoch 197/200
19/19 [==============================] - 1s 56ms/step - loss: 7.4132 - mean_absolute_percentage_error: 7.4132
Epoch 198/200
19/19 [==============================] - 1s 56ms/step - loss: 7.3620 - mean_absolute_percentage_error: 7.3620
Epoch 199/200
19/19 [==============================] - 1s 57ms/step - loss: 7.3689 - mean_absolute_percentage_error: 7.3689
Epoch 200/200
19/19 [==============================] - 1s 55ms/step - loss: 7.2126 - mean_absolute_percentage_error: 7.2126
78/78 [==============================] - 0s 4ms/step - loss: 11.9850 - mean_absolute_percentage_error: 11.9850

==============================================================================

tf.random.set_seed(42)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.001)
model = Sequential([
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(1)
])
model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=1200)

Epoch 196/200
19/19 [==============================] - 1s 56ms/step - loss: 7.6457 - mean_absolute_percentage_error: 7.6457
Epoch 197/200
19/19 [==============================] - 1s 56ms/step - loss: 7.9100 - mean_absolute_percentage_error: 7.9100
Epoch 198/200
19/19 [==============================] - 1s 55ms/step - loss: 7.7104 - mean_absolute_percentage_error: 7.7104
Epoch 199/200
19/19 [==============================] - 1s 55ms/step - loss: 7.3769 - mean_absolute_percentage_error: 7.3769
Epoch 200/200
19/19 [==============================] - 1s 56ms/step - loss: 7.5075 - mean_absolute_percentage_error: 7.5075
78/78 [==============================] - 0s 4ms/step - loss: 10.3731 - mean_absolute_percentage_error: 10.3731



==================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


tf.random.set_seed(42)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.002)
model = Sequential([
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(500, activation=leaky_relu),
    Dense(1)
])
model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=1000)

Epoch 196/200
23/23 [==============================] - 1s 53ms/step - loss: 7.5629 - mean_absolute_percentage_error: 7.5629
Epoch 197/200
23/23 [==============================] - 1s 52ms/step - loss: 7.7844 - mean_absolute_percentage_error: 7.7844
Epoch 198/200
23/23 [==============================] - 1s 51ms/step - loss: 8.4265 - mean_absolute_percentage_error: 8.4265
Epoch 199/200
23/23 [==============================] - 1s 51ms/step - loss: 7.7808 - mean_absolute_percentage_error: 7.7808
Epoch 200/200
23/23 [==============================] - 1s 52ms/step - loss: 7.9025 - mean_absolute_percentage_error: 7.9025
78/78 [==============================] - 0s 4ms/step - loss: 10.9036 - mean_absolute_percentage_error: 10.9036

'''
