from matplotlib import pyplot as plt
import numpy as np
from easy_SQL.easy_SQL import *
from scipy import stats
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import time


SEP = os.sep

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', None)


# Getting data from SQLite
df = extract_table_from_sql(table_name="main_ships_data_raw",
                            sql_column_names="uni_type, ice_class, loa, boa, draft, displacement, speed, power, year, engine_num, engine_rpm, propulsion_num, propulsion_type",
                            df_column_names=["type", "ice", "loa", "boa", "draft", "displacement", "speed", "power", "year", "engine_num", "engine_rpm", "propulsion_num", "propulsion_type"],
                            additional_parameters="where year > 1985 and year != '' and loa > 15 and uni_type not NULL and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 3 and 100 and power between 10 and 120000 and engine_num > 0 and engine_rpm > 0 and propulsion_num > 0 and propulsion_type not NULL and displacement > 0")


# Converting to numbers
df["loa"] = df["loa"].astype("float64")
df["boa"] = df["boa"].astype("float64")
df["draft"] = df["draft"].astype("float64")
df["power"] = df["power"].astype("float64")
df["speed"] = df["speed"].astype("float64")
df["year"] = df["year"].astype("int32")
df["displacement"] = df["displacement"].astype("float64")
df["engine_num"] = df["engine_num"].astype("int32")
df["engine_rpm"] = df["engine_rpm"].astype("float64")
df["propulsion_num"] = df["propulsion_num"].astype("int32")
df["cx"] = df["power"] / (df["speed"] ** 2 * df["draft"] * df["boa"])
df["fatness"] = df["loa"] / df["boa"]
df["vol"] = df["loa"] * df["boa"] * df["draft"]
df = df.drop(df[df['type'] == "cargo"].sample(frac=.95).index)
df = df.drop(df[df['type'] == "tanker / gas carrier"].sample(frac=.95).index)
df = df.drop(df[df['type'] == "container ship"].sample(frac=.85).index)
print(df["type"].value_counts())
ship_type = pd.get_dummies(df["type"], prefix="type")
propulsion_type = pd.get_dummies(df["propulsion_type"], prefix="propulsion_type")
ice = pd.get_dummies(df["ice"], prefix="ice")
df_one_hot = pd.concat([df.drop(["type", "ice", "propulsion_type"], axis=1), ship_type, ice, propulsion_type], axis=1)
print(df_one_hot.shape)
# Removing outliers
print(df_one_hot.head(10))
# good_rows = (np.abs(stats.zscore(df_one_hot["fatness"])) < 3)
# df_one_hot = df_one_hot[good_rows]
# print(df_one_hot.shape)
# time.sleep(3)

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
y = df_one_hot["engine_rpm"]
print("================================")
X = df_one_hot.drop(["propulsion_num", "engine_num", "engine_rpm", "displacement", "propulsion_type_Azimuth", "propulsion_type_Contra-Rotating", "propulsion_type_Controllable Pitch", "propulsion_type_Fixed Pitch", "propulsion_type_Voith-Schneider", "propulsion_type_Waterjets"], axis=1)

print("================================")

# splitting to train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


tf.random.set_seed(32)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.001)
model = Sequential([
    # Dropout(0.5),
    Dense(240, activation=leaky_relu),
    Dense(240, activation=leaky_relu),
    Dense(240, activation=leaky_relu),
    Dense(240, activation=leaky_relu),
    Dense(240, activation=leaky_relu),
    Dense(240, activation=leaky_relu),

    # Dense(200, activation=leaky_relu),
    # Dense(200, activation=leaky_relu),
    # Dense(200, activation=leaky_relu),
    Dense(1)
])

model.compile(loss=tf.keras.losses.mape,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              metrics=["mean_absolute_percentage_error"])

history = model.fit(X_train, y_train, epochs=400, verbose=1, batch_size=300)


model.evaluate(X_test, y_test)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

model.save("engine_rpm.h5")
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
