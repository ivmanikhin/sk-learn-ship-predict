import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from pickle import dump
from time import sleep


def extract_table_from_sql(table_name, sql_column_names, df_column_names, additional_parameters):
    try:
        with sqlite3.connect('data/ships.db') as con:
            df = pd.read_sql_query(f"select {sql_column_names} from {table_name} {additional_parameters}", con)
        df.columns = df_column_names
        return df
    except:
        print("Failed to read SQL")


def mean_square_percentage_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = float((100 * (y_pred - y_true) / y_true) ** 2)
    return loss


df = extract_table_from_sql(table_name="ships_details",
                            sql_column_names="uni_type, loa, boa, draft, speed, power, engine_num, engine_rpm, propulsion_num, dynpos",
                            df_column_names=["type", "loa", "boa", "draft", "speed", "power", "engine_num", "engine_rpm", "propulsion_num", "dynpos"],
                            additional_parameters="where year > 1985 and uni_type not in ('icebreaker', 'drillship / crane / pipelayer', 'tugboat', 'light') and year != '' and loa > 15 and engine_num not NULL and engine_rpm not NULL and propulsion_num not NULL and uni_type not NULL and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 3 and 100 and power between 10 and 120000")

df["loa"] = df["loa"].astype("float64")
df["boa"] = df["boa"].astype("float64")
df["draft"] = df["draft"].astype("float64")
df["power"] = df["power"].astype("float64")
df["speed"] = df["speed"].astype("float64")
df["engine_num"] = df["engine_num"].astype("int32")
df["engine_rpm"] = df["engine_rpm"].astype("float64")
df["propulsion_num"] = df["propulsion_num"].astype("int32")
df["cx"] = (df["speed"] ** 3 * (df["loa"] * df["boa"] * df["draft"]) **(2/3)) / df["power"]
# df["cx1"] =  df["speed"] ** 3 * (df["loa"] * df["boa"] * df["draft"]) **(2/3)
df["cx1"] =  df["speed"] ** 2 * (df["boa"] * df["draft"])
df["fatness"] = df["loa"] / df["boa"]
good_rows = np.abs(stats.zscore(df["cx"])) < 3
df = df[good_rows]
df = df.drop(df[(df['type'] == "cargo")].sample(frac=.8, random_state=1).index)
df = df.drop(df[(df['type'] == "tanker")].sample(frac=.75, random_state=1).index)
df = df.drop(df[(df['type'] == "container ship")].sample(frac=.2, random_state=1).index)
print(df["type"].value_counts())
print(df.shape)
ship_type = pd.get_dummies(df["type"], prefix="type")
dynpos = pd.get_dummies(df["dynpos"], prefix="dynpos")
df_one_hot = pd.concat([df.drop(["type", "dynpos"], axis=1), ship_type, dynpos], axis=1)


X = df_one_hot.drop(["power", "cx", "engine_num", "speed", "engine_rpm", "propulsion_num"], axis=1)
scaler = RobustScaler()
scaler.fit(X)
dump(scaler, open('machinery_scaler.pkl', 'wb'))
print(scaler.feature_names_in_)


for y_name in ["power", "engine_num", "engine_rpm", "propulsion_num"]:
    print(f'============{y_name}=================================')
    y = df_one_hot.loc[:, [y_name]]

    # splitting to train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # normalize dataset
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    tf.random.set_seed(32)
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    model = Sequential([
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(1)
    ])

    es = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      mode='min',
      verbose=1,
      patience=15,
      restore_best_weights=True)

    model.compile(loss=mean_square_percentage_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mean_absolute_percentage_error"])

    history = model.fit(X_train, y_train, epochs=400, verbose=1, batch_size=360, validation_data=(X_test, y_test),
              callbacks=[es])

    model.evaluate(X_test, y_test)

    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()



    model.save(f'{y_name}.h5')