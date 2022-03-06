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
                            sql_column_names="uni_type, loa, boa, draft, speed, dynpos",
                            df_column_names=["type", "loa", "boa", "draft", "speed", "dynpos"],
                            additional_parameters="where year > 1985 and uni_type not in ('icebreaker', 'drillship / crane / pipelayer', 'tugboat', 'light') and year != '' and loa > 15 and engine_num not NULL and engine_rpm not NULL and propulsion_num not NULL and uni_type not NULL and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 3 and 100 and power between 10 and 120000")

df = extract_table_from_sql(table_name="ships_details",
                            sql_column_names="uni_type, ice_class, loa, boa, draft, speed, dynpos, deadweight",
                            df_column_names=["type", "ice", "loa", "boa", "draft", "speed", "dynpos", "deadweight"],
                            additional_parameters="where year > 1985 and year != '' and loa > 15 and uni_type not in ('icebreaker', 'drillship / crane / pipelayer', 'tugboat', 'light') and uni_type not NULL and boa not NULL and (loa / boa) > 2 and draft between 1 and 30 and speed between 3 and 100 and deadweight > 1 and deadweight not in ('', '--')")

df["loa"] = df["loa"].astype("float64")
df["boa"] = df["boa"].astype("float64")
df["draft"] = df["draft"].astype("float64")
df["deadweight"] = df["deadweight"].astype("float64")
df["speed"] = df["speed"].astype("float64")
df = df.drop(df[(df['type'] == "cargo") & (df['ice'] != 1)].sample(frac=.93, random_state=1).index)
df = df.drop(df[(df['type'] == "tanker") & (df['ice'] != 1)].sample(frac=.75, random_state=1).index)
df = df.drop(df[(df['type'] == "container ship") & (df['ice'] != 1)].sample(frac=.3, random_state=1).index)
print(df["type"].value_counts())
ship_type = pd.get_dummies(df["type"], prefix="type")
ice = pd.get_dummies(df["ice"], prefix="ice")
dynpos = pd.get_dummies(df["dynpos"], prefix="dynpos")
df_one_hot = pd.concat([df.drop(["ice", "type", "dynpos"], axis=1), ice, ship_type, dynpos], axis=1)


X = df_one_hot.drop(["loa", "boa", "draft"], axis=1)
scaler = RobustScaler()
scaler.fit(X)
dump(scaler, open('main_scaler.pkl', 'wb'))
print(scaler.feature_names_in_)


for y_name in ["loa", "boa", "draft"]:
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
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
        Dense(160, activation=leaky_relu),
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