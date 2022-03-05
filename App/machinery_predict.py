import tensorflow as tf
from tensorflow.keras.models import load_model
from pickle import load
from os import sep


def mean_square_percentage_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = float((100 * (y_pred - y_true) / y_true) ** 2)
    return loss


power_predict = load_model(f'App{sep}nn_machinery{sep}power.h5', custom_objects={"mean_square_percentage_loss": mean_square_percentage_loss})
scaler = load(open(f'App{sep}nn_machinery{sep}machinery_scaler.pkl', 'rb'))
engine_num_predict = load_model(f'App{sep}nn_machinery{sep}engine_num.h5', custom_objects={"mean_square_percentage_loss": mean_square_percentage_loss})
engine_rpm_predict = load_model(f'App{sep}nn_machinery{sep}engine_rpm.h5', custom_objects={"mean_square_percentage_loss": mean_square_percentage_loss})
propulsion_num_predict = load_model(f'App{sep}nn_machinery{sep}propulsion_num.h5', custom_objects={"mean_square_percentage_loss": mean_square_percentage_loss})

ship_types = ['Fishing', 'Cargo', 'Container ship', 'Gas carrier',
              'Passenger ship', 'Research', 'Supply', 'Tanker', 'Tug',
              'Vehicles carrier / Ro-Ro', 'Yacht / High-speed craft']


def predict_machinery(form_data):
    df = dict()
    df['loa'] = float(form_data['loa'])
    df['boa'] = float(form_data['boa'])
    df['draft'] = float(form_data['draught'])
    df['cx1'] = float(form_data['speed']) ** 2 * (df['boa'] * df['draft'])
    df['fatness'] = df["loa"] / df["boa"]
    for ship_type in ship_types:
        df[f'type_{ship_type}'] = 1 if form_data['ship_type'] == ship_type else 0
    df['dynpos_0'] = 0 if form_data['dynpos'] == 'on' else 1
    df['dynpos_1'] = 1 if form_data['dynpos'] == 'on' else 0
    df = [list(df.values())]
    # print(df)
    input_data = scaler.transform(df)
    # print(input_data)
    power = round(power_predict.predict(input_data)[0][0])
    engine_num = round(engine_num_predict.predict(input_data)[0][0])
    engine_rpm = round(engine_rpm_predict.predict(input_data)[0][0])
    propulsion_num = round(propulsion_num_predict.predict(input_data)[0][0])
    return {'Total engine power, kW': power,
            'Number of main engines': engine_num,
            'Engine RPM': engine_rpm,
            'Number of propulsion units': propulsion_num}
