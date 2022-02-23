from App import app
from flask import render_template, request
import tensorflow as tf
from tensorflow import keras
from pickle import load
import pandas as pd
from sklearn.preprocessing import RobustScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option('max_colwidth', None)

power_predict = keras.models.load_model('power.h5')
power_scaler = load(open('power_scaler.pkl', 'rb'))

engine_num_predict = keras.models.load_model('engine_num.h5')
engine_num_scaler = load(open('engine_num_scaler.pkl', 'rb'))

engine_rpm_predict = keras.models.load_model('engine_rpm.h5')
engine_rpm_scaler = load(open('engine_rpm_scaler.pkl', 'rb'))

propulsion_num_predict = keras.models.load_model('propulsion_num.h5')
propulsion_num_scaler = load(open('propulsion_num_scaler.pkl', 'rb'))


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/machinery', methods=['POST', 'GET'])
def machinery_page():
    ship_types = ['Fishing', 'cargo', 'container ship', 'drillship / crane / pipelayer', 'light',
                  'research', 'supply / tug / AHTS', 'tanker / gas carrier', 'tug', 'passenger ship',
                  'vehicles carrier / ro-ro', 'yacht / high-speed craft']
    if request.method == 'POST':
        form_data = {}
        df = {}
        form_data['ice_class'] = 0
        form_data.update(request.form)
        df['loa'] = [float(form_data['loa'])]
        df['boa'] = [float(form_data['boa'])]
        df['draft'] = [float(form_data['draught'])]
        df['speed'] = [float(form_data['speed'])]
        df['year'] = [int(form_data['year'])]
        df["cx1"] =  df["speed"][0] ** 2 * (df["boa"][0] * df["draft"][0])
        df['fatness'] = [df["loa"][0] / df["boa"][0]]
        df['ice_0'] = [0 if form_data['ice_class'] == 'on' else 1]
        df['ice_1'] = [1 if form_data['ice_class'] == 'on' else 0]
        for type in ship_types:
            df[f'type_{type}'] = [1 if form_data['type'] == type else 0]

        df = pd.DataFrame.from_dict(df)
        print(df)
        input_data = power_scaler.transform(df)
        power = power_predict.predict(input_data)
        engine_num = engine_num_predict.predict(input_data)
        engine_rpm = engine_rpm_predict.predict(input_data)
        propulsion_num = propulsion_num_predict.predict(input_data)
        return render_template('machinery_result.html', form_data={'Total engine power, kW': power[0][0],
                                                                   'Number of engines': round(engine_num[0][0]),
                                                                   'Engines RPM': engine_rpm[0][0],
                                                                   'Number of propulsion units': round(propulsion_num[0][0])})

    return render_template('machinery_calc.html', ship_types=ship_types)


@app.route('/about/<about>')
def about_page(about):
    return f'<h1 style="text-align: center;">About {about}</h1>'