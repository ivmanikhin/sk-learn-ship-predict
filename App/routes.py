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
scaler = load(open('scaler.pkl', 'rb'))
engine_num_predict = keras.models.load_model('engine_num.h5')
engine_rpm_predict = keras.models.load_model('engine_rpm.h5')
propulsion_num_predict = keras.models.load_model('propulsion_num.h5')



@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/machinery', methods=['POST', 'GET'])
def machinery_page():
    ship_types = ['Fishing', 'cargo', 'container ship', 'drillship / crane / pipelayer', 'light',
                  'passenger ship', 'research', 'supply / tug / AHTS', 'tanker / gas carrier', 'tug',
                  'vehicles carrier / ro-ro', 'yacht / high-speed craft']
    if request.method == 'POST':
        form_data = {}
        df = {}
        form_data['ice_class'] = 0
        form_data['dynpos'] = 0
        form_data.update(request.form)
        df['loa'] = [float(form_data['loa'])]
        df['boa'] = [float(form_data['boa'])]
        df['draft'] = [float(form_data['draught'])]
        # df['speed'] = [float(form_data['speed'])]
        df['year'] = [int(form_data['year'])]
        df["cx1"] = float(form_data['speed']) ** 2 * (df["boa"][0] * df["draft"][0])
        df['fatness'] = [df["loa"][0] / df["boa"][0]]
        df['ice_0'] = [0 if form_data['ice_class'] == 'on' else 1]
        df['ice_1'] = [1 if form_data['ice_class'] == 'on' else 0]
        for type in ship_types:
            df[f'type_{type}'] = [1 if form_data['type'] == type else 0]
        df['dynpos_0'] = [0 if form_data['dynpos'] == 'on' else 1]
        df['dynpos_1'] = [1 if form_data['dynpos'] == 'on' else 0]
        df = pd.DataFrame.from_dict(df)
        print(df)
        input_data = scaler.transform(df)
        power = power_predict.predict(input_data)
        engine_num = engine_num_predict.predict(input_data)
        engine_rpm = engine_rpm_predict.predict(input_data)
        propulsion_num = propulsion_num_predict.predict(input_data)

        return render_template('machinery_result.html', form_data={'Total engine power, kW': round(power[0][0]),
                                                                   'Number of engines': round(engine_num[0][0]),
                                                                   'Engines RPM': round(engine_rpm[0][0]),
                                                                   'Number of propulsion units': round(propulsion_num[0][0])})

    return render_template('machinery_calc.html', ship_types=ship_types)


@app.route('/about/<about>')
def about_page(about):
    return f'<h1 style="text-align: center;">About {about}</h1>'