from tensorflow.keras.models import load_model
from pickle import load

power_predict = load_model('power.h5')
scaler = load(open('scaler.pkl', 'rb'))
engine_num_predict = load_model('engine_num.h5')
engine_rpm_predict = load_model('engine_rpm.h5')
propulsion_num_predict = load_model('propulsion_num.h5')

ship_types = ['Fishing', 'Cargo', 'Container ship', 'Drillship / Crane / Pipelayer', 'Lite',
              'Passenger ship', 'Research', 'Supply', 'Tanker / Gas carrier', 'Tug',
              'Vehicles carrier / Ro-Ro', 'Yacht / High-speed craft']


def predict_machinery(form_data):
    df = dict()
    df['loa'] = float(form_data['loa'])
    df['boa'] = float(form_data['boa'])
    df['draft'] = float(form_data['draught'])
    df['year'] = int(form_data['year'])
    df['cx1'] = float(form_data['speed']) ** 2 * (df['boa'] * df['draft'])
    df['fatness'] = df["loa"] / df["boa"]
    df['ice_0'] = 0 if form_data['ice_class'] == 'on' else 1
    df['ice_1'] = 1 if form_data['ice_class'] == 'on' else 0
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
    return {'Total propulsion power, kW': power,
            'Number of main engines': engine_num,
            'Engine RPM': engine_rpm,
            'Number of propulsion units': propulsion_num}
