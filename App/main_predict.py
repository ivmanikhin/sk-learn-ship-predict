from tensorflow.keras.models import load_model
from pickle import load
from os import sep


loa_predict = load_model(f'App{sep}nn_main{sep}loa.h5')
scaler = load(open(f'App{sep}nn_main{sep}main_scaler.pkl', 'rb'))
boa_predict = load_model(f'App{sep}nn_main{sep}boa.h5')
draft_predict = load_model(f'App{sep}nn_main{sep}draft.h5')

ship_types = ['Fishing', 'Cargo', 'Container ship',
              'Passenger ship', 'Research', 'Supply', 'Tanker / Gas carrier', 'Tug',
              'Vehicles carrier / Ro-Ro', 'Yacht / High-speed craft']


def predict_main(form_data):
    df = dict()
    df['speed'] = float(form_data['speed'])
    df['deadweight'] = int(form_data['deadweight'])
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
    loa = round(loa_predict.predict(input_data)[0][0])
    boa = round(boa_predict.predict(input_data)[0][0])
    draught = round(draft_predict.predict(input_data)[0][0])
    return {'loa': loa,
            'boa': boa,
            'draught': draught}
