from App import app
from flask import render_template, request
import pandas as pd


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/machinery', methods=['POST', 'GET'])
def machinery_page():
    if request.method == 'POST':
        form_data = {}
        df = pd.DataFrame()
        form_data['ice_class'] = 'off'
        form_data.update(request.form)
        df['loa'] = form_data['loa']
        df['boa'] = form_data['boa']
        df['draft'] = form_data['draught']
        df['speed'] = form_data['speed']
        df['year'] = 2022
        df['type'] = form_data['type']
        df['ice'] = [True if form_data['ice_class'] == 'on' else False]
        return render_template('result.html', form_data=form_data)
    machinery = [
        {'name': 'Total engine power', 'value': 0, 'unit': 'kW'},
        {'name': 'Number of engines', 'value': 0, 'unit': ''},
        {'name': 'Engines RPM', 'value': 6, 'unit': 'RPM'},
        {'name': 'Type of propulsion', 'value': '', 'unit': ''},
        {'name': 'Number of propulsion units', 'value': 2, 'unit': ''}
    ]
    return render_template('calc.html', machinery=machinery)


@app.route('/about/<about>')
def about_page(about):
    return f'<h1 style="text-align: center;">About {about}</h1>'