from App import app, machinery_predict, main_predict
from flask import render_template, request


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/main', methods=['POST', 'GET'])
def main_page():
    form_data = {
        'ice_class': '',
        'dynpos': '',
        'ship_type': 'Cargo',
        'deadweight': 20000,
        'speed': 15,
        'year': 2024
    }

    result = [
        {
            'Length overall, m': 0,
            'Breadth overall, m': 0,
            'Draught, m': 0,
        }, {
            'Total propulsion power, kW': 0,
            'Number of main engines': 0,
            'Engine RPM': 0,
            'Number of propulsion units': 0
        }
    ]
    if request.method == 'POST':
        form_data.update(request.form)
        result[0] = main_predict.predict_main(form_data)
        form_data['loa'] = result[0]['loa']
        form_data['boa'] = result[0]['boa']
        form_data['draught'] = result[0]['draught']
        result[1] = machinery_predict.predict_machinery(form_data)
        return render_template('main_calc.html', ship_types=main_predict.ship_types, form_data=form_data, result=result)

    return render_template('main_calc.html', ship_types=main_predict.ship_types, form_data=form_data, result=result)


@app.route('/machinery', methods=['POST', 'GET'])
def machinery_page():
    form_data = {
        'ice_class': '',
        'dynpos': '',
        'ship_type': 'Research',
        'loa': 122,
        'boa': 20,
        'draught': 5.6,
        'speed': 15,
        'year': 2024,
        }

    result = {
        'Total propulsion power, kW': 0,
        'Number of main engines': 0,
        'Engine RPM': 0,
        'Number of propulsion units': 0,
        }

    if request.method == 'POST':
        form_data.update(request.form)
        result = machinery_predict.predict_machinery(form_data)
        return render_template('machinery_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data, result=result)

    return render_template('machinery_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data, result=result)


@app.route('/about/<about>')
def about_page(about):
    return f'<h1 style="text-align: center;">About {about}</h1>'