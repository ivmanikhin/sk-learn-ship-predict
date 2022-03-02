from App import app, machinery_predict, main_predict
from flask import render_template, request




@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/main', methods=['POST', 'GET'])
def main_page():
    form_data_main = {
        'ice_class': '',
        'dynpos': '',
        'ship_type': 'Cargo',
        'deadweight': 20000,
        'speed': 15,
        'year': 2024
    }

    form_data_machinery = {
        'ice_class': form_data_main['ice_class'],
        'dynpos': form_data_main['dynpos'],
        'ship_type': form_data_main['ship_type'],
        'loa': 0,
        'boa': 0,
        'draught': 0,
        'speed': form_data_main['speed'],
        'year': form_data_main['year']
    }
    if request.method == 'POST':
        form_data_main.update(request.form)
        form_data_machinery = main_predict.predict_main(form_data_main)
        return render_template('main_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data_main, result=result_main)

    return render_template('machinery_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data_machinery, result=result_main)


@app.route('/machinery', methods=['POST', 'GET'])
def machinery_page():
    form_data = {'ice_class': '',
                 'dynpos': '',
                 'ship_type': 'Research',
                 'loa': 122,
                 'boa': 20,
                 'draught': 5.6,
                 'speed': 15,
                 'year': 2024}

    result = {'Total propulsion power, kW': 0,
              'Number of main engines': 0,
              'Engine RPM': 0,
              'Number of propulsion units': 0}
    if request.method == 'POST':
        form_data.update(request.form)
        result = machinery_predict.predict_machinery(form_data)
        return render_template('machinery_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data, result=result)

    return render_template('machinery_calc.html', ship_types=machinery_predict.ship_types, form_data=form_data, result=result)


@app.route('/about/<about>')
def about_page(about):
    return f'<h1 style="text-align: center;">About {about}</h1>'