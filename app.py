# Import required libraries
import json
import numpy as np
import pandas as pd
import joblib
# Flask tools for web app, templates, requests, and JSON responses
from flask import Flask, render_template, request, jsonify
# Initialize Flask application
app = Flask(__name__)
# Load saved ML artifacts
model = joblib.load('artifacts/best_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')
le = joblib.load('artifacts/label_encoder_make.pkl')
# Load feature metadata
with open('artifacts/feature_metadata.json') as f:
    meta = json.load(f)
# Extract metadata values
MAKE_CLASSES = meta['make_classes']
CAFV_MAP = meta['cafv_map']
# Sample vehicles shown in the app for quick testing
SAMPLES = [
    {
        "label": "Tesla Model 3 (2023, Urban, BEV)",
        "model_year": 2023,
        "make": "TESLA",
        "ev_type": "BEV",
        "area_type": "Urban",
        "cafv": "Clean Alternative Fuel Vehicle Eligible",
    },
    {
        "label": "Toyota RAV4 Prime (2022, Urban, PHEV)",
        "model_year": 2022,
        "make": "TOYOTA",
        "ev_type": "PHEV",
        "area_type": "Urban",
        "cafv": "Eligibility unknown as battery range has not been researched",
    },
    {
        "label": "Nissan LEAF (2020, Rural, BEV)",
        "model_year": 2020,
        "make": "NISSAN",
        "ev_type": "BEV",
        "area_type": "Rural",
        "cafv": "Clean Alternative Fuel Vehicle Eligible",
    },
    {
        "label": "Jeep Wrangler 4xe (2021, Rural, PHEV)",
        "model_year": 2021,
        "make": "JEEP",
        "ev_type": "PHEV",
        "area_type": "Rural",
        "cafv": "Not eligible due to low battery range",
    },
    {
        "label": "Ford Mustang Mach-E (2024, Urban, BEV)",
        "model_year": 2024,
        "make": "FORD",
        "ev_type": "BEV",
        "area_type": "Urban",
        "cafv": "Clean Alternative Fuel Vehicle Eligible",
    },
    {
        "label": "BMW 330e (2019, Urban, PHEV)",
        "model_year": 2019,
        "make": "BMW",
        "ev_type": "PHEV",
        "area_type": "Urban",
        "cafv": "Not eligible due to low battery range",
    },
]

# Washington State urban counties (Census Bureau MSA definitions)
URBAN_COUNTIES = {
    'King', 'Pierce', 'Snohomish', 'Spokane', 'Clark',
    'Thurston', 'Whatcom', 'Kitsap', 'Benton', 'Franklin',
    'Yakima', 'Cowlitz', 'Skagit', 'Walla Walla'
}

# Load vehicle-registration dataset once at startup
REG_DF = None
try:
    _reg = pd.read_csv(
        'dataset/Vehicle_Registrations_by_Class_and_County_20260422.csv'
    )
    _reg['Count'] = pd.to_numeric(_reg['Count'], errors='coerce').fillna(0).astype(int)
    _reg['is_ev'] = (_reg['Fuel Type'] == 'Electric')
    _reg['area_type'] = _reg['Residential County'].apply(
        lambda c: 'Urban' if c in URBAN_COUNTIES else 'Rural'
    )
    REG_DF = _reg
except Exception as _exc:
    print(f'Warning: Could not load registration dataset: {_exc}')


def encode_and_predict(model_year, make, ev_type, area_type, cafv):
    make_upper = make.upper()
    if make_upper not in le.classes_:
        make_upper = "OTHER"
    make_encoded = int(le.transform([make_upper])[0])

    ev_binary = 1 if ev_type == "BEV" else 0
    area_binary = 1 if area_type == "Urban" else 0
    cafv_code = CAFV_MAP.get(cafv, 1)

    features = np.array([[model_year, make_encoded, ev_binary, area_binary, cafv_code]], dtype=float)
    prediction = float(model.predict(features)[0])
    return round(prediction, 1)

# Home page route
@app.route('/')
def index():
    return render_template(
        'index.html',
        makes=MAKE_CLASSES,
        cafv_options=list(CAFV_MAP.keys()),
        samples=SAMPLES,
        current_year=2026
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        model_year = int(data['model_year'])
        make = str(data['make'])
        ev_type = str(data['ev_type'])
        area_type = str(data['area_type'])
        cafv = str(data['cafv'])

        if model_year < 1990 or model_year > 2030:
            return jsonify({'error': 'Model year must be between 1990 and 2030.'}), 400

        predicted_range = encode_and_predict(model_year, make, ev_type, area_type, cafv)
        return jsonify({'predicted_range': predicted_range})
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

# Analysis data API route for dashboard charts
@app.route('/analysis_data')
def analysis_data():
    if REG_DF is None:
        return jsonify({'error': 'Registration dataset not available.'}), 500

    year_from = request.args.get('year_from', 2020, type=int)
    year_to   = request.args.get('year_to',   2025, type=int)
    area_type = request.args.get('area_type', 'All')

    if year_from > year_to:
        return jsonify({'error': 'Year From must be ≤ Year To.'}), 400

    df = REG_DF[
        (REG_DF['Fiscal Year'] >= year_from) &
        (REG_DF['Fiscal Year'] <= year_to)
    ]
    if area_type in ('Urban', 'Rural'):
        df = df[df['area_type'] == area_type]

    ev_by_year     = df[df['is_ev']].groupby('Fiscal Year')['Count'].sum()
    non_ev_by_year = df[~df['is_ev']].groupby('Fiscal Year')['Count'].sum()

    years         = list(range(year_from, year_to + 1))
    ev_counts     = [int(ev_by_year.get(y, 0))     for y in years]
    non_ev_counts = [int(non_ev_by_year.get(y, 0)) for y in years]
    total_counts  = [e + n for e, n in zip(ev_counts, non_ev_counts)]
    adoption_rates = [
        round(e / t * 100, 2) if t > 0 else 0
        for e, t in zip(ev_counts, total_counts)
    ]

    return jsonify({
        'years': years,
        'ev_counts': ev_counts,
        'non_ev_counts': non_ev_counts,
        'total_counts': total_counts,
        'adoption_rates': adoption_rates
    })

# Run Flask app locally
if __name__ == '__main__':
    app.run(debug=True, port=5000)
