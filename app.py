from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import all model modules
from models.football_arima import train_football_arima, predict_future as predict_football_arima
from models.football_sarima import train_football_sarima, predict_future as predict_football_sarima
from models.basketball_arima import train_basketball_arima, predict_future as predict_basketball_arima
from models.basketball_sarima import train_basketball_sarima, predict_future as predict_basketball_sarima
from models.cricket_arima import train_cricket_arima, predict_future as predict_cricket_arima
from models.cricket_sarima import train_cricket_sarima, predict_future as predict_cricket_sarima
from models.tennis_arima import train_tennis_arima, predict_future as predict_tennis_arima
from models.tennis_sarima import train_tennis_sarima, predict_future as predict_tennis_sarima

app = Flask(__name__)

# Load the dataset
DATASET_PATH = 'multiTimeline.csv'

def load_dataset():
    """Load and preprocess the dataset"""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file {DATASET_PATH} not found")
    
    df = pd.read_csv(DATASET_PATH, index_col='Month', parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M')
    return df

def prepare_data(data, sport):
    """Prepare data for the specific sport"""
    if sport == 'football':
        column = 'Premier League'
    elif sport == 'basketball':
        column = 'NBA'
    else:
        # For cricket and tennis, we'll use NBA data as placeholder
        column = 'NBA'
    
    return data[column]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sport = data['sport'].lower()
        model_type = data['model_type'].lower()
        
        # Load the dataset
        df = load_dataset()
        
        # Prepare data for the specific sport
        series = prepare_data(df, sport)
        
        # Train appropriate model based on sport and model type
        if sport == 'football':
            if model_type == 'arima':
                model = train_football_arima(series)
                forecast = predict_football_arima(model, steps=6)
            else:
                model = train_football_sarima(series)
                forecast = predict_football_sarima(model, steps=6)
        elif sport == 'basketball':
            if model_type == 'arima':
                model = train_basketball_arima(series)
                forecast = predict_basketball_arima(model, steps=6)
            else:
                model = train_basketball_sarima(series)
                forecast = predict_basketball_sarima(model, steps=6)
        elif sport == 'cricket':
            if model_type == 'arima':
                model = train_cricket_arima(series)
                forecast = predict_cricket_arima(model, steps=6)
            else:
                model = train_cricket_sarima(series)
                forecast = predict_cricket_sarima(model, steps=6)
        elif sport == 'tennis':
            if model_type == 'arima':
                model = train_tennis_arima(series)
                forecast = predict_tennis_arima(model, steps=6)
            else:
                model = train_tennis_sarima(series)
                forecast = predict_tennis_sarima(model, steps=6)
        else:
            return jsonify({'error': 'Invalid sport selected'}), 400
        
        # Prepare response
        forecast_dates = [(datetime.now() + timedelta(days=30*i)).strftime('%Y-%m') 
                         for i in range(1, 7)]
        
        response = {
            'dates': forecast_dates,
            'predictions': forecast.tolist()
        }
        
        return jsonify(response)
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/check_accuracy', methods=['POST'])
def check_accuracy():
    try:
        data = request.json
        sport = data['sport'].lower()
        model_type = data['model_type'].lower()
        
        # Load the dataset
        df = load_dataset()
        
        # Prepare data for the specific sport
        series = prepare_data(df, sport)
        
        # Split data into train and test sets
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Train appropriate model based on sport and model type
        if sport == 'football':
            if model_type == 'arima':
                model = train_football_arima(train)
                predictions = predict_football_arima(model, steps=len(test))
            else:
                model = train_football_sarima(train)
                predictions = predict_football_sarima(model, steps=len(test))
        elif sport == 'basketball':
            if model_type == 'arima':
                model = train_basketball_arima(train)
                predictions = predict_basketball_arima(model, steps=len(test))
            else:
                model = train_basketball_sarima(train)
                predictions = predict_basketball_sarima(model, steps=len(test))
        elif sport == 'cricket':
            if model_type == 'arima':
                model = train_cricket_arima(train)
                predictions = predict_cricket_arima(model, steps=len(test))
            else:
                model = train_cricket_sarima(train)
                predictions = predict_cricket_sarima(model, steps=len(test))
        elif sport == 'tennis':
            if model_type == 'arima':
                model = train_tennis_arima(train)
                predictions = predict_tennis_arima(model, steps=len(test))
            else:
                model = train_tennis_sarima(train)
                predictions = predict_tennis_sarima(model, steps=len(test))
        else:
            return jsonify({'error': 'Invalid sport selected'}), 400
        
        # Prepare response
        response = {
            'historical_dates': [str(date) for date in series.index],
            'historical_values': series.values.tolist(),
            'prediction_dates': [str(date) for date in test.index],
            'prediction_values': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Catch-all handler to ensure JSON responses and log the issue."""
    app.logger.exception('Unhandled exception: %s', error)
    if isinstance(error, HTTPException):
        return jsonify({'error': error.description}), error.code
    return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 