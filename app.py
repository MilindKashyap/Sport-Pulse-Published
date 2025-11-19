from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
import pandas as pd
from datetime import datetime, timedelta
import os
import threading

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load the dataset
DATASET_PATH = 'multiTimeline.csv'
_dataset_cache = None
_dataset_lock = threading.Lock()
_model_cache = {}
_model_cache_lock = threading.Lock()

MODEL_CONFIG = {
    'football': {
        'arima': {'order': (2, 1, 2)},
        'sarima': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
    },
    'basketball': {
        'arima': {'order': (2, 1, 1)},
        'sarima': {'order': (1, 1, 1), 'seasonal_order': (0, 1, 1, 12)}
    },
    'cricket': {
        'arima': {'order': (1, 1, 2)},
        'sarima': {'order': (1, 1, 0), 'seasonal_order': (1, 1, 0, 12)}
    },
    'tennis': {
        'arima': {'order': (1, 1, 1)},
        'sarima': {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 12)}
    }
}

def load_dataset():
    """Load and preprocess the dataset with simple in-memory caching."""
    global _dataset_cache
    if _dataset_cache is None:
        with _dataset_lock:
            if _dataset_cache is None:
                if not os.path.exists(DATASET_PATH):
                    raise FileNotFoundError(f"Dataset file {DATASET_PATH} not found")
                df = pd.read_csv(DATASET_PATH, index_col='Month', parse_dates=True)
                df.index = pd.DatetimeIndex(df.index).to_period('M')
                _dataset_cache = df
    return _dataset_cache.copy()

def prepare_data(data, sport):
    """Prepare data for the specific sport"""
    if sport == 'football':
        column = 'Premier League'
    elif sport == 'basketball':
        column = 'NBA'
    else:
        # For cricket and tennis, we'll use NBA data as placeholder
        column = 'NBA'
    
    series = data[column].astype(float)
    try:
        series.index = series.index.to_timestamp()
    except AttributeError:
        pass
    return series


def _train_model(series, sport, model_type):
    """Train ARIMA/SARIMA model using predefined hyperparameters."""
    config = MODEL_CONFIG.get(sport, {}).get(model_type)
    if not config:
        raise ValueError('Unsupported sport/model configuration')
    order = config.get('order', (1, 1, 1))
    seasonal_order = config.get('seasonal_order')

    if seasonal_order:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False)
    else:
        model = ARIMA(series, order=order)
        return model.fit()


def _get_or_train_model(sport, model_type, series):
    key = (sport, model_type)
    model = _model_cache.get(key)
    if model is not None:
        return model
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            model = _train_model(series, sport, model_type)
            _model_cache[key] = model
    return model


def _predict_with_model(model, steps=6):
    forecast = model.forecast(steps=steps)
    return forecast.tolist()


def _validate_input(sport, model_type):
    valid_sports = {'football', 'basketball', 'cricket', 'tennis'}
    valid_models = {'arima', 'sarima'}
    if sport not in valid_sports:
        raise ValueError('Invalid sport selected')
    if model_type not in valid_models:
        raise ValueError('Invalid model type selected')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sport = data['sport'].lower()
        model_type = data['model_type'].lower()
        _validate_input(sport, model_type)
        
        # Load the dataset
        df = load_dataset()
        
        # Prepare data for the specific sport
        series = prepare_data(df, sport)
        
        model = _get_or_train_model(sport, model_type, series)
        forecast = _predict_with_model(model, steps=6)
        
        # Prepare response
        forecast_dates = [(datetime.now() + timedelta(days=30*i)).strftime('%Y-%m') 
                         for i in range(1, 7)]
        
        response = {
            'dates': forecast_dates,
            'predictions': forecast
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
        _validate_input(sport, model_type)
        
        # Load the dataset
        df = load_dataset()
        
        # Prepare data for the specific sport
        series = prepare_data(df, sport)
        
        # Split data into train and test sets
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        model = _train_model(train, sport, model_type)
        predictions = model.forecast(steps=len(test)).tolist()
        
        # Prepare response
        response = {
            'historical_dates': [str(date) for date in series.index],
            'historical_values': series.values.tolist(),
            'prediction_dates': [str(date) for date in test.index],
            'prediction_values': predictions
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