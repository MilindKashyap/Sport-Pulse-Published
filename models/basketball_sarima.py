import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='multiTimeline.csv'):
    """
    Load and preprocess the dataset with improved data cleaning
    """
    df = pd.read_csv(file_path, index_col='Month', parse_dates=True)
    df.index = pd.to_datetime(df.index)
    
    # Handle missing values
    df = df.interpolate(method='time')
    
    # Remove outliers using IQR method
    Q1 = df['NBA'].quantile(0.25)
    Q3 = df['NBA'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['NBA'] < (Q1 - 1.5 * IQR)) | (df['NBA'] > (Q3 + 1.5 * IQR)))]
    
    return df

def check_stationarity(series):
    """
    Check if the time series is stationary using ADF test
    """
    result = adfuller(series)
    return result[1] <= 0.05

def make_stationary(series):
    """
    Make the time series stationary using differencing
    """
    d = 0
    while not check_stationarity(series):
        series = series.diff().dropna()
        d += 1
    return series, d

def find_best_sarima_params(series, seasonal_period=12, max_p=2, max_d=2, max_q=2, max_P=2, max_D=1, max_Q=2):
    """
    Find the best SARIMA parameters using grid search
    """
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    # Make series stationary
    stationary_series, d = make_stationary(series)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(series,
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, seasonal_period))
                                results = model.fit()
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, seasonal_period)
                            except:
                                continue
                                
    return best_order, best_seasonal_order

def train_test_split(data, column='NBA', test_size=0.2):
    """
    Split the dataset into training and testing sets
    """
    train_size = int(len(data) * (1 - test_size))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test

def train_basketball_sarima(train_series):
    """
    Train the SARIMA model with optimized parameters
    """
    # Find best parameters
    best_order, best_seasonal_order = find_best_sarima_params(train_series)
    
    # Train model with best parameters
    model = SARIMAX(train_series,
                   order=best_order,
                   seasonal_order=best_seasonal_order)
    results = model.fit()
    
    return results

def predict_future(model, steps=6):
    """
    Make future predictions using the trained model
    """
    forecast = model.forecast(steps=steps)
    return forecast

def evaluate_model(train, test, predictions):
    """
    Evaluate model performance using multiple metrics
    """
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def plot_predictions(train, test, predictions, column='NBA'):
    """
    Plot the training, testing, and predicted values with improved visualization
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train[column], label='Train', color='blue')
    plt.plot(test[column], label='Test', color='green')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    
    # Add confidence intervals
    conf_int = model.get_forecast(len(test)).conf_int()
    plt.fill_between(test.index, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1], 
                    color='red', alpha=0.1)
    
    plt.title(f'SARIMA Prediction for {column}')
    plt.xlabel('Date')
    plt.ylabel('Popularity')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, filename='basketball_sarima_model.pkl'):
    """
    Save the trained SARIMA model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='basketball_sarima_model.pkl'):
    """
    Load a trained model from a file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    df = load_data()
    train, test = train_test_split(df, 'NBA')
    model = train_basketball_sarima(train['NBA'])
    predictions = predict_future(model, steps=len(test))
    
    # Evaluate model
    metrics = evaluate_model(test['NBA'], predictions)
    print("Model Evaluation Metrics:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    
    # Plot results
    plot_predictions(train, test, predictions, 'NBA')
    save_model(model)

if __name__ == "__main__":
    main() 