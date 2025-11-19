import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

def load_data(file_path='MultiTimeline.csv'):
    """
    Load and preprocess the dataset
    """
    df = pd.read_csv(file_path, index_col='Month', parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M')
    return df

def train_test_split(data, column='Premier League', split_index=150):
    """
    Split the dataset into training and testing sets
    """
    train = data[0:split_index]
    test = data[split_index:]
    return train, test

def train_football_sarima(train_series, order=(2, 1, 4), seasonal_order=(1, 1, 1, 12)):
    """
    Train the SARIMA model for football
    """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

def predict_future(model, steps=6):
    """
    Make future predictions using the trained model
    """
    forecast = model.forecast(steps=steps)
    return forecast

def plot_predictions(train, test, predictions, column='Premier League'):
    """
    Plot the training data, testing data, and predictions
    """
    plt.figure(figsize=(16, 8))
    plt.plot(train[column], label='Train')
    plt.plot(test[column], label='Test')
    plt.plot(predictions, label='Predicted', color='red')
    plt.title(f'SARIMA Prediction for {column}')
    plt.xlabel('Date')
    plt.ylabel('Popularity')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, filename='football_sarima_model.pkl'):
    """
    Save the trained model to a file
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='football_sarima_model.pkl'):
    """
    Load a trained model from a file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    df = load_data()
    train, test = train_test_split(df, 'Premier League')
    model = train_football_sarima(train['Premier League'])
    predictions = predict_future(model, steps=6)
    
    plot_predictions(train, test, predictions, 'Premier League')
    save_model(model)

if __name__ == "__main__":
    main() 