# SportPulse - Sports Popularity Prediction Platform

SportPulse is a Flask-based web application that forecasts the popularity of major sports (cricket, football, tennis, and basketball) using advanced ARIMA and SARIMA time-series models. The platform features an interactive, mobile-responsive dashboard with real-time predictions and model accuracy analysis.

## Features

- 🏀 **Multi-Sport Support**: Predict popularity trends for Basketball (NBA), Football (Premier League), Cricket, and Tennis
- 📊 **Dual Model System**: Choose between ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA) models
- 📈 **Interactive Visualizations**: Beautiful, responsive charts powered by Plotly.js
- 📱 **Mobile-Responsive Design**: Optimized for both desktop and mobile devices
- 🎨 **Modern UI**: Vibrant gradients, grid patterns, and professional typography
- ✅ **Model Accuracy Analysis**: Compare predictions against historical data

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Statsmodels, pmdarima
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Plotly.js
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MilindKashyap/Sport-Pulse-ARIMA-SARIMA.git
cd Sport-Pulse-ARIMA-SARIMA
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Select a sport from the dropdown menu
2. Choose your preferred model (ARIMA or SARIMA)
3. Click "Generate Prediction" to get 6-month forecasts
4. View detailed predictions in the interactive chart and data table
5. Click "Check Model Accuracy" to see how the model performs against historical data

## Project Structure

```
Sport-Pulse-ARIMA-SARIMA/
├── app.py                 # Flask application main file
├── models/                # ARIMA/SARIMA model implementations
│   ├── basketball_arima.py
│   ├── basketball_sarima.py
│   ├── cricket_arima.py
│   ├── cricket_sarima.py
│   ├── football_arima.py
│   ├── football_sarima.py
│   ├── tennis_arima.py
│   └── tennis_sarima.py
├── templates/             # HTML templates
│   └── index.html
├── static/                # CSS and JavaScript files
│   ├── style.css
│   └── script.js
├── multiTimeline.csv      # Dataset
└── requirements.txt       # Python dependencies
```

## Author

Made by Milind Kashyap

## License

This project is open source and available for educational purposes.
