document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const predictionResult = document.getElementById('predictionResult');
    const plotDiv = document.getElementById('plot');
    const accuracyPlot = document.getElementById('accuracyPlot');
    const checkAccuracyBtn = document.getElementById('checkAccuracy');
    const predictionTable = document.getElementById('predictionTable').getElementsByTagName('tbody')[0];

    let currentData = null;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const sport = document.getElementById('sport').value;
        const model = document.getElementById('model').value;
        
        // Validate input
        if (!sport || !model) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = 'Please select both sport and model';
            return;
        }

        try {
            // Show loading state
            loadingDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            predictionResult.style.display = 'none';
            accuracyPlot.style.display = 'none';

            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sport: sport,
                    model_type: model
                })
            });

            const data = await response.json();

            if (response.ok) {
                currentData = data;
                
                // Create visualization
                const trace = {
                    x: data.dates,
                    y: data.predictions,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Predictions',
                    line: {
                        color: '#667eea',
                        width: 3,
                        shape: 'spline'
                    },
                    marker: {
                        color: '#667eea',
                        size: 8,
                        line: {
                            color: '#ffffff',
                            width: 2
                        }
                    },
                    fill: 'tonexty',
                    fillcolor: 'rgba(102, 126, 234, 0.1)'
                };

                const layout = {
                    title: {
                        text: `${sport.charAt(0).toUpperCase() + sport.slice(1)} Popularity Prediction`,
                        font: {
                            family: 'Poppins, sans-serif',
                            size: 20,
                            color: '#1a202c'
                        }
                    },
                    xaxis: {
                        title: {
                            text: 'Date',
                            font: {
                                family: 'Inter, sans-serif',
                                size: 14,
                                color: '#718096'
                            }
                        },
                        gridcolor: '#e2e8f0',
                        gridwidth: 1,
                        showgrid: true
                    },
                    yaxis: {
                        title: {
                            text: 'Popularity Index',
                            font: {
                                family: 'Inter, sans-serif',
                                size: 14,
                                color: '#718096'
                            }
                        },
                        gridcolor: '#e2e8f0',
                        gridwidth: 1,
                        showgrid: true
                    },
                    hovermode: 'x unified',
                    showlegend: true,
                    legend: {
                        font: {
                            family: 'Inter, sans-serif',
                            size: 12
                        }
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        family: 'Inter, sans-serif'
                    },
                    margin: {
                        l: 60,
                        r: 30,
                        t: 60,
                        b: 50
                    }
                };

                Plotly.newPlot(plotDiv, [trace], layout);
                
                // Update prediction table
                predictionTable.innerHTML = '';
                data.dates.forEach((date, index) => {
                    const row = predictionTable.insertRow();
                    const dateCell = row.insertCell(0);
                    const predictionCell = row.insertCell(1);
                    dateCell.textContent = date;
                    predictionCell.textContent = data.predictions[index].toFixed(2);
                });

                // Show results
                predictionResult.style.display = 'block';
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = `Error: ${error.message}`;
        } finally {
            loadingDiv.style.display = 'none';
        }
    });

    checkAccuracyBtn.addEventListener('click', async function() {
        if (!currentData) return;

        try {
            // Show loading state
            loadingDiv.style.display = 'block';
            errorDiv.style.display = 'none';

            // Make accuracy check request
            const response = await fetch('/check_accuracy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sport: document.getElementById('sport').value,
                    model_type: document.getElementById('model').value
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Create accuracy visualization
                const trace1 = {
                    x: data.historical_dates,
                    y: data.historical_values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Historical Data',
                    line: {
                        color: '#43e97b',
                        width: 3,
                        shape: 'spline'
                    },
                    marker: {
                        color: '#43e97b',
                        size: 6,
                        line: {
                            color: '#ffffff',
                            width: 1
                        }
                    }
                };

                const trace2 = {
                    x: data.prediction_dates,
                    y: data.prediction_values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Model Predictions',
                    line: {
                        color: '#f5576c',
                        width: 3,
                        dash: 'dash',
                        shape: 'spline'
                    },
                    marker: {
                        color: '#f5576c',
                        size: 8,
                        symbol: 'diamond',
                        line: {
                            color: '#ffffff',
                            width: 2
                        }
                    }
                };

                const layout = {
                    title: {
                        text: 'Model Accuracy Analysis',
                        font: {
                            family: 'Poppins, sans-serif',
                            size: 20,
                            color: '#1a202c'
                        }
                    },
                    xaxis: {
                        title: {
                            text: 'Date',
                            font: {
                                family: 'Inter, sans-serif',
                                size: 14,
                                color: '#718096'
                            }
                        },
                        gridcolor: '#e2e8f0',
                        gridwidth: 1,
                        showgrid: true
                    },
                    yaxis: {
                        title: {
                            text: 'Popularity Index',
                            font: {
                                family: 'Inter, sans-serif',
                                size: 14,
                                color: '#718096'
                            }
                        },
                        gridcolor: '#e2e8f0',
                        gridwidth: 1,
                        showgrid: true
                    },
                    hovermode: 'x unified',
                    showlegend: true,
                    legend: {
                        font: {
                            family: 'Inter, sans-serif',
                            size: 12
                        },
                        x: 0.5,
                        y: -0.2,
                        xanchor: 'center'
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        family: 'Inter, sans-serif'
                    },
                    margin: {
                        l: 60,
                        r: 30,
                        t: 60,
                        b: 80
                    }
                };

                Plotly.newPlot(accuracyPlot, [trace1, trace2], layout);
                accuracyPlot.style.display = 'block';
            } else {
                throw new Error(data.error || 'Failed to check accuracy');
            }
        } catch (error) {
            errorDiv.style.display = 'block';
            errorDiv.querySelector('p').textContent = `Error: ${error.message}`;
        } finally {
            loadingDiv.style.display = 'none';
        }
    });
}); 