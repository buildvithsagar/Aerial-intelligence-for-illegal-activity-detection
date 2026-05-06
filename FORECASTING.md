# Crowd Density Forecasting Documentation

## Overview

The Heat Crowd dashboard now includes **AI-powered crowd density forecasting** with multiple forecasting methods and confidence intervals. Predict crowd density up to 10 minutes in advance to enable proactive management.

## Features

### Multi-Method Forecasting
- **LSTM Neural Network**: Deep learning model trained on historical crowd patterns (when TensorFlow is installed)
- **Linear Regression**: Trend-based forecasting 
- **Exponential Smoothing**: Statistical method for time-series smoothing
- **Hybrid Mode**: Automatically tries LSTM first, falls back to linear+exponential methods

### Confidence Intervals
- 95% confidence bands based on historical prediction residuals
- Helps assess forecast uncertainty

### Configurable Horizon
- Predict 10 seconds to 5 minutes ahead
- Default: 10 minutes (120 frames at ~2fps)

## Installation

### 1. Install Dependencies

```bash
pip install scikit-learn tensorflow
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

**Note**: TensorFlow is optional. If not installed, the system automatically falls back to fast statistical methods (linear regression + exponential smoothing).

### 2. Start the Web Server

```bash
python web_app.py
```

## API Reference

### Forecast Endpoint

**URL**: `POST /api/forecast`

**Request**:
```json
{
  "method": "hybrid",  // "fast", "lstm", or "hybrid"
  "steps": 120         // 10-600 frames (default: 120)
}
```

**Response**:
```json
{
  "timestamps": [1703101234.5, 1703101235.0, ...],
  "forecast_density": [45.2, 46.1, 47.3, ...],         // % capacity
  "forecast_count": [158, 161, 165, ...],              // people count
  "confidence_upper": [52.1, 53.2, 54.5, ...],         // 95% CI upper
  "confidence_lower": [38.3, 39.0, 40.1, ...],         // 95% CI lower
  "method": "lstm",                                     // method used
  "steps": 120,
  "current_count": 150,
  "current_density": 42.86
}
```

## Usage Examples

### cURL

```bash
# Get 10-minute forecast (120 frames at ~2fps)
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"method":"hybrid","steps":120}'

# Get 5-minute forecast with fast methods only
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"method":"fast","steps":60}'

# Get LSTM-only forecast
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"method":"lstm","steps":120}'
```

### JavaScript (Frontend)

```javascript
// Fetch forecast
async function getForecast(method = 'hybrid', minutes = 10) {
  const frameRate = 2; // fps
  const steps = minutes * 60 * frameRate / 1000; // convert to frames
  
  const response = await fetch('/api/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method, steps: Math.min(600, steps) })
  });
  
  return response.json();
}

// Display forecast on chart
async function displayForecast() {
  const data = await getForecast('hybrid', 10);
  
  // Plot current series + forecast
  const chartData = {
    labels: data.timestamps,
    datasets: [{
      label: 'Forecasted Density',
      data: data.forecast_density,
      borderColor: '#43b3ff',
      borderWidth: 2,
      fill: false
    }, {
      label: 'Confidence Band',
      data: data.confidence_upper,
      borderColor: 'rgba(67, 179, 255, 0.3)',
      fill: false
    }]
  };
  
  // Update Chart.js / Plotly chart
}
```

### Python

```python
import requests

# Get forecast
response = requests.post(
    'http://localhost:5000/api/forecast',
    json={'method': 'hybrid', 'steps': 120}
)

forecast_data = response.json()
print(f"Current: {forecast_data['current_count']} people")
print(f"Predicted (10 min): {max(forecast_data['forecast_count'])} people")
print(f"Method: {forecast_data['method']}")
```

## Configuration

### ForecastingEngine Parameters

Edit `web_app.py` to customize:

```python
self.forecaster = ForecastingEngine(
    lookback_window=120,      # Use last 120 seconds of data
)

# In ForecastingEngine.__init__:
self.sequence_length = 30     # LSTM input sequence length
self.training_interval = 300  # Retrain LSTM every 5 min
```

### LSTM Model Architecture

```
- Input: 30-frame sequences
- Layer 1: LSTM(64 units) + Dropout(0.2)
- Layer 2: LSTM(32 units) + Dropout(0.2)
- Dense: 16 units
- Output: Single density value (0-1 normalized)
```

## Performance

| Method | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| Linear + Exponential | ~5ms | Good | Always available, fast |
| LSTM | ~20ms (after training) | Excellent | Requires TensorFlow, captures complex patterns |
| Hybrid | ~5-20ms | Excellent | Best overall, tries LSTM then fallback |

### LSTM Training
- Automatic retraining every 5 minutes (when 50+ data points available)
- Runs in background, non-blocking
- Trains on rolling 120-second window

## Troubleshooting

### LSTM Not Available

**Problem**: Forecast always uses "linear+exponential" method

**Solution**:
```bash
pip install tensorflow
# If on Mac/Linux with M1/M2:
pip install tensorflow-macos
```

### Slow Forecasts

- LSTM training is memory-intensive; reduce `batch_size` in `train_lstm()`
- Increase `training_interval` to retrain less often
- Use `method: 'fast'` to skip LSTM

### Inaccurate Forecasts

- Need at least 60 seconds of historical data
- Confidence intervals expand if predictions are uncertain
- Recheck `current_count` tracking accuracy first

## Technical Details

### Confidence Interval Calculation

95% confidence bands computed as:
```
upper = forecast + 1.96 * σ
lower = forecast - 1.96 * σ
```

Where σ = standard deviation of historical prediction residuals.

### LSTM Sequence Preparation

```
Input: [t-30, t-29, ..., t-1] → Output: [t]
```

The model learns patterns in 30-frame sequences to predict the next frame's density.

### Normalization

- Data scaled to [0, 1] before LSTM training using MinMaxScaler
- Denormalized to percentage (0-100%) in output

## Future Enhancements

- [ ] Prophet-based forecasting for holiday/event patterns
- [ ] Multi-step LSTM with attention mechanism
- [ ] Real-time model evaluation metrics
- [ ] Forecast visualization on web dashboard
- [ ] Anomaly detection in forecast residuals

## Support

For issues or questions:
1. Check logs: `tail -f logs/forecast.log`
2. Run test script: `python test_forecasting.py`
3. Verify data: Call `/api/stats` to check series length
