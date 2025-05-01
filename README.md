# Freezer Gambit Forecasting

## Overview
Time series forecasting model for predicting agricultural commodity prices using weather and historical price data.

## Features
- Weekly price aggregation
- Weather feature engineering
- Time series cross-validation
- Model persistence
- Incremental model updates with new data
- Feature importance analysis using permutation importance

## Model Components
- Feature Engineering:
  - Lag features (1-3 weeks)
  - Rolling statistics (mean, std, min, max)
  - Seasonal decomposition
  - Weather-price interactions
  - Yield-price ratios
  - Volatility metrics (price and yield)
- Validation Strategy:
  - Temporal train-test split
  - Time series cross-validation
  - Hold-out validation period
- Performance Metrics:
  - RMSE (Root Mean Square Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score
- Model Update:
  - Data drift detection
  - Incremental updates for supported models
  - Full retraining when necessary

## Setup
1. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```
2. Install requirements: 
```bash
pip install -r requirements.txt
```
3. Run training: 
```bash
python model.py
```
4. Use API: 
```bash
uvicorn main:app --reload
```

## Project Structure
```
freezer_gambit_forecasting/
├── model.py          # Main training pipeline
├── main.py          # FastAPI endpoints
├── cleanup.py       # Data cleaning utilities
├── WeatherData/     # Weather training data
├── PriceData/      # Price training data
└── api.yml         # API documentation
```

## Error Handling
- Robust error handling for data processing
- Validation checks for input data
- Logging of model performance metrics
- Exception handling for API endpoints

## Future Improvements
- [ ] Add more domain-specific features
- [ ] Implement ensemble methods
- [ ] Add confidence intervals for predictions
- [ ] Expand API documentation