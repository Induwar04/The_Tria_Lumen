from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from datetime import datetime
import logging
from model import ModelManager, predict_with_model, train_and_save

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Freezer Gambit Forecasting API")
model_manager = ModelManager()

class WeatherData(BaseModel):
    date: str
    region: str
    temperature: float
    rainfall: float
    humidity: float
    yield_impact: float

class PriceData(BaseModel):
    date: str
    region: str
    commodity: str
    type: str
    price: float

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up application...")
    
    # Load and validate data
    try:
        weather_data = pd.read_csv("WeatherData/train_data.csv", parse_dates=["Date"])
        price_data = pd.read_csv("PriceData/train_data.csv", parse_dates=["Date"])
        
        logger.info(f"Available regions: {weather_data['Region'].unique()}")
        logger.info(f"Available commodities: {price_data['Commodity'].unique()}")
        
        if not model_manager.load_model():
            logger.info("Training new model...")
            train_and_save()
            model_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API status"""
    return {
        "status": "online",
        "model_ready": model_manager.is_ready(),
        "uptime": model_manager.get_uptime(),
        "version": model_manager.get_version(),
        "request_count": model_manager.get_request_count()
    }

@app.get("/forecast/{region}/{commodity}")
async def get_forecast(region: str, commodity: str):
    """Get price forecast for region and commodity."""
    try:
        if not model_manager.is_ready():
            raise HTTPException(status_code=503, detail="Model not ready")

        # Load data
        weather_data = pd.read_csv("WeatherData/train_data.csv", parse_dates=["Date"])
        price_data = pd.read_csv("PriceData/train_data.csv", parse_dates=["Date"])

        # Prepare features
        features = model_manager.prepare_features(weather_data, price_data, region, commodity)
        logger.info(f"Prepared features for {region} - {commodity}:\n{features}")

        # Generate predictions
        predictions = model_manager.predict_sequence(features, n_steps=4)
        logger.info(f"Predictions for {region} - {commodity}: {predictions}")

        # Calculate RMSE for each week (assuming actual values are available)
        actual_prices = price_data[
            (price_data["Region"] == region) & (price_data["Commodity"] == commodity)
        ].sort_values(by="Date")["Price per Unit (Silver Drachma/kg)"].values[:4]

        rmse_values = []
        for i, (pred, _) in enumerate(predictions):
            if i < len(actual_prices):
                rmse = ((pred - actual_prices[i]) ** 2) ** 0.5
                rmse_values.append(rmse)
            else:
                rmse_values.append(None)  # No actual value available for this week

        # Format response
        forecast = [
            {
                "week": i + 1,
                "predicted_price": pred,
                "confidence": conf,
                "rmse": rmse_values[i]
            }
            for i, (pred, conf) in enumerate(predictions)
        ]

        return {
            "region": region,
            "commodity": commodity,
            "forecast": forecast,
            "model_version": model_manager.get_version()
        }

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/data/weather")
async def Set_new_weather_data(data: List[WeatherData]):
    """Submit new weather data"""
    try:
        # Convert to DataFrame
        new_data = pd.DataFrame([{
            'Date': item.date,
            'Region': item.region,
            'Temperature (K)': item.temperature,
            'Rainfall (mm)': item.rainfall,
            'Humidity (%)': item.humidity,
            'Crop Yield Impact Score': item.yield_impact
        } for item in data])
        
        # Load existing data
        existing_data = pd.read_csv("WeatherData/train_data.csv", parse_dates=["Date"])
        
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Remove duplicates based on Date and Region
        updated_data = updated_data.drop_duplicates(subset=['Date', 'Region'], keep='last')
        
        # Save updated dataset
        updated_data.to_csv("WeatherData/train_data.csv", index=False)
        
        # Update model if needed
        if model_manager.should_update():
            model_manager.update_model(new_data)
        
        return {
            "status": "success",
            "records_added": len(new_data),
            "total_records": len(updated_data)
        }
        
    except Exception as e:
        logger.error(f"Error updating weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/prices")
async def Set_new_price_data(data: List[PriceData]):
    """Submit new price data"""
    try:
        # Convert to DataFrame
        new_data = pd.DataFrame([{
            'Date': item.date,
            'Region': item.region,
            'Commodity': item.commodity,
            'Type': item.type,
            'Price per Unit (Silver Drachma/kg)': item.price
        } for item in data])
        
        # Load existing data
        existing_data = pd.read_csv("PriceData/train_data.csv", parse_dates=["Date"])
        
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Remove duplicates based on Date, Region, and Commodity
        updated_data = updated_data.drop_duplicates(
            subset=['Date', 'Region', 'Commodity'], 
            keep='last'
        )
        
        # Save updated dataset
        updated_data.to_csv("PriceData/train_data.csv", index=False)
        
        # Update model if needed
        if model_manager.should_update():
            model_manager.update_model(new_data)
        
        return {
            "status": "success",
            "records_added": len(new_data),
            "total_records": len(updated_data)
        }
        
    except Exception as e:
        logger.error(f"Error updating price data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get model metrics"""
    return model_manager.get_metrics()