#!/usr/bin/env python3
"""
Simple model training script to create models specifically for climate scenario feature.
This script creates basic regression models using synthetic data that are compatible
with the climate scenario prediction page.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crops to generate models for
CROPS = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "RICE"]

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("models", exist_ok=True)
    logger.info("Ensured models directory exists")

def create_synthetic_data(n_samples=100):
    """
    Create synthetic climate-yield data for model training.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of X (features) and y (targets) DataFrames
    """
    # Generate random feature values
    np.random.seed(42)  # For reproducibility
    
    # Generate base features
    data = {
        "temperature": np.random.uniform(15, 30, n_samples),
        "precipitation": np.random.uniform(300, 800, n_samples),
        "soil_quality": np.random.uniform(3, 9, n_samples),
        "growing_season": np.random.uniform(90, 180, n_samples),
        "elevation": np.random.uniform(50, 500, n_samples),
        "fertilizer": np.random.uniform(20, 80, n_samples)
    }
    
    # Create DataFrame
    X = pd.DataFrame(data)
    
    # Generate synthetic yields for each crop
    y_crops = {}
    
    # Corn: Prefers moderate temperatures, good rainfall
    y_crops["CORN"] = (
        120 
        + 2.5 * (X["temperature"] - 25).abs() 
        + 0.05 * X["precipitation"]
        + 2 * X["soil_quality"]
        + 0.1 * X["growing_season"]
        + 0.01 * X["fertilizer"]
        + np.random.normal(0, 5, n_samples)
    )
    
    # Soybeans: Similar to corn but less sensitive to temperature
    y_crops["SOYBEANS"] = (
        40 
        + 1.0 * (X["temperature"] - 23).abs()
        + 0.02 * X["precipitation"]
        + 1.5 * X["soil_quality"]
        + 0.05 * X["growing_season"]
        + 0.02 * X["fertilizer"]
        + np.random.normal(0, 2, n_samples)
    )
    
    # Wheat: Better in cooler temperatures, less water
    y_crops["WHEAT"] = (
        50 
        - 1.5 * (X["temperature"] - 20)
        + 0.01 * X["precipitation"]
        + 2 * X["soil_quality"]
        + 0.03 * X["growing_season"]
        + 0.015 * X["fertilizer"]
        + np.random.normal(0, 3, n_samples)
    )
    
    # Cotton: Prefers warm and dry
    y_crops["COTTON"] = (
        800
        + 10 * (X["temperature"] - 18)
        - 0.05 * (X["precipitation"] - 400)
        + 15 * X["soil_quality"]
        + 0.5 * X["growing_season"]
        + 0.2 * X["fertilizer"]
        + np.random.normal(0, 30, n_samples)
    )
    
    # Rice: Needs more water
    y_crops["RICE"] = (
        7000
        + 50 * (X["temperature"] - 22)
        + 0.5 * X["precipitation"]
        + 100 * X["soil_quality"]
        + 5 * X["growing_season"]
        + 2 * X["fertilizer"]
        + np.random.normal(0, 200, n_samples)
    )
    
    return X, y_crops

def train_and_save_models():
    """Train synthetic models and save them for use in the scenario page."""
    ensure_directories()
    
    # Generate synthetic data
    logger.info("Generating synthetic training data")
    X, y_crops = create_synthetic_data(n_samples=200)
    
    # Train models for each crop
    for crop in CROPS:
        logger.info(f"Training model for {crop}")
        
        # Train a Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X, y_crops[crop])
        
        # Save model
        model_path = Path("models") / f"{crop.lower()}_random_forest_climate.joblib"
        joblib.dump(rf_model, model_path)
        logger.info(f"Saved {crop} model to {model_path}")
        
        # Also train a Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X, y_crops[crop])
        
        # Save linear model
        lr_path = Path("models") / f"{crop.lower()}_linear_climate.joblib"
        joblib.dump(lr_model, lr_path)
        logger.info(f"Saved {crop} linear model to {lr_path}")
    
    logger.info("All models trained and saved successfully!")
    print(f"Created {len(CROPS) * 2} models for use in climate scenario analysis")

if __name__ == "__main__":
    train_and_save_models()