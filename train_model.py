#!/usr/bin/env python3
"""
Train Machine Learning Models for Agricultural Yield Prediction.

This script provides a straightforward way to:
1. Create sample data (if needed)
2. Load and integrate USDA and PRISM climate data
3. Train yield prediction models for major crops
4. Save trained models and visualizations
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules with error handling
try:
    from src.data.integrate_data import load_and_integrate_all_data
    from src.models.yield_prediction_models import (
        train_model, evaluate_model, plot_feature_importance,
        plot_actual_vs_predicted, plot_prediction_error
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Make sure all required modules are installed and accessible")
    MODULES_AVAILABLE = False

# Constants
CROPS = ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'RICE']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PRISM_DIR = os.path.join(PROCESSED_DIR, "prism")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def ensure_directories():
    """Create required directories if they don't exist."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(PRISM_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "model_evaluation"), exist_ok=True)
    logger.info("Created required directories")

def create_sample_usda_data(file_path=None):
    """
    Create sample USDA crop yield data.
    
    Args:
        file_path (str, optional): Path to save the sample data
        
    Returns:
        pd.DataFrame: USDA crop sample data
    """
    if file_path is None:
        file_path = os.path.join(SAMPLE_DIR, "usda_crop_data_sample.csv")
    
    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Sample USDA data already exists at {file_path}")
        return pd.read_csv(file_path)
    
    # Create sample USDA data
    logger.info("Creating sample USDA crop data...")
    
    # Census years
    years = [1997, 2002, 2007, 2012, 2017]
    
    # Create sample data for each crop
    data = []
    
    for crop in CROPS:
        for year in years:
            # Generate realistic yield values with increasing trend and some variation
            base_yield = {
                'CORN': 120,
                'SOYBEANS': 40,
                'WHEAT': 45,
                'COTTON': 750,
                'RICE': 7000
            }.get(crop, 100)
            
            # Add time trend and random variation
            trend_factor = 1.0 + (year - 1997) * 0.01  # 1% increase per census
            random_factor = np.random.uniform(0.9, 1.1)  # 10% random variation
            
            yield_value = base_yield * trend_factor * random_factor
            
            # Generate realistic harvested acres
            base_acres = {
                'CORN': 80000000,
                'SOYBEANS': 70000000,
                'WHEAT': 50000000,
                'COTTON': 10000000,
                'RICE': 3000000
            }.get(crop, 10000000)
            
            # Add time trend and random variation
            acres_trend = 1.0 + (year - 1997) * 0.005  # 0.5% increase per census
            acres_random = np.random.uniform(0.95, 1.05)  # 5% random variation
            
            harvested_acres = base_acres * acres_trend * acres_random
            
            # Calculate production
            production = yield_value * harvested_acres
            
            data.append({
                'YEAR': year,
                'CROP': crop,
                'YIELD': round(yield_value, 1),
                'HARVESTED_ACRES': int(harvested_acres),
                'PRODUCTION': int(production)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    df.to_csv(file_path, index=False)
    logger.info(f"Created sample USDA data with {len(df)} rows and saved to {file_path}")
    
    return df

def create_sample_prism_data(file_path=None):
    """
    Create sample PRISM climate data.
    
    Args:
        file_path (str, optional): Path to save the sample data
        
    Returns:
        pd.DataFrame: PRISM climate sample data
    """
    if file_path is None:
        file_path = os.path.join(PRISM_DIR, "prism_growing_season_sample.csv")
    
    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Sample PRISM data already exists at {file_path}")
        return pd.read_csv(file_path)
    
    # Create sample PRISM data
    logger.info("Creating sample PRISM climate data...")
    
    # Census years
    years = [1997, 2002, 2007, 2012, 2017]
    
    # Create sample data
    data = []
    
    for year in years:
        # Temperature base values (degrees C)
        base_tmax = 28.0  # Base max temperature
        base_tmin = 15.0  # Base min temperature
        base_tmean = 22.0  # Base mean temperature
        
        # Precipitation base value (mm)
        base_ppt = 500.0  # Base precipitation
        
        # Add time trend for climate change simulation and random variation
        year_factor = (year - 1997) / 20  # Gradual change over time
        
        # Add climate trends (warming and precipitation variability)
        tmax = base_tmax + year_factor * 1.5 + np.random.uniform(-1.0, 1.0)  # Warming trend
        tmin = base_tmin + year_factor * 1.0 + np.random.uniform(-0.8, 0.8)  # Warming trend
        tmean = base_tmean + year_factor * 1.2 + np.random.uniform(-0.9, 0.9)  # Warming trend
        
        # More variable precipitation
        precip_factor = 1.0 + year_factor * 0.1  # 10% change over 20 years
        ppt = base_ppt * precip_factor * np.random.uniform(0.8, 1.2)
        
        # Calculate derived variables
        growing_degree_days = (tmean - 10) * 150 if tmean > 10 else 0  # Simplified GDD calculation
        frost_free_days = 150 + year_factor * 5 + np.random.uniform(-5, 5)  # Increasing frost-free period
        heat_stress_days = 10 + year_factor * 3 + np.random.uniform(-2, 2)  # Increasing heat stress
        drought_index = np.random.uniform(0, 10)  # Random drought index
        
        # Add data point
        data.append({
            'year': year,
            'growing_season_tmax_mean': round(tmax, 1),
            'growing_season_tmin_mean': round(tmin, 1),
            'growing_season_tmean': round(tmean, 1),
            'growing_season_ppt_total': round(ppt, 1),
            'growing_degree_days_base10': round(growing_degree_days, 1),
            'frost_free_days': round(frost_free_days, 1),
            'heat_stress_days': round(heat_stress_days, 1),
            'drought_index': round(drought_index, 1)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    df.to_csv(file_path, index=False)
    logger.info(f"Created sample PRISM data with {len(df)} rows and saved to {file_path}")
    
    return df

def prepare_modeling_data(usda_df, prism_df):
    """
    Prepare modeling data by merging USDA and PRISM data.
    
    Args:
        usda_df (pd.DataFrame): USDA crop data
        prism_df (pd.DataFrame): PRISM climate data
        
    Returns:
        tuple: (Dictionary of crop-specific DataFrames, Combined integrated DataFrame)
    """
    logger.info("Preparing modeling data...")
    
    # Check for required columns
    required_usda_cols = ['YEAR', 'CROP', 'YIELD']
    required_prism_cols = ['year', 'growing_season_tmax_mean', 'growing_season_ppt_total']
    
    for col in required_usda_cols:
        if col not in usda_df.columns:
            logger.error(f"USDA data is missing required column: {col}")
            return {}, None
    
    for col in required_prism_cols:
        if col not in prism_df.columns:
            logger.error(f"PRISM data is missing required column: {col}")
            return {}, None
    
    # Create consistent column names
    if 'year' not in usda_df.columns:
        usda_df['year'] = usda_df['YEAR']
    
    # Dictionary to store crop-specific DataFrames
    crop_datasets = {}
    all_merged_data = []
    
    # Process each crop separately
    for crop in CROPS:
        # Filter USDA data for this crop
        crop_usda = usda_df[usda_df['CROP'] == crop].copy()
        
        if len(crop_usda) == 0:
            logger.warning(f"No USDA data found for {crop}")
            continue
        
        # Merge with climate data
        merged_df = pd.merge(
            crop_usda,
            prism_df,
            on='year',
            how='inner'
        )
        
        if len(merged_df) == 0:
            logger.warning(f"No data matches found for {crop} after merging")
            continue
        
        logger.info(f"Created dataset for {crop} with {len(merged_df)} rows")
        crop_datasets[crop] = merged_df
        all_merged_data.append(merged_df)
    
    # Create combined dataset with all crops
    if all_merged_data:
        combined_df = pd.concat(all_merged_data, ignore_index=True)
        logger.info(f"Created combined integrated dataset with {len(combined_df)} rows")
    else:
        combined_df = None
        logger.error("Failed to create any integrated datasets")
    
    return crop_datasets, combined_df

def save_integrated_datasets(crop_datasets, combined_df):
    """
    Save integrated datasets to the processed/integrated directory.
    
    Args:
        crop_datasets (dict): Dictionary of crop-specific DataFrames
        combined_df (pd.DataFrame): Combined integrated DataFrame
        
    Returns:
        bool: Success status
    """
    logger.info("Saving integrated datasets...")
    
    # Create integrated directory if it doesn't exist
    integrated_dir = os.path.join(PROCESSED_DIR, "integrated")
    os.makedirs(integrated_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save combined dataset
    if combined_df is not None:
        combined_path = os.path.join(integrated_dir, f"integrated_dataset_{timestamp}.csv")
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined integrated dataset to {combined_path}")
    
    # Save crop-specific datasets
    for crop, df in crop_datasets.items():
        crop_path = os.path.join(integrated_dir, f"{crop.lower()}_dataset_{timestamp}.csv")
        df.to_csv(crop_path, index=False)
        logger.info(f"Saved {crop} integrated dataset to {crop_path}")
    
    return True

def train_and_evaluate_models(crop_datasets, model_type="random_forest"):
    """
    Train and evaluate models for each crop.
    
    Args:
        crop_datasets (dict): Dictionary of crop-specific DataFrames
        model_type (str): Type of model to train
        
    Returns:
        dict: Dictionary of results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    
    logger.info(f"Training {model_type} models for {len(crop_datasets)} crops...")
    
    # Dictionary to store results
    results = {}
    
    # Train model for each crop
    for crop, df in crop_datasets.items():
        logger.info(f"Training model for {crop}...")
        
        # Prepare features and target
        if 'YIELD' in df.columns:
            y = df['YIELD']
        else:
            logger.error(f"No yield column found for {crop}")
            continue
        
        # Select climate features
        climate_cols = [col for col in df.columns if any(
            term in col.lower() for term in ['tmax', 'tmin', 'tmean', 'ppt', 'heat', 'frost', 'growing', 'drought']
        )]
        
        if not climate_cols:
            logger.error(f"No climate columns found for {crop}")
            continue
        
        # Create feature matrix
        X = df[climate_cols]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            continue
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{crop} model: RMSE = {rmse:.2f}, R² = {r2:.2f}")
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_{model_type}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {crop} model to {model_path}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': climate_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create feature importance visualization
        fig_importance = plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for {crop}')
        plt.tight_layout()
        
        # Save feature importance plot
        importance_path = os.path.join(RESULTS_DIR, "feature_importance", f"{crop.lower()}_importance_{timestamp}.png")
        plt.savefig(importance_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {importance_path}")
        
        # Create actual vs predicted plot
        fig_actual_pred = plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(f'Actual vs Predicted Yield for {crop}')
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
        plt.tight_layout()
        
        # Save actual vs predicted plot
        actual_pred_path = os.path.join(RESULTS_DIR, "model_evaluation", f"{crop.lower()}_actual_vs_pred_{timestamp}.png")
        plt.savefig(actual_pred_path)
        plt.close()
        logger.info(f"Saved actual vs predicted plot to {actual_pred_path}")
        
        # Store results
        results[crop] = {
            'model': model,
            'model_path': model_path,
            'feature_importance': feature_importance,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            },
            'features': climate_cols
        }
    
    return results

def main():
    """Main function to run the model training pipeline."""
    parser = argparse.ArgumentParser(description="Train crop yield prediction models")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting", "elasticnet"], 
                        default="random_forest", help="Type of model to train")
    parser.add_argument("--force-sample", action="store_true", 
                        help="Force creation of new sample data even if files exist")
    parser.add_argument("--tune-hyperparameters", action="store_true",
                        help="Tune model hyperparameters (takes longer)")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model training and only create integrated datasets")
    
    args = parser.parse_args()
    
    # Create directories
    ensure_directories()
    
    # Create sample data
    if args.force_sample:
        # Remove existing sample files if forcing new samples
        usda_sample_path = os.path.join(SAMPLE_DIR, "usda_crop_data_sample.csv")
        prism_sample_path = os.path.join(PRISM_DIR, "prism_growing_season_sample.csv")
        
        if os.path.exists(usda_sample_path):
            os.remove(usda_sample_path)
        
        if os.path.exists(prism_sample_path):
            os.remove(prism_sample_path)
    
    # Create sample data
    usda_df = create_sample_usda_data()
    prism_df = create_sample_prism_data()
    
    if usda_df.empty or prism_df.empty:
        logger.error("Failed to create sample data")
        return
    
    # Prepare modeling data
    crop_datasets, combined_df = prepare_modeling_data(usda_df, prism_df)
    
    if not crop_datasets:
        logger.error("Failed to prepare modeling data")
        return
    
    # Save integrated datasets
    if combined_df is not None:
        save_integrated_datasets(crop_datasets, combined_df)
        logger.info("Created integrated datasets for the Streamlit app")
    
    # Skip model training if requested
    if args.skip_models:
        logger.info("Skipping model training as requested")
        print("\nIntegrated datasets created successfully! You can now run the Streamlit app.")
        return
    
    # Train and evaluate models
    try:
        results = train_and_evaluate_models(crop_datasets, model_type=args.model_type)
        
        if results:
            logger.info(f"Successfully trained models for {len(results)} crops")
            
            # Print summary of results
            print("\nModel Training Results Summary:")
            print("=" * 50)
            for crop, result in results.items():
                print(f"\n{crop}:")
                print(f"  RMSE: {result['metrics']['rmse']:.2f}")
                print(f"  R²: {result['metrics']['r2']:.2f}")
                print(f"  Model saved to: {result['model_path']}")
                print(f"  Top 3 important features:")
                for i, (feature, importance) in enumerate(zip(
                    result['feature_importance']['feature'].head(3),
                    result['feature_importance']['importance'].head(3)
                )):
                    print(f"    {i+1}. {feature}: {importance:.4f}")
            
            # Check for additional dependencies
            statsmodels_available = False
            try:
                import statsmodels
                statsmodels_available = True
            except ImportError:
                pass
                
            print("\nAll done! The Streamlit app should now be able to:")
            print("1. Show historical yield analysis")
            print("2. Display climate-yield correlations")
            print("3. Show model evaluations and feature importance")
            print("4. Create climate change scenarios")
            
            # Print dependency warnings if needed
            if not statsmodels_available:
                print("\nNOTE: For full Climate-Yield Correlation functionality, install statsmodels:")
                print("pip install statsmodels")
            
            print("\nIf you encounter any issues, check the TROUBLESHOOTING.md file for solutions.")
        else:
            logger.error("No models were successfully trained")
    except Exception as e:
        logger.error(f"Error training models: {e}")

if __name__ == "__main__":
    main()