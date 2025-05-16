#!/usr/bin/env python3
"""
Machine learning models for crop yield prediction based on climate variables.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths for saving models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def prepare_data_for_modeling(df, target='YIELD', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into training and testing sets.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with features
        target (str): Target variable name
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    # Ensure target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    # Identify numeric features (exclude the target and non-numeric columns)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [f for f in numeric_features if f != target and not pd.isna(df[f]).any()]
    
    # Identify categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove any columns with missing values
    for col in df.columns:
        if df[col].isna().any():
            if col in numeric_features:
                numeric_features.remove(col)
            elif col in categorical_features:
                categorical_features.remove(col)
    
    # Define features and target
    X = df[numeric_features + categorical_features].copy()
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Numeric features preprocessing
    numeric_transformer = StandardScaler()
    
    # Categorical features preprocessing
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor

def train_random_forest_model(X_train, y_train, numeric_features, categorical_features, 
                             n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed for reproducibility
        
    Returns:
        Pipeline: Trained model pipeline
    """
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    
    model.fit(X_train, y_train)
    
    return model

def train_gradient_boosting_model(X_train, y_train, numeric_features, categorical_features,
                                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Train a Gradient Boosting regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        n_estimators (int): Number of boosting stages
        learning_rate (float): Learning rate
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed for reproducibility
        
    Returns:
        Pipeline: Trained model pipeline
    """
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    
    model.fit(X_train, y_train)
    
    return model

def train_linear_regression_model(X_train, y_train, numeric_features, categorical_features):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        Pipeline: Trained model pipeline
    """
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Create a DataFrame of actual vs predicted values
    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': y_test - y_pred,
        'Percent_Error': ((y_test - y_pred) / y_test) * 100
    })
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': prediction_df
    }

def get_feature_importance(model, numeric_features, categorical_features):
    """
    Extract feature importance from a trained model.
    
    Args:
        model (Pipeline): Trained model pipeline
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        pd.DataFrame: Feature importance data
    """
    # Extract the regressor
    regressor = model.named_steps['regressor']
    
    # Check if the model has feature_importances_ attribute
    if not hasattr(regressor, 'feature_importances_'):
        return pd.DataFrame(columns=['Feature', 'Importance'])
    
    # Get feature importance
    importance = regressor.feature_importances_
    
    # Get feature names from preprocessor
    preprocessor = model.named_steps['preprocessor']
    
    # Get transformed feature names
    one_hot_encoder = preprocessor.named_transformers_['cat']
    
    # Get categorical feature names after one-hot encoding
    categorical_feature_names = []
    if categorical_features:
        categorical_feature_names = list(one_hot_encoder.get_feature_names_out(categorical_features))
    
    # Combine all feature names
    feature_names = numeric_features + categorical_feature_names
    
    # Create DataFrame of feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return importance_df

def save_model(model, model_name):
    """
    Save a trained model to disk.
    
    Args:
        model (Pipeline): Trained model pipeline
        model_name (str): Name to save the model as
        
    Returns:
        str: Path to the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    return model_path

def load_model(model_name):
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        Pipeline: Loaded model pipeline
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return model

def train_crop_specific_models(df, crops, features, target='YIELD'):
    """
    Train separate models for each crop type.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with features
        crops (list): List of crop types to train models for
        features (list): List of feature names to use
        target (str): Target variable name
        
    Returns:
        dict: Dictionary of trained models by crop
    """
    crop_models = {}
    
    for crop in crops:
        print(f"Training models for {crop}...")
        
        # Filter data for this crop
        crop_data = df[df['CROP'] == crop].copy()
        
        # Skip if not enough data
        if len(crop_data) < 10:  # Need enough data points to split and train
            print(f"Not enough data for {crop}, skipping.")
            continue
        
        # Prepare data
        numeric_features = [f for f in features if f in crop_data.columns and 
                          crop_data[f].dtype in ['int64', 'float64'] and
                          not pd.isna(crop_data[f]).any()]
        
        categorical_features = [f for f in crop_data.columns if 
                               f not in numeric_features + [target] and
                               crop_data[f].dtype in ['object', 'category'] and
                               not pd.isna(crop_data[f]).any()]
        
        # Split data
        X_train, X_test, y_train, y_test, _, _ = prepare_data_for_modeling(
            crop_data, target=target, test_size=0.2
        )
        
        # Train models
        rf_model = train_random_forest_model(
            X_train, y_train, numeric_features, categorical_features
        )
        
        gb_model = train_gradient_boosting_model(
            X_train, y_train, numeric_features, categorical_features
        )
        
        lr_model = train_linear_regression_model(
            X_train, y_train, numeric_features, categorical_features
        )
        
        # Evaluate models
        rf_eval = evaluate_model(rf_model, X_test, y_test)
        gb_eval = evaluate_model(gb_model, X_test, y_test)
        lr_eval = evaluate_model(lr_model, X_test, y_test)
        
        # Get feature importance
        rf_importance = get_feature_importance(rf_model, numeric_features, categorical_features)
        gb_importance = get_feature_importance(gb_model, numeric_features, categorical_features)
        
        # Save models
        rf_path = save_model(rf_model, f"{crop.lower()}_rf_model")
        gb_path = save_model(gb_model, f"{crop.lower()}_gb_model")
        lr_path = save_model(lr_model, f"{crop.lower()}_lr_model")
        
        # Store model info
        crop_models[crop] = {
            'random_forest': {
                'model': rf_model,
                'path': rf_path,
                'evaluation': rf_eval,
                'importance': rf_importance
            },
            'gradient_boosting': {
                'model': gb_model,
                'path': gb_path,
                'evaluation': gb_eval,
                'importance': gb_importance
            },
            'linear_regression': {
                'model': lr_model,
                'path': lr_path,
                'evaluation': lr_eval
            }
        }
        
        print(f"Models for {crop} trained and saved successfully.")
    
    return crop_models

def predict_yield_for_scenario(crop, model_type, climate_scenario):
    """
    Predict crop yield for a given climate scenario.
    
    Args:
        crop (str): Crop type to predict for
        model_type (str): Type of model to use ('random_forest', 'gradient_boosting', 'linear_regression')
        climate_scenario (dict): Climate scenario variables
        
    Returns:
        float: Predicted yield
    """
    # Load the appropriate model
    model_name = f"{crop.lower()}_{model_type.split('_')[0]}_model"
    
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise ValueError(f"No trained model found for {crop} using {model_type}")
    
    # Convert scenario to DataFrame
    scenario_df = pd.DataFrame([climate_scenario])
    
    # Make prediction
    predicted_yield = model.predict(scenario_df)[0]
    
    return predicted_yield

def simulate_climate_scenarios(base_scenario, crops, model_types, temp_range=(-2, 5), precip_range=(-30, 30)):
    """
    Simulate crop yields under various climate scenarios.
    
    Args:
        base_scenario (dict): Base climate scenario to modify
        crops (list): List of crops to simulate
        model_types (list): List of model types to use
        temp_range (tuple): Range of temperature changes to simulate (min, max)
        precip_range (tuple): Range of precipitation changes to simulate (min, max)
        
    Returns:
        pd.DataFrame: Simulation results
    """
    # Generate temperature and precipitation changes
    temp_steps = np.linspace(temp_range[0], temp_range[1], 8)  # 8 temperature steps
    precip_steps = np.linspace(precip_range[0], precip_range[1], 7)  # 7 precipitation steps
    
    # List to store simulation results
    results = []
    
    for crop in crops:
        for model_type in model_types:
            for temp_change in temp_steps:
                for precip_change in precip_steps:
                    # Create modified scenario
                    scenario = base_scenario.copy()
                    
                    # Modify temperature-related features
                    if 'AVG_GDD' in scenario:
                        # Approximate effect of temperature change on GDD
                        scenario['AVG_GDD'] *= (1 + temp_change / 20)  # Rough approximation
                    
                    # Modify precipitation-related features
                    if 'GROWING_SEASON_PRCP' in scenario:
                        scenario['GROWING_SEASON_PRCP'] *= (1 + precip_change / 100)
                    
                    # Update interaction term if it exists
                    if 'GDD_PRCP_INTERACTION' in scenario and 'AVG_GDD' in scenario and 'GROWING_SEASON_PRCP' in scenario:
                        scenario['GDD_PRCP_INTERACTION'] = scenario['AVG_GDD'] * scenario['GROWING_SEASON_PRCP']
                    
                    try:
                        # Predict yield
                        predicted_yield = predict_yield_for_scenario(crop, model_type, scenario)
                        
                        # Store result
                        results.append({
                            'Crop': crop,
                            'ModelType': model_type,
                            'TemperatureChange': temp_change,
                            'PrecipitationChange': precip_change,
                            'PredictedYield': predicted_yield
                        })
                    except Exception as e:
                        print(f"Error predicting {crop} yield with {model_type}: {str(e)}")
    
    # Convert to DataFrame
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test code to verify functionality
    import os
    from src.features.feature_engineering import create_climate_features, merge_climate_crop_data, engineer_features_for_modeling
    
    # Sample paths
    sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample")
    usda_sample_path = os.path.join(sample_data_dir, "usda_crop_data_sample.csv")
    noaa_sample_path = os.path.join(sample_data_dir, "noaa_climate_normals_sample.csv")
    
    # Load sample data
    try:
        usda_data = pd.read_csv(usda_sample_path)
        noaa_data = pd.read_csv(noaa_sample_path)
        
        # Convert date column
        noaa_data['DATE'] = pd.to_datetime(noaa_data['DATE'])
        
        # Create climate features
        climate_features = create_climate_features(noaa_data)
        
        # Merge data
        merged_data = merge_climate_crop_data(usda_data, climate_features)
        
        # Engineer features
        final_features = engineer_features_for_modeling(merged_data)
        
        # Select a subset of the data for testing
        test_data = final_features[final_features['CROP'] == 'CORN'].copy()
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_data_for_modeling(
            test_data, target='YIELD'
        )
        
        # Train a model
        rf_model = train_random_forest_model(
            X_train, y_train, numeric_features, categorical_features, n_estimators=10
        )
        
        # Evaluate the model
        rf_eval = evaluate_model(rf_model, X_test, y_test)
        
        print("Model training successful!")
        print(f"R² Score: {rf_eval['r2']:.4f}")
        print(f"RMSE: {rf_eval['rmse']:.4f}")
        
    except FileNotFoundError:
        print("Sample data files not found. Please run the data download scripts first.")