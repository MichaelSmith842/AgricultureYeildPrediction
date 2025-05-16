#!/usr/bin/env python3
"""
Module for training and evaluating crop yield prediction models using climate data.
Implements multiple machine learning models to predict crop yields based on climate variables.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Import SHAP if available (not required)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not available. Feature interaction analysis will be disabled.")
    logging.warning("To enable SHAP, install with: pip install shap")

# Import project modules
from src.data.integrate_data import load_and_integrate_all_data, create_modeling_ready_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "model_evaluation"), exist_ok=True)

def prepare_data_for_modeling(integrated_df, crop_name=None, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by creating train/test splits.
    
    Args:
        integrated_df (pd.DataFrame): Integrated dataset
        crop_name (str, optional): Specific crop to model
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    # Filter for specific crop if requested
    if crop_name and 'crop' in integrated_df.columns:
        crop_df = integrated_df[integrated_df['crop'] == crop_name].copy()
    else:
        crop_df = integrated_df.copy()
    
    if len(crop_df) == 0:
        logger.error(f"No data available for crop: {crop_name}")
        return None, None, None, None, None
    
    # Select target variable
    if 'yield' in crop_df.columns:
        target = 'yield'
    elif 'YIELD' in crop_df.columns:
        target = 'YIELD'
    else:
        logger.error("Yield column not found in dataset")
        return None, None, None, None, None
    
    # Select features
    # Prefer standardized columns when available
    std_cols = [col for col in crop_df.columns if col.endswith('_std')]
    
    # If no standardized columns, use all numeric columns except target and identifiers
    if not std_cols:
        feature_cols = crop_df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in feature_cols 
                      if col != target and not col.startswith(('YEAR', 'year', 'STATE', 'state'))]
    else:
        feature_cols = std_cols
    
    # Handle categorical variables
    cat_cols = crop_df.select_dtypes(include=['category']).columns.tolist()
    
    # Create dummy variables for categorical columns
    if cat_cols:
        crop_df = pd.get_dummies(crop_df, columns=cat_cols, drop_first=True)
        
        # Add new dummy columns to feature list
        dummy_cols = [col for col in crop_df.columns 
                    if any(col.startswith(c + '_') for c in cat_cols)]
        feature_cols.extend(dummy_cols)
    
    # Create feature and target arrays
    X = crop_df[feature_cols]
    y = crop_df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Prepared data with {len(X_train)} training samples and {len(X_test)} test samples")
    logger.info(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train, model_type="random_forest", params=None, cv=5):
    """
    Train a machine learning model on the provided data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        model_type (str): Type of model to train
        params (dict, optional): Model hyperparameters
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Trained model
    """
    if params is None:
        params = {}
    
    logger.info(f"Training {model_type} model")
    
    # Create model based on type
    if model_type == "random_forest":
        # Default parameters
        rf_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', None),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
            'random_state': params.get('random_state', 42)
        }
        model = RandomForestRegressor(**rf_params)
    
    elif model_type == "gradient_boosting":
        # Default parameters
        gb_params = {
            'n_estimators': params.get('n_estimators', 100),
            'learning_rate': params.get('learning_rate', 0.1),
            'max_depth': params.get('max_depth', 3),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
            'random_state': params.get('random_state', 42)
        }
        model = GradientBoostingRegressor(**gb_params)
    
    elif model_type == "elasticnet":
        # Default parameters
        en_params = {
            'alpha': params.get('alpha', 1.0),
            'l1_ratio': params.get('l1_ratio', 0.5),
            'random_state': params.get('random_state', 42)
        }
        model = ElasticNet(**en_params)
    
    elif model_type == "neural_network":
        # Default parameters
        nn_params = {
            'hidden_layer_sizes': params.get('hidden_layer_sizes', (100, 50)),
            'activation': params.get('activation', 'relu'),
            'solver': params.get('solver', 'adam'),
            'alpha': params.get('alpha', 0.0001),
            'learning_rate': params.get('learning_rate', 'adaptive'),
            'max_iter': params.get('max_iter', 500),
            'random_state': params.get('random_state', 42)
        }
        model = MLPRegressor(**nn_params)
    
    elif model_type == "svr":
        # Default parameters
        svr_params = {
            'kernel': params.get('kernel', 'rbf'),
            'C': params.get('C', 1.0),
            'epsilon': params.get('epsilon', 0.1),
            'gamma': params.get('gamma', 'scale')
        }
        model = SVR(**svr_params)
    
    elif model_type == "linear":
        model = LinearRegression()
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate cross-validation performance
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
    logger.info(f"Cross-validation R² scores: {cv_scores}")
    logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return pipeline

def hyperparameter_tuning(X_train, y_train, model_type="random_forest", n_iter=20, cv=5):
    """
    Tune model hyperparameters using randomized search.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        model_type (str): Type of model to tune
        n_iter (int): Number of parameter settings to try
        cv (int): Number of cross-validation folds
        
    Returns:
        tuple: (best_params, best_model)
    """
    logger.info(f"Tuning hyperparameters for {model_type} model")
    
    # Define parameter spaces based on model type
    if model_type == "random_forest":
        model = RandomForestRegressor(random_state=42)
        param_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
        param_space = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_type == "elasticnet":
        model = ElasticNet(random_state=42)
        param_space = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    
    elif model_type == "neural_network":
        model = MLPRegressor(random_state=42)
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    
    elif model_type == "svr":
        model = SVR()
        param_space = {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Set up randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={'model__' + key: value for key, value in param_space.items()},
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = {key.replace('model__', ''): value for key, value in search.best_params_.items()}
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV R² score: {search.best_score_:.4f}")
    
    return best_params, search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate a model on test data and return performance metrics.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        feature_names (list, optional): Feature names
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        # Calculate manually if sklearn version doesn't have MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    logger.info(f"Model evaluation metrics:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    
    # Create DataFrame with actual vs predicted values
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'pct_error': ((y_test - y_pred) / y_test) * 100
    })
    
    # Extract feature importance if available
    feature_importance = {}
    
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        
        if feature_names is not None:
            # Create DataFrame with feature names and importances
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    return {
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'predictions': predictions_df,
        'feature_importance': feature_importance
    }

def plot_feature_importance(feature_importance, crop_name=None, top_n=20, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with feature importance
        crop_name (str, optional): Crop name for title
        top_n (int): Number of top features to show
        save_path (str, optional): Path to save the plot
        
    Returns:
        plt.Figure: Figure object
    """
    if feature_importance is None or len(feature_importance) == 0:
        logger.warning("No feature importance data available")
        return None
    
    # Limit to top N features
    if len(feature_importance) > top_n:
        plot_data = feature_importance.head(top_n)
    else:
        plot_data = feature_importance
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    bars = ax.barh(plot_data['feature'], plot_data['importance'])
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    title = 'Feature Importance'
    if crop_name:
        title = f'{title} for {crop_name}'
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig

def plot_actual_vs_predicted(predictions_df, crop_name=None, save_path=None):
    """
    Plot actual vs predicted values.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with actual and predicted values
        crop_name (str, optional): Crop name for title
        save_path (str, optional): Path to save the plot
        
    Returns:
        plt.Figure: Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter plot
    ax.scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('Actual Yield')
    ax.set_ylabel('Predicted Yield')
    title = 'Actual vs Predicted Yield'
    if crop_name:
        title = f'{title} for {crop_name}'
    ax.set_title(title)
    
    # Add R² value
    r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Actual vs predicted plot saved to {save_path}")
    
    return fig

def plot_prediction_error(predictions_df, crop_name=None, save_path=None):
    """
    Plot prediction error distribution.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with prediction errors
        crop_name (str, optional): Crop name for title
        save_path (str, optional): Path to save the plot
        
    Returns:
        plt.Figure: Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot error distribution
    sns.histplot(predictions_df['pct_error'], bins=30, kde=True, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Percent Error (%)')
    ax.set_ylabel('Frequency')
    title = 'Prediction Error Distribution'
    if crop_name:
        title = f'{title} for {crop_name}'
    ax.set_title(title)
    
    # Add statistics
    mean_error = predictions_df['pct_error'].mean()
    median_error = predictions_df['pct_error'].median()
    ax.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}%')
    ax.axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}%')
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Error distribution plot saved to {save_path}")
    
    return fig

def analyze_feature_interactions(model, X_test, feature_names, crop_name=None, save_path=None):
    """
    Analyze feature interactions using SHAP values.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        feature_names (list): Feature names
        crop_name (str, optional): Crop name for title
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: (shap_values, plt.Figure)
    """
    # Check if SHAP is available
    if not SHAP_AVAILABLE:
        logger.warning("SHAP library not available. Cannot perform feature interaction analysis.")
        # If SHAP is not available, create a basic feature importance plot instead
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            
            # Create a simple feature importance plot
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(feature_names))
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            
            # Plot top 15 features
            top_n = min(15, len(sorted_names))
            ax.barh(np.arange(top_n), sorted_importances[:top_n])
            ax.set_yticks(np.arange(top_n))
            ax.set_yticklabels(sorted_names[:top_n])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance for {crop_name}' if crop_name else 'Feature Importance')
            
            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Feature importance plot saved to {save_path}")
            
            return None, fig
        return None, None
    
    # Extract the underlying model from pipeline
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps['model']
    else:
        estimator = model
    
    # Check if model is supported by SHAP
    supported_models = (RandomForestRegressor, GradientBoostingRegressor, SVR, LinearRegression)
    if not isinstance(estimator, supported_models):
        logger.warning(f"SHAP analysis not supported for model type: {type(estimator)}")
        return None, None
    
    try:
        # Transform data through pipeline without final prediction
        X_test_transformed = X_test.copy()
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            X_test_transformed = pd.DataFrame(
                model.named_steps['scaler'].transform(X_test),
                columns=X_test.columns
            )
        
        # Create explainer
        explainer = shap.Explainer(estimator, X_test_transformed)
        
        # Calculate SHAP values
        shap_values = explainer(X_test_transformed)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        title = 'SHAP Feature Importance'
        if crop_name:
            title = f'{title} for {crop_name}'
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
        plt.title(title)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        return shap_values, plt.gcf()
    
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
        return None, None

def train_all_crop_models(merge_level="national", tune_hyperparameters=False, model_type="random_forest"):
    """
    Train models for all crops in the dataset.
    
    Args:
        merge_level (str): Level at which to merge data
        tune_hyperparameters (bool): Whether to tune hyperparameters
        model_type (str): Type of model to train
        
    Returns:
        dict: Dictionary of trained models and evaluation results
    """
    ensure_directories()
    
    # Load and integrate data
    integrated_df, crop_datasets = load_and_integrate_all_data(merge_level=merge_level)
    
    if integrated_df.empty:
        logger.error("Failed to load integrated data")
        return {}
    
    # Get unique crops
    if 'crop' in integrated_df.columns:
        unique_crops = integrated_df['crop'].unique()
    elif 'CROP' in integrated_df.columns:
        unique_crops = integrated_df['CROP'].unique()
        # Create lowercase 'crop' column for consistency
        integrated_df['crop'] = integrated_df['CROP']
    else:
        logger.error("No crop column found in dataset")
        return {}
    
    # Dictionary to store models and results
    results = {}
    
    # Train model for each crop
    for crop in unique_crops:
        logger.info(f"Training model for {crop}")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = prepare_data_for_modeling(
            integrated_df, crop_name=crop
        )
        
        if X_train is None:
            logger.warning(f"Skipping {crop} due to data preparation issues")
            continue
        
        # Train model
        if tune_hyperparameters:
            logger.info(f"Tuning hyperparameters for {crop} model")
            best_params, model = hyperparameter_tuning(
                X_train, y_train, model_type=model_type
            )
        else:
            model = train_model(
                X_train, y_train, model_type=model_type
            )
        
        # Evaluate model
        evaluation = evaluate_model(model, X_test, y_test, feature_names)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d")
        model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_{model_type}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Generate and save plots
        if 'feature_importance' in evaluation and not evaluation['feature_importance'].empty:
            fig_importance = plot_feature_importance(
                evaluation['feature_importance'],
                crop_name=crop,
                save_path=os.path.join(RESULTS_DIR, "feature_importance", f"{crop.lower()}_importance_{timestamp}.png")
            )
        
        fig_actual_pred = plot_actual_vs_predicted(
            evaluation['predictions'],
            crop_name=crop,
            save_path=os.path.join(RESULTS_DIR, "model_evaluation", f"{crop.lower()}_actual_vs_pred_{timestamp}.png")
        )
        
        fig_error = plot_prediction_error(
            evaluation['predictions'],
            crop_name=crop,
            save_path=os.path.join(RESULTS_DIR, "model_evaluation", f"{crop.lower()}_error_dist_{timestamp}.png")
        )
        
        # SHAP analysis
        shap_values, fig_shap = analyze_feature_interactions(
            model, X_test, feature_names,
            crop_name=crop,
            save_path=os.path.join(RESULTS_DIR, "feature_importance", f"{crop.lower()}_shap_{timestamp}.png")
        )
        
        # Store results
        results[crop] = {
            'model': model,
            'model_path': model_path,
            'evaluation': evaluation,
            'feature_names': feature_names,
            'data_shape': {
                'X_train': X_train.shape,
                'X_test': X_test.shape
            }
        }
        
        # Include hyperparameters if tuned
        if tune_hyperparameters:
            results[crop]['best_params'] = best_params
        
        # Save results as JSON
        results_summary = {
            'crop': crop,
            'model_type': model_type,
            'metrics': evaluation['metrics'],
            'data_shape': {
                'X_train': X_train.shape,
                'X_test': X_test.shape
            }
        }
        
        results_path = os.path.join(RESULTS_DIR, "model_evaluation", f"{crop.lower()}_results_{timestamp}.json")
        pd.Series(results_summary).to_json(results_path)
        logger.info(f"Results saved to {results_path}")
    
    return results

def predict_yield_for_scenario(scenario_data, crop_name, model_path=None):
    """
    Predict yield for a given climate scenario.
    
    Args:
        scenario_data (dict): Climate scenario variables
        crop_name (str): Crop to predict for
        model_path (str, optional): Path to model file
        
    Returns:
        float: Predicted yield
    """
    # Find model file if not provided
    if model_path is None:
        model_files = [f for f in os.listdir(MODELS_DIR) 
                     if f.startswith(crop_name.lower()) and f.endswith('.joblib')]
        
        if not model_files:
            logger.error(f"No model file found for {crop_name}")
            return None
        
        # Use most recent model
        model_files.sort(reverse=True)
        model_path = os.path.join(MODELS_DIR, model_files[0])
    
    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Prepare scenario data
    scenario_df = pd.DataFrame([scenario_data])
    
    # Make prediction
    try:
        predicted_yield = model.predict(scenario_df)[0]
        logger.info(f"Predicted yield for {crop_name}: {predicted_yield:.2f}")
        return predicted_yield
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def simulate_climate_scenarios(base_scenario, crop_name, model_path=None, 
                              temp_range=(-2, 5), precip_range=(-30, 30)):
    """
    Simulate yields under different climate scenarios.
    
    Args:
        base_scenario (dict): Base climate variables
        crop_name (str): Crop to simulate
        model_path (str, optional): Path to model file
        temp_range (tuple): Range of temperature changes (min, max)
        precip_range (tuple): Range of precipitation changes (min, max)
        
    Returns:
        pd.DataFrame: Simulation results
    """
    # Find model file if not provided
    if model_path is None:
        model_files = [f for f in os.listdir(MODELS_DIR) 
                     if f.startswith(crop_name.lower()) and f.endswith('.joblib')]
        
        if not model_files:
            logger.error(f"No model file found for {crop_name}")
            return None
        
        # Use most recent model
        model_files.sort(reverse=True)
        model_path = os.path.join(MODELS_DIR, model_files[0])
    
    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Create temperature and precipitation changes
    temp_steps = np.linspace(temp_range[0], temp_range[1], 8)
    precip_steps = np.linspace(precip_range[0], precip_range[1], 8)
    
    # Create grid of scenarios
    scenarios = []
    
    for temp_change in temp_steps:
        for precip_change in precip_steps:
            # Create modified scenario
            scenario = base_scenario.copy()
            
            # Find temperature and precipitation variables
            temp_keys = [k for k in scenario.keys() if 'temp' in k.lower() or 'tmax' in k.lower() 
                      or 'tmin' in k.lower() or 'tmean' in k.lower()]
            
            precip_keys = [k for k in scenario.keys() if 'prec' in k.lower() or 'prcp' in k.lower() 
                         or 'ppt' in k.lower()]
            
            # Modify temperature variables
            for key in temp_keys:
                scenario[key] = scenario[key] * (1 + temp_change / 100)
            
            # Modify precipitation variables
            for key in precip_keys:
                scenario[key] = scenario[key] * (1 + precip_change / 100)
            
            # Add scenario info
            scenario['temp_change'] = temp_change
            scenario['precip_change'] = precip_change
            
            scenarios.append(scenario)
    
    # Convert to DataFrame
    scenarios_df = pd.DataFrame(scenarios)
    
    # Make predictions
    try:
        # Drop scenario info for prediction
        X = scenarios_df.drop(['temp_change', 'precip_change'], axis=1)
        
        # Predict yields
        predictions = model.predict(X)
        
        # Add predictions to scenarios
        scenarios_df['predicted_yield'] = predictions
        
        logger.info(f"Simulated {len(scenarios_df)} climate scenarios for {crop_name}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = os.path.join(RESULTS_DIR, f"{crop_name.lower()}_climate_scenarios_{timestamp}.csv")
        scenarios_df.to_csv(output_path, index=False)
        logger.info(f"Scenario results saved to {output_path}")
        
        return scenarios_df
    
    except Exception as e:
        logger.error(f"Error simulating scenarios: {e}")
        return None

def plot_scenario_heatmap(scenarios_df, crop_name=None, save_path=None):
    """
    Create a heatmap visualization of scenario results.
    
    Args:
        scenarios_df (pd.DataFrame): Scenario simulation results
        crop_name (str, optional): Crop name for title
        save_path (str, optional): Path to save the plot
        
    Returns:
        plt.Figure: Figure object
    """
    if 'temp_change' not in scenarios_df.columns or 'precip_change' not in scenarios_df.columns:
        logger.error("Scenario data missing temperature or precipitation change columns")
        return None
    
    # Pivot data for heatmap
    pivot_data = scenarios_df.pivot_table(
        index='temp_change',
        columns='precip_change',
        values='predicted_yield'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        ax=ax
    )
    
    # Add labels and title
    ax.set_xlabel('Precipitation Change (%)')
    ax.set_ylabel('Temperature Change (%)')
    title = 'Predicted Yield Under Climate Scenarios'
    if crop_name:
        title = f'{title} for {crop_name}'
    ax.set_title(title)
    
    # Add baseline indicator
    ax.plot([3.5, 3.5], [0, 7], 'r--', linewidth=1)  # Vertical line
    ax.plot([0, 7], [3.5, 3.5], 'r--', linewidth=1)  # Horizontal line
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Scenario heatmap saved to {save_path}")
    
    return fig

def main():
    """Main execution function."""
    logger.info("Starting yield prediction model training and evaluation...")
    
    # Train models for all crops
    results = train_all_crop_models(
        merge_level="national",
        tune_hyperparameters=False,
        model_type="random_forest"
    )
    
    if not results:
        logger.error("No models were successfully trained")
        return
    
    logger.info(f"Successfully trained models for {len(results)} crops")
    
    # Example of climate scenario simulation
    crop_to_simulate = next(iter(results.keys()))
    logger.info(f"Simulating climate scenarios for {crop_to_simulate}")
    
    # Create base scenario
    # We'll use the mean values from the test data
    X_test_mean = results[crop_to_simulate]['evaluation']['predictions']['actual'].mean()
    
    # For a real simulation, we would use actual climate values
    # Here we'll create a simple base scenario with placeholder values
    base_scenario = {feature: 0.0 for feature in results[crop_to_simulate]['feature_names']}
    
    # Simulate scenarios
    scenarios_df = simulate_climate_scenarios(
        base_scenario, 
        crop_to_simulate,
        model_path=results[crop_to_simulate]['model_path']
    )
    
    if scenarios_df is not None:
        # Plot scenario heatmap
        timestamp = datetime.now().strftime("%Y%m%d")
        plot_scenario_heatmap(
            scenarios_df,
            crop_name=crop_to_simulate,
            save_path=os.path.join(RESULTS_DIR, f"{crop_to_simulate.lower()}_scenario_heatmap_{timestamp}.png")
        )
    
    logger.info("Yield prediction modeling complete")

if __name__ == "__main__":
    main()