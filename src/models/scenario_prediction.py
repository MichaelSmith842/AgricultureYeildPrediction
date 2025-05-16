"""
Climate scenario prediction functions for the agricultural yield prediction app.
These functions provide simplified, robust implementations for climate scenario
modeling in the Streamlit application.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_yield_for_scenario(
    model_path: str,
    base_features: Dict[str, Any],
    temp_change: float = 0.0,
    precip_change: float = 0.0
) -> Tuple[Optional[float], str]:
    """
    Predict crop yield for a specific climate change scenario.
    
    Args:
        model_path: Path to the trained joblib model file
        base_features: Dictionary of base feature values
        temp_change: Temperature change in degrees Celsius
        precip_change: Precipitation change as a percentage (e.g., 10 for +10%)
        
    Returns:
        Tuple of (predicted_yield, message)
        - predicted_yield: Predicted yield value or None if prediction failed
        - message: Status message or error information
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Create a copy of base features for this scenario
        scenario_features = base_features.copy()
        
        # Apply climate change modifications
        if 'temperature' in scenario_features:
            scenario_features['temperature'] += temp_change
        else:
            return None, "Error: Temperature feature not found in base features"
            
        if 'precipitation' in scenario_features:
            # Apply percent change to precipitation
            precip_factor = 1 + (precip_change / 100)
            scenario_features['precipitation'] *= precip_factor
        else:
            return None, "Error: Precipitation feature not found in base features"
        
        # Check if all features required by the model are present
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        missing_features = [feat for feat in model_features if feat not in scenario_features]
        
        if missing_features:
            return None, f"Error: Missing required features: {', '.join(missing_features)}"
        
        # Filter features to only include those expected by the model
        if model_features:
            input_features = {k: scenario_features[k] for k in model_features if k in scenario_features}
        else:
            # Fallback if model doesn't expose feature names
            input_features = scenario_features
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return prediction, "Success"
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None, f"Error: Model file not found at {model_path}"
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, f"Error: Failed to make prediction: {str(e)}"

def simulate_climate_scenarios(
    model_path: str,
    base_features: Dict[str, Any],
    temp_range: Tuple[float, float, int] = (-2.0, 5.0, 8),
    precip_range: Tuple[float, float, int] = (-20.0, 20.0, 5)
) -> Tuple[Optional[np.ndarray], List[float], List[float], str]:
    """
    Simulate crop yields across a grid of climate change scenarios.
    
    Args:
        model_path: Path to the trained joblib model file
        base_features: Dictionary of base feature values
        temp_range: Tuple of (min_temp_change, max_temp_change, num_steps)
        precip_range: Tuple of (min_precip_change, max_precip_change, num_steps)
        
    Returns:
        Tuple of (yield_grid, temp_values, precip_values, message)
        - yield_grid: 2D numpy array of yield predictions or None if simulation failed
        - temp_values: List of temperature change values
        - precip_values: List of precipitation change values
        - message: Status message or error information
    """
    try:
        # Generate temperature and precipitation ranges
        temp_values = np.linspace(temp_range[0], temp_range[1], temp_range[2])
        precip_values = np.linspace(precip_range[0], precip_range[1], precip_range[2])
        
        # Create empty grid for results
        yield_grid = np.zeros((len(temp_values), len(precip_values)))
        
        # Load model once outside the loop
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return None, [], [], f"Error: Model file not found at {model_path}"
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None, [], [], f"Error: Failed to load model: {str(e)}"
            
        # Check if required features exist in base_features
        if 'temperature' not in base_features or 'precipitation' not in base_features:
            missing = []
            if 'temperature' not in base_features:
                missing.append('temperature')
            if 'precipitation' not in base_features:
                missing.append('precipitation')
            return None, [], [], f"Error: Missing required base features: {', '.join(missing)}"
        
        # Check model features
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        missing_features = [feat for feat in model_features if feat not in base_features]
        
        if missing_features:
            return None, [], [], f"Error: Missing required features for model: {', '.join(missing_features)}"
        
        # Run simulations for each combination
        for i, temp_change in enumerate(temp_values):
            for j, precip_change in enumerate(precip_values):
                # Create scenario features
                scenario_features = base_features.copy()
                scenario_features['temperature'] += temp_change
                
                # Apply percent change to precipitation
                precip_factor = 1 + (precip_change / 100)
                scenario_features['precipitation'] *= precip_factor
                
                # Filter features to only include those expected by the model
                if model_features:
                    input_features = {k: scenario_features[k] for k in model_features if k in scenario_features}
                else:
                    # Fallback if model doesn't expose feature names
                    input_features = scenario_features
                
                # Convert to DataFrame for prediction
                input_df = pd.DataFrame([input_features])
                
                # Make prediction
                try:
                    prediction = model.predict(input_df)[0]
                    yield_grid[i, j] = prediction
                except Exception as e:
                    logger.warning(f"Prediction failed for T:{temp_change}, P:{precip_change}: {str(e)}")
                    yield_grid[i, j] = np.nan
        
        return yield_grid, temp_values.tolist(), precip_values.tolist(), "Success"
        
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        return None, [], [], f"Error: Failed to run simulation: {str(e)}"