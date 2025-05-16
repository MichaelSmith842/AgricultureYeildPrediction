#!/usr/bin/env python3
"""
Main Streamlit application for the Agricultural Yield Climate Impact Analysis System.
"""

import os
import pandas as pd
import streamlit as st
import sys
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import project modules - with error handling for missing dependencies
try:
    from src.app.climate_scenario_page import show_climate_scenario_page
    CLIMATE_SCENARIO_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing climate scenario module: {e}")
    CLIMATE_SCENARIO_AVAILABLE = False

try:
    from src.data.integrate_data import load_and_integrate_all_data
    DATA_INTEGRATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing data integration module: {e}")
    DATA_INTEGRATION_AVAILABLE = False

try:
    from src.visualization.visualization_utils import (
        create_time_series_plot, 
        create_bar_chart, 
        create_correlation_matrix,
        create_scatter_plot
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing visualization utilities: {e}")
    VISUALIZATION_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Agricultural Yield Climate Impact Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to data
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
INTEGRATED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "integrated")

@st.cache_data
def load_data():
    """
    Load the data for the application.
    
    Returns:
        tuple: USDA data, PRISM data, integrated data
    """
    # Check if data integration is available
    if not DATA_INTEGRATION_AVAILABLE:
        # Load sample data only
        usda_sample_path = os.path.join(SAMPLE_DATA_DIR, "usda_crop_data_sample.csv")
        prism_sample_path = os.path.join(PROCESSED_DATA_DIR, "prism", "prism_growing_season_sample.csv")
        
        try:
            usda_data = pd.read_csv(usda_sample_path)
            
            if os.path.exists(prism_sample_path):
                prism_data = pd.read_csv(prism_sample_path)
            else:
                prism_data = None
            
            return usda_data, prism_data, None, {}
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return None, None, None, {}
    
    # Check for integrated data first
    if os.path.exists(INTEGRATED_DATA_DIR):
        try:
            integrated_files = [f for f in os.listdir(INTEGRATED_DATA_DIR) if f.startswith("integrated_dataset") and f.endswith(".csv")]
            
            if integrated_files:
                # Use most recent file
                integrated_files.sort(reverse=True)
                integrated_path = os.path.join(INTEGRATED_DATA_DIR, integrated_files[0])
                integrated_df = pd.read_csv(integrated_path)
                
                # Get crop-specific datasets
                crop_datasets = {}
                crop_files = [f for f in os.listdir(INTEGRATED_DATA_DIR) if "_dataset_" in f and f.endswith(".csv")]
                
                for file in crop_files:
                    crop_name = file.split("_")[0].upper()
                    crop_path = os.path.join(INTEGRATED_DATA_DIR, file)
                    crop_datasets[crop_name] = pd.read_csv(crop_path)
                
                return None, None, integrated_df, crop_datasets
        except Exception as e:
            st.warning(f"Error reading integrated data: {e}")
    
    # If no integrated data, try to load and integrate
    try:
        integrated_df, crop_datasets = load_and_integrate_all_data(save_output=True)
        return None, None, integrated_df, crop_datasets
    except Exception as e:
        st.warning(f"Error loading integrated data: {e}")
    
    # If that fails, load sample data
    usda_sample_path = os.path.join(SAMPLE_DATA_DIR, "usda_crop_data_sample.csv")
    prism_sample_path = os.path.join(PROCESSED_DATA_DIR, "prism", "prism_growing_season_sample.csv")
    
    try:
        usda_data = pd.read_csv(usda_sample_path)
        
        if os.path.exists(prism_sample_path):
            prism_data = pd.read_csv(prism_sample_path)
        else:
            prism_data = None
        
        return usda_data, prism_data, None, {}
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, None, None, {}

def create_sidebar():
    """Create the sidebar with controls."""
    st.sidebar.title("Analysis Controls")
    
    # Create sections in the sidebar
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Historical Yield Analysis", "Climate-Yield Correlation", "Yield Prediction", "Climate Change Scenarios"]
    )
    
    # Crop selection
    crops = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "RICE"]
    crop_selection = st.sidebar.multiselect(
        "Select Crops",
        crops,
        default=["CORN"]
    )
    
    # Year range selection
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=1997,
        max_value=2017,
        value=(1997, 2017),
        step=5
    )
    
    # Climate variables selection
    climate_vars = st.sidebar.multiselect(
        "Climate Variables",
        ["growing_season_tmax_mean", "growing_season_tmin_mean", "growing_season_ppt_total", 
         "growing_degree_days_base10", "frost_free_days"],
        default=["growing_season_tmax_mean", "growing_season_ppt_total"]
    )
    
    # Return the selections
    return {
        "analysis_type": analysis_type,
        "crops": crop_selection,
        "year_range": year_range,
        "climate_vars": climate_vars
    }

def display_historical_yield(data, selections):
    """Display historical yield analysis."""
    st.header("Historical Yield Analysis")
    
    # Check if the dataframe has required columns
    required_columns = False
    
    # Determine which dataset to use and check if columns exist
    if "crop" in data.columns:
        crop_col = "crop"
        required_columns = True
    elif "CROP" in data.columns:
        crop_col = "CROP"
        required_columns = True
    else:
        st.error("Dataset does not contain a crop column. Please check your data.")
        st.dataframe(data.head())
        st.write("Available columns:", ", ".join(data.columns))
        return
    
    if "year" in data.columns:
        year_col = "year"
    elif "YEAR" in data.columns:
        year_col = "YEAR"
    else:
        st.error("Dataset does not contain a year column. Please check your data.")
        st.dataframe(data.head())
        return
    
    if "yield" in data.columns:
        yield_col = "yield"
    elif "YIELD" in data.columns:
        yield_col = "YIELD"
    else:
        st.error("Dataset does not contain a yield column. Please check your data.")
        st.dataframe(data.head())
        return
    
    # Check if the determined columns actually exist
    try:
        # Verify columns exist by accessing them
        data[crop_col]
        data[year_col]
        data[yield_col]
    except KeyError as e:
        st.error(f"Error accessing column: {e}")
        st.write("Available columns:", ", ".join(data.columns))
        st.dataframe(data.head())
        return
    
    try:
        # Filter data based on selections
        filtered_data = data[
            (data[crop_col].isin(selections["crops"])) &
            (data[year_col] >= selections["year_range"][0]) &
            (data[year_col] <= selections["year_range"][1])
        ]
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        st.write("This might be due to incorrect data types or missing columns.")
        st.dataframe(data.head())
        return
    
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create yield time series plot
    fig = create_time_series_plot(
        filtered_data,
        x_col=year_col,
        y_col=yield_col,
        color_col=crop_col,
        title="Historical Crop Yields (1997-2017)",
        x_label="Year",
        y_label="Yield (bushels/acre)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Check if production data is available
    if "PRODUCTION" in filtered_data.columns or "production" in filtered_data.columns:
        # Create production comparison
        prod_col = "production" if "production" in filtered_data.columns else "PRODUCTION"
        
        st.subheader("Production Comparison")
        
        fig_production = create_bar_chart(
            filtered_data,
            x_col=year_col,
            y_col=prod_col,
            color_col=crop_col,
            title="Crop Production Over Time",
            x_label="Year",
            y_label="Production (bushels)"
        )
        
        st.plotly_chart(fig_production, use_container_width=True)
    
    # Display data table
    st.subheader("Crop Data")
    st.dataframe(filtered_data)

def display_climate_correlation(data, selections):
    """Display climate-yield correlation analysis."""
    st.header("Climate-Yield Correlation Analysis")
    
    # Check if we have climate data columns
    climate_columns = [col for col in data.columns 
                     if any(var in col.lower() for var in ['temp', 'prcp', 'precip', 'gdd', 'frost'])]
    
    if not climate_columns:
        st.error("No climate data columns found in the dataset.")
        st.info("""
        To perform climate correlation analysis, the dataset needs to include climate variables.
        This typically requires running the data integration pipeline with PRISM climate data.
        
        Available columns:
        """)
        st.write(", ".join(data.columns))
        st.dataframe(data.head())
        return
    
    # Determine which dataset to use
    if "crop" in data.columns:
        crop_col = "crop"
    elif "CROP" in data.columns:
        crop_col = "CROP"
    else:
        st.error("Dataset does not contain a crop column. Please check your data.")
        st.dataframe(data.head())
        return
    
    if "yield" in data.columns:
        yield_col = "yield"
    elif "YIELD" in data.columns:
        yield_col = "YIELD"
    else:
        st.error("Dataset does not contain a yield column. Please check your data.")
        st.dataframe(data.head())
        return
    
    try:
        # Filter data based on selections
        filtered_data = data[data[crop_col].isin(selections["crops"])]
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        st.dataframe(data.head())
        return
    
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create tabs for different correlation visualizations
    tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Scatter Plots", "Time Series"])
    
    with tab1:
        # Only include numeric columns in correlation
        numeric_cols = filtered_data.select_dtypes(include=["number"]).columns.tolist()
        
        # Keep only yield and selected climate variables
        corr_cols = [yield_col] + [col for col in numeric_cols if any(var in col for var in selections["climate_vars"])]
        
        # Create correlation matrix
        fig = create_correlation_matrix(
            filtered_data,
            columns=corr_cols,
            title="Correlation Between Yield and Climate Variables"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create scatter plots for each climate variable
        st.subheader("Climate Variable vs. Yield")
        
        # Allow user to select specific climate variable
        climate_var = st.selectbox(
            "Select Climate Variable",
            selections["climate_vars"]
        )
        
        # Find matching columns
        matching_cols = [col for col in filtered_data.columns if climate_var in col]
        
        if matching_cols:
            selected_var = matching_cols[0]
            
            fig = create_scatter_plot(
                filtered_data,
                x_col=selected_var,
                y_col=yield_col,
                color_col=crop_col,
                title=f"{selected_var} vs. {yield_col}",
                x_label=selected_var,
                y_label=yield_col,
                add_trendline=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data found for {climate_var}")
    
    with tab3:
        # Create time series of yield and climate variables
        st.subheader("Yield and Climate Variables Over Time")
        
        # Allow user to select specific climate variable
        climate_var = st.selectbox(
            "Select Climate Variable for Time Series",
            selections["climate_vars"],
            key="time_series_var"
        )
        
        # Find matching columns
        matching_cols = [col for col in filtered_data.columns if climate_var in col]
        
        if matching_cols:
            selected_var = matching_cols[0]
            
            for crop in selections["crops"]:
                crop_data = filtered_data[filtered_data[crop_col] == crop]
                
                if not crop_data.empty:
                    # Create two y-axes plot
                    st.subheader(f"{crop}: {yield_col} and {selected_var}")
                    
                    # Create figure with secondary y-axis
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add yield trace
                    fig.add_trace(
                        go.Scatter(
                            x=crop_data["year"] if "year" in crop_data.columns else crop_data["YEAR"],
                            y=crop_data[yield_col],
                            name=yield_col,
                            line=dict(color="blue")
                        ),
                        secondary_y=False
                    )
                    
                    # Add climate variable trace
                    fig.add_trace(
                        go.Scatter(
                            x=crop_data["year"] if "year" in crop_data.columns else crop_data["YEAR"],
                            y=crop_data[selected_var],
                            name=selected_var,
                            line=dict(color="red")
                        ),
                        secondary_y=True
                    )
                    
                    # Add figure layout
                    fig.update_layout(
                        title=f"{crop}: {yield_col} and {selected_var} Over Time",
                        xaxis_title="Year"
                    )
                    
                    # Set y-axes titles
                    fig.update_yaxes(title_text=yield_col, secondary_y=False)
                    fig.update_yaxes(title_text=selected_var, secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data found for {climate_var}")

def display_yield_prediction(data, crop_datasets, selections):
    """Display yield prediction models."""
    st.header("Yield Prediction Models")
    
    # Check if we have trained models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        st.info("""
        This section requires trained machine learning models.
        
        To train models:
        1. Run the yield_prediction_models.py script
        2. Models will be saved in the models/ directory
        3. Refresh this page to see model results
        """)
        
        # Provide button to run training script
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    from src.models.yield_prediction_models import train_all_crop_models
                    results = train_all_crop_models(merge_level="national", model_type="random_forest")
                    if results:
                        st.success(f"Successfully trained models for {len(results)} crops.")
                    else:
                        st.error("Model training failed.")
                except Exception as e:
                    st.error(f"Error training models: {e}")
        return
    
    # If we have models, show model evaluation results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    
    if os.path.exists(results_dir):
        # Check for feature importance plots
        importance_dir = os.path.join(results_dir, "feature_importance")
        if os.path.exists(importance_dir):
            importance_files = [f for f in os.listdir(importance_dir) if f.endswith(".png")]
            
            if importance_files:
                st.subheader("Feature Importance")
                
                # Show importance plots for selected crops
                for crop in selections["crops"]:
                    crop_files = [f for f in importance_files if f.startswith(crop.lower())]
                    
                    if crop_files:
                        # Sort by date (newest first)
                        crop_files.sort(reverse=True)
                        importance_path = os.path.join(importance_dir, crop_files[0])
                        
                        st.write(f"**{crop} Feature Importance**")
                        st.image(importance_path)
        
        # Check for evaluation plots
        eval_dir = os.path.join(results_dir, "model_evaluation")
        if os.path.exists(eval_dir):
            eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".png")]
            
            if eval_files:
                st.subheader("Model Evaluation")
                
                # Show actual vs predicted plots for selected crops
                col1, col2 = st.columns(2)
                
                for i, crop in enumerate(selections["crops"]):
                    actual_vs_pred_files = [f for f in eval_files if f.startswith(crop.lower()) and "actual_vs_pred" in f]
                    error_dist_files = [f for f in eval_files if f.startswith(crop.lower()) and "error_dist" in f]
                    
                    if actual_vs_pred_files and error_dist_files:
                        # Sort by date (newest first)
                        actual_vs_pred_files.sort(reverse=True)
                        error_dist_files.sort(reverse=True)
                        
                        actual_vs_pred_path = os.path.join(eval_dir, actual_vs_pred_files[0])
                        error_dist_path = os.path.join(eval_dir, error_dist_files[0])
                        
                        # Display in columns
                        with col1 if i % 2 == 0 else col2:
                            st.write(f"**{crop} Actual vs Predicted**")
                            st.image(actual_vs_pred_path)
                        
                        with col2 if i % 2 == 0 else col1:
                            st.write(f"**{crop} Error Distribution**")
                            st.image(error_dist_path)
    
    # Display information about the models
    st.subheader("About the Prediction Models")
    
    st.write("""
    The yield prediction models use machine learning to forecast crop yields based on climate variables.
    
    **Features used in the models:**
    - Growing season temperature (min, max, mean)
    - Growing season precipitation
    - Growing degree days
    - Frost-free days
    - Heat and drought stress indicators
    
    **Model types:**
    - Random Forest: Ensemble of decision trees that captures complex, non-linear relationships
    - Gradient Boosting: Sequential ensemble that improves prediction accuracy
    - Elastic Net: Linear model with regularization for feature selection
    
    **Evaluation metrics:**
    - R²: Proportion of variance explained by the model
    - RMSE: Root Mean Square Error (accuracy in yield units)
    - MAPE: Mean Absolute Percentage Error (percentage accuracy)
    """)

def main():
    """Main application function."""
    # Display header
    st.title("Agricultural Yield Climate Impact Analysis")
    st.write("Analyze relationships between climate variables and field crop yields")
    
    # Check if required modules are available
    if not VISUALIZATION_AVAILABLE:
        st.error("Visualization utilities are required for this application.")
        st.info("""
        To enable visualizations, install the required dependencies:
        ```
        pip install -r requirements.txt
        ```
        """)
        return
    
    # Load data
    try:
        if DATA_INTEGRATION_AVAILABLE:
            usda_data, prism_data, integrated_data, crop_datasets = load_data()
        else:
            # Fallback to basic data loading if integration is not available
            usda_sample_path = os.path.join(SAMPLE_DATA_DIR, "usda_crop_data_sample.csv")
            try:
                usda_data = pd.read_csv(usda_sample_path)
                st.info("Using sample USDA data. Data integration module not available.")
                integrated_data = None
                prism_data = None
                crop_datasets = {}
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
                return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Determine which dataset to use
    if integrated_data is not None:
        data = integrated_data
    elif usda_data is not None:
        data = usda_data
    else:
        st.error("No data available. Please check data sources.")
        return
    
    # Create sidebar controls
    selections = create_sidebar()
    
    # Display the selected analysis type
    if selections["analysis_type"] == "Historical Yield Analysis":
        display_historical_yield(data, selections)
    elif selections["analysis_type"] == "Climate-Yield Correlation":
        display_climate_correlation(data, selections)
    elif selections["analysis_type"] == "Yield Prediction":
        display_yield_prediction(data, crop_datasets, selections)
    elif selections["analysis_type"] == "Climate Change Scenarios":
        # Check if climate scenario module is available
        if CLIMATE_SCENARIO_AVAILABLE:
            show_climate_scenario_page()
        else:
            st.error("Climate scenario analysis is not available.")
            st.info("""
            To enable climate scenario analysis, install the required dependencies:
            ```
            pip install -r requirements.txt
            ```
            
            Then run the application again.
            """)

if __name__ == "__main__":
    main()