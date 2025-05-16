"""
Climate scenario prediction page for the Streamlit application.
Allows users to explore crop yield predictions under different climate scenarios.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Import the new robust prediction functions
from src.models.scenario_prediction import predict_yield_for_scenario, simulate_climate_scenarios

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_climate_scenario_page():
    """
    Display the climate scenario prediction page in the Streamlit app.
    """
    st.title("Climate Scenario Explorer")
    st.write("""
    Explore how potential climate changes might affect crop yields.
    Adjust temperature and precipitation changes to see predicted outcomes.
    """)
    
    # Model selection
    model_dir = Path("models")
    if not model_dir.exists():
        st.error(f"Model directory not found at {model_dir.absolute()}")
        return
        
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        st.error("No model files found. Please train models first.")
        return
        
    selected_model = st.selectbox(
        "Select a trained model",
        options=model_files,
        format_func=lambda x: x.stem.replace("_", " ").title()
    )
    
    # Base feature inputs
    st.subheader("Base Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Temperature (°C)", value=20.0, step=0.5)
        precipitation = st.number_input("Precipitation (mm)", value=500.0, step=10.0)
        
    with col2:
        soil_quality = st.slider("Soil Quality", min_value=1, max_value=10, value=7)
        growing_season = st.number_input("Growing Season (days)", value=120, step=5)
    
    # Create base features dictionary
    base_features = {
        "temperature": temperature,
        "precipitation": precipitation,
        "soil_quality": soil_quality,
        "growing_season": growing_season
    }
    
    # Add other potential features
    with st.expander("Additional Features (Optional)"):
        elevation = st.number_input("Elevation (m)", value=100.0, step=10.0)
        fertilizer = st.slider("Fertilizer Application", min_value=0, max_value=100, value=50)
        
        base_features.update({
            "elevation": elevation,
            "fertilizer": fertilizer
        })
    
    # Tabs for single scenario vs. grid simulation
    tab1, tab2 = st.tabs(["Single Scenario", "Scenario Grid"])
    
    # Single scenario prediction
    with tab1:
        st.subheader("Climate Change Scenario")
        
        temp_change = st.slider(
            "Temperature Change (°C)", 
            min_value=-3.0, 
            max_value=5.0, 
            value=0.0, 
            step=0.1
        )
        
        precip_change = st.slider(
            "Precipitation Change (%)", 
            min_value=-30.0, 
            max_value=30.0, 
            value=0.0, 
            step=1.0
        )
        
        if st.button("Predict Yield for Scenario"):
            with st.spinner("Making prediction..."):
                prediction, message = predict_yield_for_scenario(
                    model_path=str(selected_model),
                    base_features=base_features,
                    temp_change=temp_change,
                    precip_change=precip_change
                )
                
                if prediction is not None:
                    st.success(f"Predicted Yield: {prediction:.2f}")
                    
                    # Show comparison to baseline
                    baseline_prediction, _ = predict_yield_for_scenario(
                        model_path=str(selected_model),
                        base_features=base_features
                    )
                    
                    if baseline_prediction is not None:
                        change = ((prediction - baseline_prediction) / baseline_prediction) * 100
                        change_text = f"{change:.1f}% {'increase' if change >= 0 else 'decrease'}"
                        st.info(f"Compared to baseline: {change_text}")
                else:
                    st.error(f"Prediction failed: {message}")
    
    # Grid simulation
    with tab2:
        st.subheader("Climate Scenario Grid")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp_min = st.number_input("Min Temperature Change (°C)", value=-2.0, step=0.5)
            temp_max = st.number_input("Max Temperature Change (°C)", value=4.0, step=0.5)
            temp_steps = st.slider("Temperature Steps", min_value=3, max_value=10, value=7)
            
        with col2:
            precip_min = st.number_input("Min Precipitation Change (%)", value=-20.0, step=5.0)
            precip_max = st.number_input("Max Precipitation Change (%)", value=20.0, step=5.0)
            precip_steps = st.slider("Precipitation Steps", min_value=3, max_value=10, value=5)
        
        if st.button("Run Simulation"):
            with st.spinner("Running climate scenarios..."):
                yield_grid, temp_values, precip_values, message = simulate_climate_scenarios(
                    model_path=str(selected_model),
                    base_features=base_features,
                    temp_range=(temp_min, temp_max, temp_steps),
                    precip_range=(precip_min, precip_max, precip_steps)
                )
                
                if yield_grid is not None:
                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create heatmap
                    sns.heatmap(
                        yield_grid,
                        annot=True,
                        fmt=".1f",
                        cmap="YlGnBu",
                        xticklabels=[f"{p:+.1f}%" for p in precip_values],
                        yticklabels=[f"{t:+.1f}°C" for t in temp_values],
                        ax=ax
                    )
                    
                    ax.set_xlabel("Precipitation Change (%)")
                    ax.set_ylabel("Temperature Change (°C)")
                    ax.set_title("Predicted Yield by Climate Scenario")
                    
                    st.pyplot(fig)
                    
                    # Create a DataFrame for download
                    temp_labels = [f"Temp {t:+.1f}°C" for t in temp_values]
                    precip_labels = [f"Precip {p:+.1f}%" for p in precip_values]
                    
                    df_results = pd.DataFrame(yield_grid, index=temp_labels, columns=precip_labels)
                    
                    # Add download button
                    csv = df_results.to_csv()
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "climate_scenario_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.error(f"Simulation failed: {message}")

if __name__ == "__main__":
    # For testing the page directly with streamlit run
    show_climate_scenario_page()