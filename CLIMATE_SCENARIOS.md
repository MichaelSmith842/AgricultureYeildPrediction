# Climate Scenario Analysis

This document provides details on how to use the Climate Scenario Analysis feature of the Agricultural Yield Climate Impact Analysis System.

## Overview

The Climate Scenario Analysis allows you to explore how potential changes in climate variables might affect crop yields. You can:

1. Create individual scenarios by adjusting temperature and precipitation
2. Generate a grid simulation to explore multiple climate scenarios at once
3. Compare predictions to baseline conditions

## Setup

To ensure the Climate Scenario Analysis works properly, follow these steps:

1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create models specifically for climate scenario analysis:
   ```bash
   python train_model_simple.py
   ```
   This will create specialized models for all major crops.

3. Run the application:
   ```bash
   streamlit run src/app/main.py
   ```

4. Navigate to the "Climate Change Scenarios" option in the application.

## Using the Climate Scenario Explorer

### Single Scenario Analysis

1. Select a trained model from the dropdown
2. Set the base conditions (temperature, precipitation, etc.)
3. Adjust the climate change scenario (temperature and precipitation changes)
4. Click "Predict Yield for Scenario"
5. View the predicted yield and comparison to baseline conditions

### Scenario Grid Analysis

1. Select a trained model
2. Set the base conditions
3. Configure the temperature and precipitation ranges to explore
4. Click "Run Simulation"
5. Analyze the heatmap to see how different combinations affect yield
6. Download the results as a CSV file for further analysis

## Model Training

The climate scenario models are trained using synthetic data that establishes realistic relationships between climate variables and crop yields. Each crop has different sensitivities to temperature and precipitation:

- **Corn**: Prefers moderate temperatures and good rainfall
- **Soybeans**: Similar to corn but less sensitive to temperature
- **Wheat**: Better in cooler temperatures, requires less water
- **Cotton**: Prefers warm and dry conditions
- **Rice**: Requires more water than other crops

## Interpreting Results

- **Temperature changes** are measured in absolute degrees Celsius (°C)
- **Precipitation changes** are measured as percentage changes from baseline
- **Yield predictions** are in bushels per acre (or pounds per acre for cotton)
- **Color intensity** in the heatmap indicates higher or lower yields

## Troubleshooting

If the Climate Scenario Analysis is not working:

1. Check that the model files exist in the `models/` directory
2. Run `python train_model_simple.py` again to recreate the models
3. Ensure all dependencies are installed, including statsmodels
4. Check the console log for specific error messages