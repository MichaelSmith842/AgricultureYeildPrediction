# Troubleshooting Guide for Agricultural Yield Climate Impact Analysis System

This document provides solutions for common issues encountered when setting up and running the application.

## Missing Dependencies

If you encounter errors about missing packages, install them using:

```bash
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt
```

### Specific Package Dependencies

Some analyses require specific packages:

1. **Climate-Yield Correlation** requires statsmodels:
   ```bash
   pip install statsmodels
   ```

2. **Feature Importance Analysis** requires SHAP:
   ```bash
   pip install shap
   ```

## Data Integration Issues

If you see errors related to missing data or columns:

1. **First attempt**: Run the simplified training script to create all necessary data files:
   ```bash
   python train_model.py --force-sample
   ```

2. **If specific analyses still fail**:
   - For "Historical Yield Analysis": Ensure USDA crop data is available
   - For "Climate-Yield Correlation": Ensure both USDA and PRISM data are properly integrated
   - For "Yield Prediction": Ensure models are trained and saved

## Column Name Issues

If you see errors about missing columns:
- Check the integrated dataset at `data/processed/integrated/integrated_dataset_YYYYMMDD.csv`
- Ensure it has both crop columns (CROP, YIELD) and climate columns (growing_season_*)
- If not, run `python train_model.py --force-sample` to regenerate all data

## Running the Application

Always follow this sequence for best results:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install statsmodels  # Required for correlation plots
   ```

2. Create data and train models:
   ```bash
   python train_model.py 
   ```

3. For Climate Scenario Analysis specifically:
   ```bash
   python train_model_simple.py  # Creates specialized models for the climate scenario page
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run src/app/main.py
   ```

## Need More Help?

If you still encounter issues, try these steps:

1. Check the logs in the terminal for specific error messages
2. Verify the integrated datasets by examining their contents:
   ```bash
   head data/processed/integrated/integrated_dataset_*.csv
   ```
3. Ensure your Python environment includes all required packages
4. If you're having issues with a specific analysis, try others first to isolate the problem
5. Clear the Streamlit cache by adding ?clear_cache=true to the URL