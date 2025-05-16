# Agricultural Yield Climate Impact Analysis System

A data-driven application to analyze relationships between climate variables and field crop yields, allowing users to create what-if scenarios to predict how climate changes might affect agricultural productivity.

## Project Overview

This application uses USDA Census of Agriculture Data (1997-2017) and PRISM Climate Data to:
- Perform descriptive analytics on historical relationships between climate and crop yields
- Provide predictive analytics using machine learning models
- Offer interactive visualizations of data and predictions
- Allow users to create climate change scenarios and see potential impacts

## Quick Start

The easiest way to get started is to use the setup script:

```bash
# Run the setup script to create virtual environment, install dependencies and run the app
python setup.py
```

This will set up your environment and launch the Streamlit application automatically.

## Manual Installation

If you prefer to set up manually:

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create sample data
python src/data/download_usda_data.py
python src/data/download_prism_data.py
```

## Usage

```bash
# Run the Streamlit application
streamlit run src/app/main.py
```

### Application Features

1. **Historical Yield Analysis**:
   - View historical crop yields from 1997-2017
   - Compare production across multiple crops
   - Visualize trends and patterns

2. **Climate-Yield Correlation**:
   - Explore relationships between climate variables and yields
   - View correlation matrices and scatter plots
   - Analyze how climate factors influence different crops

3. **Yield Prediction**:
   - Train machine learning models on climate and yield data
   - View feature importance to understand key climate drivers
   - Evaluate model performance with various metrics

4. **Climate Change Scenarios**:
   - Create custom climate scenarios by adjusting variables
   - View predicted yields under different climate conditions
   - Explore temperature and precipitation impact grids

## Data Sources

1. **USDA Agricultural Data**:
   - Source: USDA National Agricultural Statistics Service (NASS) QuickStats API
   - Years: 1997-2017 (Census data collected every 5 years)
   - Focus: Field crops (corn, soybeans, wheat, cotton, rice)

2. **PRISM Climate Data**:
   - Source: Oregon State PRISM Climate Group
   - Variables: Temperature (min, max, mean), precipitation, growing degree days
   - Resolution: 4km gridded data

## Project Structure

- `data/`: Raw and processed datasets
  - `raw/`: Original, immutable data
  - `interim/`: Intermediate data that has been transformed
  - `processed/`: Final, canonical data sets for modeling
  - `sample/`: Small sample datasets for testing
- `src/`: Source code for the application
  - `data/`: Scripts for downloading and processing data
  - `features/`: Feature engineering modules
  - `models/`: Machine learning model scripts
  - `visualization/`: Visualization utilities
  - `app/`: Streamlit application code
- `models/`: Saved trained machine learning models
- `results/`: Analysis outputs and visualizations
- `tests/`: Unit tests

## Technologies Used

- Python 3.7+
- Streamlit for the web application
- Pandas, NumPy, GeoPandas for data processing
- Scikit-learn for machine learning models
- Rasterio for GeoTIFF processing
- Plotly, Matplotlib, Seaborn for visualizations
- SHAP for model interpretability

## Advanced Usage

### Training Models and Creating Integrated Data

You have multiple options for training machine learning models and creating integrated data:

**Option 1: Use the simplified training script (Recommended)**

```bash
# Basic usage - creates sample data, integrated datasets, and trains Random Forest models
python train_model.py

# Force creation of new sample data even if files exist
python train_model.py --force-sample

# Train a different model type
python train_model.py --model-type gradient_boosting

# Get help and see all options
python train_model.py --help
```

**Option 2: Train models using the core module**

```bash
python src/models/yield_prediction_models.py
```

**Option 3: Train models through the app**

Use the "Train Models" button in the Yield Prediction section of the application.

**Note:** The training script automatically creates integrated datasets that merge USDA crop data with PRISM climate data. This integration is essential for the app to correctly display analyses.

### Creating Custom Scenarios

1. Navigate to the "Climate Change Scenarios" section in the app
2. Use the sliders to adjust climate variables
3. Click "Predict Yield" to see the impact on crop yields

### Adding New Climate Data

If you have an API key for PRISM or wish to use real data:

1. Edit the appropriate variables in `src/data/download_prism_data.py`
2. Run the script to download actual climate data

## Troubleshooting

**Missing Dependencies**:
If you encounter missing dependency errors, ensure you've installed all requirements:
```bash
pip install -r requirements.txt
```

**For Climate-Yield Correlation Features**:
If you see errors related to statsmodels, install it:
```bash
pip install statsmodels
```

**Data Loading Issues**:
If the application can't find the data, or if any analysis page shows missing column errors:
```bash
# Use the training script to create sample data and integrated datasets
python train_model.py --force-sample
```

For more detailed troubleshooting instructions, see the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) file.

## License

MIT License

## Contributors

- WGU Computer Science Capstone Student