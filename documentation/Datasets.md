# Datasets Documentation
# Agricultural Yield Climate Impact Analysis System

## Dataset Overview

The Agricultural Yield Climate Impact Analysis System utilizes two primary data sources:

1. **USDA Census of Agriculture Data (1997-2017)**: Contains crop yield, harvested acres, and production data for major field crops across the United States.

2. **PRISM Climate Data**: High-resolution spatial climate datasets containing temperature, precipitation, and other climate variables.

This document details the raw datasets, processing methods, and resulting cleaned datasets used throughout the system.

## Raw Datasets

### USDA Census of Agriculture Data

**Source**: USDA National Agricultural Statistics Service (NASS) QuickStats Database  
**URL**: https://quickstats.nass.usda.gov/  
**Years Covered**: 1997, 2002, 2007, 2012, 2017 (census years)  
**Access Method**: API with key (030B1C21-01BF-3BCC-855F-BEE447E9E44D)  

**Raw Format**:
```
YEAR,STATE_ALPHA,STATE_FIPS,COUNTY_CODE,COUNTY_NAME,COMMODITY,DATA_ITEM,DOMAIN,DOMAIN_CATEGORY,VALUE,CV_PERCENT
2017,US,,,,CORN,CORN - YIELD, MEASURED IN BU / ACRE,TOTAL,176.6,
2017,AL,01,000,,CORN,CORN - YIELD, MEASURED IN BU / ACRE,TOTAL,157,
...
```

**Variables**:
- `YEAR`: Census year (1997-2017)
- `STATE_ALPHA`: State abbreviation
- `STATE_FIPS`: State FIPS code
- `COUNTY_CODE`: County code
- `COUNTY_NAME`: County name
- `COMMODITY`: Crop type (CORN, SOYBEANS, WHEAT, COTTON, RICE)
- `DATA_ITEM`: Measurement type (yield, production, harvested acres)
- `VALUE`: Measurement value
- `CV_PERCENT`: Coefficient of variation

### PRISM Climate Data

**Source**: Oregon State PRISM Climate Group  
**URL**: https://prism.oregonstate.edu/  
**Years Covered**: 1981-present  
**Access Method**: Web API (no key required)  

**Raw Format**: GeoTIFF files containing gridded climate data at 4km resolution

**Variables**:
- `ppt`: Total precipitation (mm)
- `tmin`: Minimum temperature (°C)
- `tmax`: Maximum temperature (°C)
- `tmean`: Mean temperature (°C)

**Temporal Resolution**:
- Monthly averages
- Daily data (not used in current implementation)

## Data Processing Pipeline

The raw datasets undergo a series of processing steps to prepare them for analysis:

### USDA Data Processing

1. **Download and Extraction** (`src/data/download_usda_data.py`):
   - Access USDA NASS API to download census data
   - Filter for field crops (corn, soybeans, wheat, cotton, rice)
   - Extract yield, harvested acres, and production metrics

2. **Cleaning and Standardization** (`src/data/integrate_data.py`):
   - Standardize column names
   - Convert units to consistent formats
   - Filter for complete records
   - Create national and state-level aggregations

### PRISM Climate Data Processing

1. **Download and Extraction** (`src/data/download_prism_data.py`):
   - Access PRISM web service API
   - Download monthly climate data for census years
   - Process and extract from GeoTIFF format

2. **Spatial Aggregation** (`src/features/spatial_aggregation.py`):
   - Aggregate 4km gridded data to state level
   - Calculate area-weighted averages
   - Align with administrative boundaries

3. **Temporal Alignment** (`src/features/temporal_alignment.py`):
   - Align climate data with growing seasons for each crop
   - Calculate seasonal averages and totals
   - Match with census years

### Feature Engineering

1. **Climate Feature Derivation** (`src/features/prism_feature_engineering.py`):
   - Calculate growing degree days (base 10°C)
   - Derive heat stress indices (days above threshold)
   - Compute drought indicators
   - Determine frost-free period length

2. **Data Integration** (`src/data/integrate_data.py`):
   - Merge USDA yield data with PRISM climate data
   - Align spatial and temporal dimensions
   - Create crop-specific integrated datasets
   - Generate final analysis-ready datasets

## Cleaned Datasets

The data processing pipeline produces several cleaned datasets:

### 1. USDA Crop Data Sample

**Location**: `data/sample/usda_crop_data_sample.csv`  
**Format**: CSV file with standardized crop data  

**Fields**:
```
YEAR,CROP,YIELD,HARVESTED_ACRES,PRODUCTION
1997,CORN,115.1,78529716,9036293304
2002,CORN,113.4,79027353,8965254934
...
```

### 2. PRISM Climate Data (State-Level)

**Location**: `data/processed/prism/prism_state_monthly_sample.csv`  
**Format**: CSV file with monthly climate data aggregated by state  

**Fields**:
```
year,month,state_code,tmin_mean,tmax_mean,tmean,ppt_total
1997,5,AL,15.2,28.1,21.7,121.3
1997,6,AL,19.5,31.2,25.4,98.6
...
```

### 3. PRISM Growing Season Data

**Location**: `data/processed/prism/prism_growing_season_sample.csv`  
**Format**: CSV file with growing season climate metrics  

**Fields**:
```
year,growing_season_tmax_mean,growing_season_tmin_mean,growing_season_tmean,growing_season_ppt_total,growing_degree_days_base10,frost_free_days,heat_stress_days,drought_index
1997,27.2,14.4,22.3,439.2,1851.6,153.3,8.9,4.1
...
```

### 4. Integrated Crop-Climate Dataset

**Location**: `data/processed/integrated/integrated_dataset_YYYYMMDD.csv`  
**Format**: CSV file combining crop and climate data  

**Fields**:
```
YEAR,CROP,YIELD,HARVESTED_ACRES,PRODUCTION,growing_season_tmax_mean,growing_season_tmin_mean,growing_season_tmean,growing_season_ppt_total,growing_degree_days_base10,frost_free_days,heat_stress_days,drought_index
1997,CORN,115.1,78529716,9036293304,27.2,14.4,22.3,439.2,1851.6,153.3,8.9,4.1
...
```

### 5. Crop-Specific Datasets

**Location**: `data/processed/integrated/{crop}_dataset_YYYYMMDD.csv`  
**Format**: CSV files for each crop type with complete data  

**Structure**: Same as integrated dataset but filtered for a specific crop

## Sample Data Generation

For development and testing, the system can generate synthetic sample data:

1. **USDA Sample Data Generation** (`train_model.py:create_sample_usda_data`):
   - Creates realistic yield, acreage, and production values
   - Maintains correct relationships between variables
   - Covers all census years and major crops

2. **PRISM Sample Data Generation** (`train_model.py:create_sample_prism_data`):
   - Generates realistic climate variables with proper relationships
   - Includes systematic trends to simulate climate change
   - Ensures variable ranges match real-world data

## Data Access Code

The data acquisition and processing code is located in the following files:

1. **USDA Data Download**: `src/data/download_usda_data.py`
2. **PRISM Data Download**: `src/data/download_prism_data.py`
3. **Data Integration**: `src/data/integrate_data.py`
4. **Sample Data Creation**: `train_model.py`

## Dataset Usage in the System

The cleaned, integrated datasets are used throughout the system for:

1. **Historical Analysis**: Time series visualization and comparison of crop yields
2. **Correlation Analysis**: Examining relationships between climate variables and yields
3. **Model Training**: Creating machine learning models to predict yields
4. **Scenario Analysis**: Simulating potential climate change impacts

## Data Limitations and Considerations

1. **Temporal Resolution**: The USDA data is limited to 5-year intervals (census years)
2. **Spatial Aggregation**: Climate data is aggregated to state level, losing some local variation
3. **Sample Data**: When using generated sample data, results are illustrative rather than actual predictions
4. **Climate Variables**: Limited to temperature and precipitation metrics; other factors like CO2 levels not included

## Conclusion

The data processing pipeline transforms raw USDA agricultural data and PRISM climate data into clean, integrated datasets ready for analysis and modeling. The system provides both actual data processing capabilities and sample data generation for development and demonstration purposes.