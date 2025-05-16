#!/usr/bin/env python3
"""
Module for integrating USDA agricultural data with PRISM climate data.
Creates merged datasets ready for analysis and modeling.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

# USDA Census years
CENSUS_YEARS = [1997, 2002, 2007, 2012, 2017]

# Crop list
CROPS = ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'RICE']

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, "integrated"), exist_ok=True)

def load_usda_data(filepath=None):
    """
    Load USDA crop yield data.
    
    Args:
        filepath (str, optional): Path to USDA data file
        
    Returns:
        pd.DataFrame: USDA crop data
    """
    if filepath is None:
        # Look for sample data first
        sample_path = os.path.join(RAW_DATA_DIR, "sample", "usda_crop_data_sample.csv")
        processed_path = os.path.join(PROCESSED_DATA_DIR, "usda_crop_data_processed.csv")
        
        if os.path.exists(sample_path):
            filepath = sample_path
        elif os.path.exists(processed_path):
            filepath = processed_path
        else:
            # Look for any CSV file in the processed directory that might contain USDA data
            processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv') and 'usda' in f.lower()]
            if processed_files:
                filepath = os.path.join(PROCESSED_DATA_DIR, processed_files[0])
            else:
                logger.error("Could not find USDA crop data file")
                return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded USDA data from {filepath} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading USDA data: {e}")
        return pd.DataFrame()

def load_prism_climate_data(type="growing_season", crop=None, year=None):
    """
    Load PRISM climate data.
    
    Args:
        type (str): Type of climate data to load ("growing_season", "monthly", "state_level")
        crop (str, optional): Crop to filter for
        year (int, optional): Year to filter for
        
    Returns:
        pd.DataFrame: PRISM climate data
    """
    # Determine file pattern to look for
    if type == "growing_season":
        file_pattern = "prism_growing_season"
    elif type == "monthly":
        file_pattern = "prism_monthly" 
    elif type == "state_level":
        file_pattern = "prism_state"
    else:
        file_pattern = "prism"
    
    # Look for sample data first
    sample_path = os.path.join(PROCESSED_DATA_DIR, "prism", f"{file_pattern}_sample.csv")
    
    if os.path.exists(sample_path):
        filepath = sample_path
    else:
        # Look for any CSV file in the processed directory that might contain PRISM data
        processed_files = [f for f in os.listdir(os.path.join(PROCESSED_DATA_DIR, "prism")) 
                         if f.endswith('.csv') and file_pattern in f.lower()]
        
        if processed_files:
            # Sort by creation time to get the most recent file
            processed_files.sort(key=lambda f: os.path.getmtime(os.path.join(PROCESSED_DATA_DIR, "prism", f)), reverse=True)
            filepath = os.path.join(PROCESSED_DATA_DIR, "prism", processed_files[0])
        else:
            logger.error(f"Could not find PRISM {type} data file")
            return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded PRISM {type} data from {filepath} with {len(df)} rows")
        
        # Filter for specific crop if requested
        if crop and 'crop' in df.columns:
            df = df[df['crop'] == crop]
            logger.info(f"Filtered to {len(df)} rows for crop {crop}")
        
        # Filter for specific year if requested
        if year and 'year' in df.columns:
            df = df[df['year'] == year]
            logger.info(f"Filtered to {len(df)} rows for year {year}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading PRISM data: {e}")
        return pd.DataFrame()

def merge_usda_with_prism_data(usda_df, prism_df, merge_level="national"):
    """
    Merge USDA crop data with PRISM climate data.
    
    Args:
        usda_df (pd.DataFrame): USDA crop data
        prism_df (pd.DataFrame): PRISM climate data
        merge_level (str): Level at which to merge data ("national", "state", "county")
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    if usda_df.empty or prism_df.empty:
        logger.error("Cannot merge empty dataframes")
        return pd.DataFrame()
    
    # Ensure year column exists in both dataframes
    if 'YEAR' in usda_df.columns and 'year' not in usda_df.columns:
        usda_df['year'] = usda_df['YEAR']
    elif 'year' not in usda_df.columns:
        logger.error("USDA data is missing year column")
        return pd.DataFrame()
    
    if 'YEAR' in prism_df.columns and 'year' not in prism_df.columns:
        prism_df['year'] = prism_df['YEAR']
    elif 'year' not in prism_df.columns:
        logger.error("PRISM data is missing year column")
        return pd.DataFrame()
    
    # National level merge (simplest case)
    if merge_level == "national":
        # For national level, we average climate data across states
        # Group PRISM data by year
        if 'state_code' in prism_df.columns:
            # Average across states
            climate_cols = [col for col in prism_df.columns 
                          if col not in ['year', 'state_code', 'crop']]
            
            prism_national = (
                prism_df
                .groupby(['year'])
                [climate_cols]
                .mean()
                .reset_index()
            )
        else:
            # Already national level
            prism_national = prism_df
        
        # Merge on year
        merged_df = pd.merge(
            usda_df,
            prism_national,
            on='year',
            how='inner'
        )
        
        logger.info(f"Merged {len(merged_df)} rows at national level")
    
    # State level merge
    elif merge_level == "state":
        # Check if both dataframes have state columns
        if 'state_code' not in prism_df.columns:
            logger.error("PRISM data is missing state_code column, cannot merge at state level")
            return pd.DataFrame()
        
        if 'STATE_CODE' in usda_df.columns:
            usda_df['state_code'] = usda_df['STATE_CODE']
        elif 'state_code' not in usda_df.columns:
            logger.error("USDA data is missing state_code column, cannot merge at state level")
            return pd.DataFrame()
        
        # Merge on year and state
        merged_df = pd.merge(
            usda_df,
            prism_df,
            on=['year', 'state_code'],
            how='inner'
        )
        
        logger.info(f"Merged {len(merged_df)} rows at state level")
    
    # County level merge
    elif merge_level == "county":
        # Check if both dataframes have county columns
        if 'county_fips' not in prism_df.columns:
            logger.error("PRISM data is missing county_fips column, cannot merge at county level")
            return pd.DataFrame()
        
        if 'COUNTY_FIPS' in usda_df.columns:
            usda_df['county_fips'] = usda_df['COUNTY_FIPS']
        elif 'county_fips' not in usda_df.columns:
            logger.error("USDA data is missing county_fips column, cannot merge at county level")
            return pd.DataFrame()
        
        # Merge on year and county
        merged_df = pd.merge(
            usda_df,
            prism_df,
            on=['year', 'county_fips'],
            how='inner'
        )
        
        logger.info(f"Merged {len(merged_df)} rows at county level")
    
    else:
        logger.error(f"Unsupported merge level: {merge_level}")
        return pd.DataFrame()
    
    return merged_df

def create_crop_specific_datasets(usda_df, prism_df, merge_level="national"):
    """
    Create crop-specific merged datasets for each major crop.
    
    Args:
        usda_df (pd.DataFrame): USDA crop data
        prism_df (pd.DataFrame): PRISM climate data
        merge_level (str): Level at which to merge data
        
    Returns:
        dict: Dictionary of crop-specific dataframes
    """
    if 'CROP' in usda_df.columns:
        crop_col = 'CROP'
    elif 'crop' in usda_df.columns:
        crop_col = 'crop'
    else:
        logger.error("USDA data is missing crop column")
        return {}
    
    # Get unique crops
    crops = usda_df[crop_col].unique()
    
    # Create crop-specific datasets
    crop_datasets = {}
    
    for crop in crops:
        # Filter USDA data for this crop
        crop_usda = usda_df[usda_df[crop_col] == crop]
        
        # Filter PRISM data for this crop if it has a crop column
        if 'crop' in prism_df.columns:
            crop_prism = prism_df[prism_df['crop'] == crop]
        else:
            crop_prism = prism_df
        
        # Merge data
        merged_df = merge_usda_with_prism_data(crop_usda, crop_prism, merge_level)
        
        if not merged_df.empty:
            crop_datasets[crop] = merged_df
            logger.info(f"Created dataset for {crop} with {len(merged_df)} rows")
    
    return crop_datasets

def create_integrated_dataset_for_modeling(crop_datasets, standardize=True):
    """
    Create a standardized integrated dataset ready for modeling.
    
    Args:
        crop_datasets (dict): Dictionary of crop-specific dataframes
        standardize (bool): Whether to standardize numerical features
        
    Returns:
        pd.DataFrame: Integrated dataset
    """
    if not crop_datasets:
        logger.error("No crop datasets provided")
        return pd.DataFrame()
    
    # Combine all crop datasets
    all_crops = pd.concat(crop_datasets.values(), ignore_index=True)
    
    # Ensure consistent column names
    rename_mapping = {
        'YEAR': 'year',
        'CROP': 'crop',
        'STATE_CODE': 'state_code',
        'COUNTY_FIPS': 'county_fips',
        'YIELD': 'yield',
        'HARVESTED_ACRES': 'harvested_acres',
        'PRODUCTION': 'production'
    }
    
    for old_col, new_col in rename_mapping.items():
        if old_col in all_crops.columns and new_col not in all_crops.columns:
            all_crops[new_col] = all_crops[old_col]
    
    # Convert crop column to categorical
    if 'crop' in all_crops.columns:
        all_crops['crop'] = all_crops['crop'].astype('category')
    
    # Convert year to int
    if 'year' in all_crops.columns:
        all_crops['year'] = all_crops['year'].astype(int)
    
    # Convert state_code to categorical
    if 'state_code' in all_crops.columns:
        all_crops['state_code'] = all_crops['state_code'].astype('category')
    
    # Standardize numeric features if requested
    if standardize:
        # Identify numeric columns (excluding the target variable 'yield')
        numeric_cols = all_crops.select_dtypes(include=[np.number]).columns.tolist()
        if 'yield' in numeric_cols:
            numeric_cols.remove('yield')
        
        # Standardize each numeric column
        for col in numeric_cols:
            mean_val = all_crops[col].mean()
            std_val = all_crops[col].std()
            
            if std_val > 0:
                all_crops[f"{col}_std"] = (all_crops[col] - mean_val) / std_val
    
    logger.info(f"Created integrated dataset with {len(all_crops)} rows and {len(all_crops.columns)} columns")
    return all_crops

def load_and_integrate_all_data(merge_level="national", save_output=True):
    """
    Load all data and create integrated datasets.
    
    Args:
        merge_level (str): Level at which to merge data
        save_output (bool): Whether to save output files
        
    Returns:
        tuple: (combined_dataset, crop_datasets)
    """
    ensure_directories()
    
    # Load USDA data
    usda_df = load_usda_data()
    
    if usda_df.empty:
        logger.error("Failed to load USDA data")
        return pd.DataFrame(), {}
    
    # Load PRISM climate data
    prism_df = load_prism_climate_data(type="growing_season")
    
    if prism_df.empty:
        logger.error("Failed to load PRISM data")
        return pd.DataFrame(), {}
    
    # Create crop-specific datasets
    crop_datasets = create_crop_specific_datasets(usda_df, prism_df, merge_level)
    
    if not crop_datasets:
        logger.error("Failed to create crop-specific datasets")
        return pd.DataFrame(), {}
    
    # Create integrated dataset
    integrated_df = create_integrated_dataset_for_modeling(crop_datasets)
    
    if integrated_df.empty:
        logger.error("Failed to create integrated dataset")
        return pd.DataFrame(), crop_datasets
    
    # Save output files if requested
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join(PROCESSED_DATA_DIR, "integrated")
        
        # Save integrated dataset
        integrated_path = os.path.join(output_dir, f"integrated_dataset_{timestamp}.csv")
        integrated_df.to_csv(integrated_path, index=False)
        logger.info(f"Saved integrated dataset to {integrated_path}")
        
        # Save crop-specific datasets
        for crop, df in crop_datasets.items():
            crop_path = os.path.join(output_dir, f"{crop.lower()}_dataset_{timestamp}.csv")
            df.to_csv(crop_path, index=False)
            logger.info(f"Saved {crop} dataset to {crop_path}")
    
    return integrated_df, crop_datasets

def create_modeling_ready_datasets(integrated_df, target_var='yield', split_by_crop=True):
    """
    Create modeling-ready datasets with features and target variables.
    
    Args:
        integrated_df (pd.DataFrame): Integrated dataset
        target_var (str): Target variable name
        split_by_crop (bool): Whether to create separate datasets by crop
        
    Returns:
        dict: Dictionary of modeling datasets
    """
    if integrated_df.empty:
        logger.error("Empty integrated dataset provided")
        return {}
    
    if target_var not in integrated_df.columns:
        logger.error(f"Target variable {target_var} not found in dataset")
        return {}
    
    # Dictionary to store modeling datasets
    modeling_datasets = {}
    
    if split_by_crop and 'crop' in integrated_df.columns:
        # Create separate datasets for each crop
        for crop in integrated_df['crop'].unique():
            crop_df = integrated_df[integrated_df['crop'] == crop].copy()
            
            # Only include relevant columns for modeling
            # Exclude non-feature columns like original (unstandardized) numeric columns
            std_cols = [col for col in crop_df.columns if col.endswith('_std')]
            feature_cols = std_cols + [col for col in crop_df.columns 
                                    if col not in std_cols + [target_var] and 
                                    not any(col.startswith(c) for c in ['YEAR', 'STATE', 'COUNTY'])]
            
            # Create X and y
            X = crop_df[feature_cols]
            y = crop_df[target_var]
            
            modeling_datasets[crop] = {
                'X': X,
                'y': y,
                'feature_cols': feature_cols,
                'crop': crop
            }
            
            logger.info(f"Created modeling dataset for {crop} with {len(X)} samples and {len(feature_cols)} features")
    else:
        # Create a single dataset with all crops
        # If crop is a column, convert to dummy variables
        if 'crop' in integrated_df.columns:
            # Create dummy variables for crop
            crop_dummies = pd.get_dummies(integrated_df['crop'], prefix='crop')
            
            # Add dummy variables to dataset
            model_df = pd.concat([integrated_df, crop_dummies], axis=1)
        else:
            model_df = integrated_df.copy()
        
        # Only include relevant columns for modeling
        std_cols = [col for col in model_df.columns if col.endswith('_std')]
        feature_cols = std_cols + [col for col in model_df.columns 
                                if col not in std_cols + [target_var, 'crop'] and
                                not any(col.startswith(c) for c in ['YEAR', 'STATE', 'COUNTY'])]
        
        # Add crop dummy columns if they exist
        if 'crop' in integrated_df.columns:
            feature_cols += [col for col in model_df.columns if col.startswith('crop_')]
        
        # Create X and y
        X = model_df[feature_cols]
        y = model_df[target_var]
        
        modeling_datasets['all_crops'] = {
            'X': X,
            'y': y,
            'feature_cols': feature_cols,
            'crops': integrated_df['crop'].unique() if 'crop' in integrated_df.columns else None
        }
        
        logger.info(f"Created modeling dataset for all crops with {len(X)} samples and {len(feature_cols)} features")
    
    return modeling_datasets

def main():
    """Main execution function."""
    logger.info("Starting data integration process...")
    
    # Load and integrate all data
    integrated_df, crop_datasets = load_and_integrate_all_data(merge_level="national", save_output=True)
    
    if not integrated_df.empty:
        # Create modeling-ready datasets
        modeling_datasets = create_modeling_ready_datasets(integrated_df, split_by_crop=True)
        
        if modeling_datasets:
            logger.info(f"Created {len(modeling_datasets)} modeling datasets")
    
    logger.info("Data integration process complete.")

if __name__ == "__main__":
    main()