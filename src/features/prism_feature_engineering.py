#!/usr/bin/env python3
"""
Feature engineering module specifically for PRISM climate data.
Creates derived features from climate data that are relevant for crop yield prediction.
Extends the base feature engineering module with PRISM-specific functionality.
"""

import os
import numpy as np
import pandas as pd
import logging
from src.features.feature_engineering import calculate_growing_degree_days
from src.features.spatial_aggregation import extract_state_data_from_geotiff
from src.features.temporal_alignment import (
    filter_climate_data_by_growing_season,
    calculate_growing_season_statistics,
    create_climate_indices,
    align_with_crop_phenology
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

# USDA Census years
CENSUS_YEARS = [1997, 2002, 2007, 2012, 2017]

# Crop list
CROPS = ['CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'RICE']

def calculate_heat_stress_features(climate_df, base_temp=30, max_temp=None):
    """
    Calculate heat stress metrics from temperature data.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with temperature data
        base_temp (float): Base temperature above which heat stress occurs (default: 30°C)
        max_temp (float, optional): Maximum temperature cap for calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional heat stress columns
    """
    # Make a copy to avoid modifying the original
    result_df = climate_df.copy()
    
    # Check if the necessary columns exist
    temp_columns = [col for col in result_df.columns if 'tmax' in col.lower()]
    if not temp_columns:
        logger.warning("Temperature columns not found in DataFrame")
        return result_df
    
    # Use the first matching temperature column
    temp_col = temp_columns[0]
    
    # Calculate heat degree days (similar to growing degree days but for high temperatures)
    result_df['heat_degree_days'] = result_df[temp_col].apply(
        lambda x: max(0, x - base_temp) if not np.isnan(x) else 0
    )
    
    # Count extreme heat days (days above base_temp)
    if 'day' in result_df.columns:  # Only if we have daily data
        result_df['extreme_heat_day'] = (result_df[temp_col] > base_temp).astype(int)
        
        # Calculate heat stress intensity (how far above the threshold)
        result_df['heat_stress_intensity'] = result_df.apply(
            lambda row: max(0, row[temp_col] - base_temp) if row['extreme_heat_day'] == 1 else 0,
            axis=1
        )
        
        # Calculate consecutive heat stress days
        result_df['heat_stress_start'] = (
            (result_df['extreme_heat_day'] == 1) & 
            ((result_df['extreme_heat_day'].shift(1) == 0) | (result_df['extreme_heat_day'].shift(1).isna()))
        )
        
        result_df['heat_stress_group'] = result_df['heat_stress_start'].cumsum()
        
        # Only count consecutive days where extreme_heat_day is 1
        heat_stress_counts = (
            result_df[result_df['extreme_heat_day'] == 1]
            .groupby(['year', 'heat_stress_group'])
            .size()
            .reset_index(name='consecutive_heat_days')
        )
        
        # Get the maximum consecutive heat days for each year
        max_consecutive_heat = (
            heat_stress_counts
            .groupby('year')['consecutive_heat_days']
            .max()
            .reset_index()
            .rename(columns={'consecutive_heat_days': 'max_consecutive_heat_days'})
        )
        
        # Merge back to the result
        if 'year' in result_df.columns:
            result_df = pd.merge(
                result_df, 
                max_consecutive_heat, 
                on='year', 
                how='left'
            )
    
    return result_df

def calculate_drought_features(climate_df):
    """
    Calculate drought-related features from precipitation data.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with precipitation data
        
    Returns:
        pd.DataFrame: DataFrame with additional drought-related columns
    """
    # Make a copy to avoid modifying the original
    result_df = climate_df.copy()
    
    # Check if the necessary columns exist
    precip_columns = [col for col in result_df.columns if 'ppt' in col.lower() or 'prcp' in col.lower()]
    if not precip_columns:
        logger.warning("Precipitation columns not found in DataFrame")
        return result_df
    
    # Use the first matching precipitation column
    precip_col = precip_columns[0]
    
    # Define dry day threshold (1mm is common)
    dry_threshold = 1.0
    
    # Count dry days
    if 'day' in result_df.columns:  # Only if we have daily data
        result_df['dry_day'] = (result_df[precip_col] < dry_threshold).astype(int)
        
        # Calculate dry spell starts
        result_df['dry_spell_start'] = (
            (result_df['dry_day'] == 1) & 
            ((result_df['dry_day'].shift(1) == 0) | (result_df['dry_day'].shift(1).isna()))
        )
        
        result_df['dry_spell_group'] = result_df['dry_spell_start'].cumsum()
        
        # Only count consecutive days where dry_day is 1
        dry_spell_counts = (
            result_df[result_df['dry_day'] == 1]
            .groupby(['year', 'dry_spell_group'])
            .size()
            .reset_index(name='consecutive_dry_days')
        )
        
        # Get the maximum consecutive dry days for each year
        max_consecutive_dry = (
            dry_spell_counts
            .groupby('year')['consecutive_dry_days']
            .max()
            .reset_index()
            .rename(columns={'consecutive_dry_days': 'max_consecutive_dry_days'})
        )
        
        # Merge back to the result
        if 'year' in result_df.columns:
            result_df = pd.merge(
                result_df, 
                max_consecutive_dry, 
                on='year', 
                how='left'
            )
    
    # Calculate simple drought index (standardized precipitation)
    # Group by month to calculate climatology (long-term average)
    if 'month' in result_df.columns and 'year' in result_df.columns:
        # Calculate monthly climatology
        climatology = (
            result_df
            .groupby('month')[precip_col]
            .agg(['mean', 'std'])
            .reset_index()
        )
        
        # Merge with original data
        result_df = pd.merge(
            result_df,
            climatology,
            on='month',
            how='left'
        )
        
        # Calculate standardized precipitation index
        # (precip - mean) / std
        result_df['standardized_precip'] = (
            (result_df[precip_col] - result_df['mean']) / 
            result_df['std'].replace(0, np.nan)  # Avoid division by zero
        ).replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        
        # Drop intermediate columns
        result_df = result_df.drop(['mean', 'std'], axis=1)
    
    return result_df

def calculate_frost_risk_features(climate_df):
    """
    Calculate frost risk features from temperature data.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with temperature data
        
    Returns:
        pd.DataFrame: DataFrame with additional frost risk columns
    """
    # Make a copy to avoid modifying the original
    result_df = climate_df.copy()
    
    # Check if the necessary columns exist
    temp_columns = [col for col in result_df.columns if 'tmin' in col.lower()]
    if not temp_columns:
        logger.warning("Minimum temperature columns not found in DataFrame")
        return result_df
    
    # Use the first matching temperature column
    temp_col = temp_columns[0]
    
    # Define frost temperature threshold (0°C)
    frost_threshold = 0.0
    
    # Count frost days
    if 'day' in result_df.columns:  # Only if we have daily data
        result_df['frost_day'] = (result_df[temp_col] <= frost_threshold).astype(int)
        
        # Calculate monthly frost day counts if we have month data
        if 'month' in result_df.columns and 'year' in result_df.columns:
            month_frost_counts = (
                result_df
                .groupby(['year', 'month'])['frost_day']
                .sum()
                .reset_index()
                .rename(columns={'frost_day': 'frost_days_in_month'})
            )
            
            # Merge back to the result
            result_df = pd.merge(
                result_df,
                month_frost_counts,
                on=['year', 'month'],
                how='left'
            )
    
    # Identify last spring frost and first fall frost
    if 'day' in result_df.columns and 'month' in result_df.columns and 'year' in result_df.columns:
        # Add day of year for easier sorting
        result_df['day_of_year'] = result_df.apply(
            lambda row: pd.Timestamp(int(row['year']), int(row['month']), int(row['day'])).dayofyear,
            axis=1
        )
        
        # For each year, find last spring frost (last frost day before July 1)
        spring_frosts = result_df[
            (result_df['frost_day'] == 1) & 
            (result_df['day_of_year'] < 182)  # Day 182 is July 1 in non-leap years
        ]
        
        last_spring_frosts = (
            spring_frosts
            .groupby('year')['day_of_year']
            .max()
            .reset_index()
            .rename(columns={'day_of_year': 'last_spring_frost_day'})
        )
        
        # For each year, find first fall frost (first frost day after July 1)
        fall_frosts = result_df[
            (result_df['frost_day'] == 1) & 
            (result_df['day_of_year'] >= 182)
        ]
        
        first_fall_frosts = (
            fall_frosts
            .groupby('year')['day_of_year']
            .min()
            .reset_index()
            .rename(columns={'day_of_year': 'first_fall_frost_day'})
        )
        
        # Merge spring and fall frost data
        frost_dates = pd.merge(
            last_spring_frosts,
            first_fall_frosts,
            on='year',
            how='outer'
        )
        
        # Calculate growing season length (days between last spring and first fall frost)
        frost_dates['frost_free_days'] = (
            frost_dates['first_fall_frost_day'] - frost_dates['last_spring_frost_day']
        )
        
        # Merge back to the result
        result_df = pd.merge(
            result_df,
            frost_dates,
            on='year',
            how='left'
        )
        
        # Clean up temporary column
        result_df = result_df.drop('day_of_year', axis=1)
    
    return result_df

def calculate_water_balance_features(climate_df):
    """
    Calculate simple water balance features (precipitation minus potential evapotranspiration).
    This is a simplified approach; a full water balance would include soil moisture.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with temperature and precipitation data
        
    Returns:
        pd.DataFrame: DataFrame with additional water balance columns
    """
    # Make a copy to avoid modifying the original
    result_df = climate_df.copy()
    
    # Check if the necessary columns exist
    temp_columns = [col for col in result_df.columns if 'tmean' in col.lower()]
    precip_columns = [col for col in result_df.columns if 'ppt' in col.lower() or 'prcp' in col.lower()]
    
    if not temp_columns or not precip_columns:
        logger.warning("Temperature or precipitation columns not found in DataFrame")
        return result_df
    
    # Use the first matching columns
    temp_col = temp_columns[0]
    precip_col = precip_columns[0]
    
    # Calculate potential evapotranspiration (PET) using the Hargreaves method
    # This is a simplified approach that only requires temperature data
    # If we have daily data with day column
    if 'day' in result_df.columns and 'month' in result_df.columns and 'year' in result_df.columns:
        # We need to calculate daily extraterrestrial radiation
        # This is a complex calculation that depends on latitude, day of year, and solar declination
        # For simplicity, we'll use an approximation based on latitude and month
        
        # Get latitude if available, otherwise use a default for the US
        if 'latitude' in result_df.columns:
            latitude = result_df['latitude'].mean()
        else:
            latitude = 40.0  # Default latitude for Central US
        
        # Calculate day of year
        result_df['day_of_year'] = result_df.apply(
            lambda row: pd.Timestamp(int(row['year']), int(row['month']), int(row['day'])).dayofyear,
            axis=1
        )
        
        # Calculate solar declination (radians)
        result_df['solar_declination'] = 0.409 * np.sin(2 * np.pi * result_df['day_of_year'] / 365 - 1.39)
        
        # Calculate extraterrestrial radiation (Ra) in MJ/m²/day
        lat_rad = np.radians(latitude)
        result_df['et_radiation'] = (
            24 * 60 / np.pi * 0.082 * 
            (1 + 0.033 * np.cos(2 * np.pi * result_df['day_of_year'] / 365)) * 
            (
                np.cos(lat_rad) * np.cos(result_df['solar_declination']) * 
                np.sin(np.arccos(-np.tan(lat_rad) * np.tan(result_df['solar_declination']))) + 
                np.arccos(-np.tan(lat_rad) * np.tan(result_df['solar_declination'])) * 
                np.sin(lat_rad) * np.sin(result_df['solar_declination'])
            )
        )
        
        # Calculate potential evapotranspiration (PET) using Hargreaves method
        # Temperature needs to be in Celsius
        # Check if temperature might be in tenths of degrees (PRISM format)
        if result_df[temp_col].max() > 100:
            temp_factor = 10.0
        else:
            temp_factor = 1.0
        
        # Get temperature range (max - min)
        tmax_col = temp_col.replace('mean', 'max')
        tmin_col = temp_col.replace('mean', 'min')
        
        if tmax_col in result_df.columns and tmin_col in result_df.columns:
            result_df['temp_range'] = (result_df[tmax_col] - result_df[tmin_col]) / temp_factor
        else:
            # If we don't have max and min, use an approximation
            result_df['temp_range'] = 10.0  # Default diurnal temperature range
        
        # Calculate PET (mm/day)
        result_df['pet'] = 0.0023 * (result_df[temp_col] / temp_factor + 17.8) * result_df['temp_range'] ** 0.5 * result_df['et_radiation'] * 0.408
        
        # Calculate water balance (P - PET)
        # Convert precipitation to mm if needed
        precip_factor = 1.0
        if 'PRCP-NORMAL' in precip_col:  # PRISM may use tenths of mm
            precip_factor = 10.0
        
        result_df['water_balance'] = (result_df[precip_col] / precip_factor) - result_df['pet']
        
        # Clean up temporary columns
        result_df = result_df.drop(['day_of_year', 'solar_declination', 'et_radiation', 'temp_range'], axis=1)
    
    # If we only have monthly data
    elif 'month' in result_df.columns:
        # Use a simpler method based on monthly means
        # This is less accurate but requires less data
        
        # Convert temperature to Celsius if needed
        if result_df[temp_col].max() > 100:
            temp_factor = 10.0
        else:
            temp_factor = 1.0
        
        # Use the Thornthwaite method for monthly PET
        # Calculate heat index
        i_values = result_df.apply(
            lambda row: max(0, (row[temp_col] / temp_factor / 5) ** 1.514),
            axis=1
        )
        
        yearly_i = i_values.groupby(result_df['year']).sum()
        result_df = pd.merge(
            result_df,
            yearly_i.reset_index().rename(columns={0: 'heat_index'}),
            on='year',
            how='left'
        )
        
        # Days in each month
        days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
                        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        
        # Calculate monthly PET
        result_df['pet'] = result_df.apply(
            lambda row: (
                16 * (10 * row[temp_col] / temp_factor / row['heat_index']) ** 
                (0.49 + 0.01792 * row['heat_index'] - 0.0000771 * row['heat_index']**2) * 
                days_in_month[row['month']] * (1 if row['month'] in [4, 5, 8, 9] else 
                                              1.1 if row['month'] in [1, 2, 11, 12] else 1.2)
            ) if row['heat_index'] > 0 and row[temp_col] / temp_factor > 0 else 0,
            axis=1
        )
        
        # Calculate water balance (P - PET)
        # Convert precipitation to mm if needed
        precip_factor = 1.0
        if 'PRCP-NORMAL' in precip_col:  # PRISM may use tenths of mm
            precip_factor = 10.0
        
        result_df['water_balance'] = (result_df[precip_col] / precip_factor) - result_df['pet']
        
        # Clean up temporary columns
        result_df = result_df.drop('heat_index', axis=1)
    
    return result_df

def create_climate_crop_interactions(climate_features, crop_type=None):
    """
    Create interaction features between climate variables and crop-specific thresholds.
    
    Args:
        climate_features (pd.DataFrame): DataFrame with climate features
        crop_type (str, optional): Crop type to create interactions for
        
    Returns:
        pd.DataFrame: DataFrame with crop-climate interaction features
    """
    # Make a copy to avoid modifying the original
    result_df = climate_features.copy()
    
    # Define crop-specific thresholds
    crop_thresholds = {
        'CORN': {
            'ideal_temp_min': 18.0,  # °C
            'ideal_temp_max': 32.0,  # °C
            'stress_temp': 35.0,     # °C - heat stress
            'frost_temp': 0.0,       # °C - frost damage
            'ideal_precip_min': 500, # mm for growing season
            'ideal_precip_max': 800  # mm for growing season
        },
        'SOYBEANS': {
            'ideal_temp_min': 20.0,
            'ideal_temp_max': 30.0,
            'stress_temp': 35.0,
            'frost_temp': 0.0,
            'ideal_precip_min': 450,
            'ideal_precip_max': 700
        },
        'WHEAT': {
            'ideal_temp_min': 15.0,
            'ideal_temp_max': 28.0,
            'stress_temp': 32.0,
            'frost_temp': -5.0,  # Winter wheat can tolerate some frost
            'ideal_precip_min': 350,
            'ideal_precip_max': 600
        },
        'COTTON': {
            'ideal_temp_min': 20.0,
            'ideal_temp_max': 32.0,
            'stress_temp': 38.0,
            'frost_temp': 5.0,  # Cotton is frost-sensitive
            'ideal_precip_min': 500,
            'ideal_precip_max': 1000
        },
        'RICE': {
            'ideal_temp_min': 20.0,
            'ideal_temp_max': 35.0,
            'stress_temp': 40.0,
            'frost_temp': 5.0,
            'ideal_precip_min': 1000,
            'ideal_precip_max': 2000
        }
    }
    
    # Use average thresholds if no specific crop is provided
    if crop_type is None:
        # Average the thresholds across crops
        thresholds = {}
        for threshold_name in ['ideal_temp_min', 'ideal_temp_max', 'stress_temp', 
                              'frost_temp', 'ideal_precip_min', 'ideal_precip_max']:
            thresholds[threshold_name] = np.mean([
                crop_thresholds[crop][threshold_name] for crop in crop_thresholds
            ])
    else:
        if crop_type not in crop_thresholds:
            logger.warning(f"No thresholds defined for {crop_type}, using CORN thresholds")
            thresholds = crop_thresholds['CORN']
        else:
            thresholds = crop_thresholds[crop_type]
    
    # Find temperature and precipitation columns
    temp_columns = [col for col in result_df.columns 
                   if any(t in col.lower() for t in ['tmean', 'tmax', 'tmin'])]
    precip_columns = [col for col in result_df.columns 
                     if any(p in col.lower() for p in ['ppt', 'prcp', 'precip'])]
    
    if not temp_columns and not precip_columns:
        logger.warning("No temperature or precipitation columns found")
        return result_df
    
    # Process each temperature column
    for temp_col in temp_columns:
        # Determine if this is mean, max, or min temperature
        temp_type = 'mean'
        if 'max' in temp_col.lower():
            temp_type = 'max'
        elif 'min' in temp_col.lower():
            temp_type = 'min'
        
        # Check if temperature might be in tenths of degrees (PRISM format)
        if result_df[temp_col].max() > 100:
            temp_factor = 10.0
        else:
            temp_factor = 1.0
        
        # Create temperature deviation features
        if temp_type == 'mean':
            # Deviation from ideal range
            result_df[f'{temp_col}_ideal_deviation'] = result_df[temp_col].apply(
                lambda t: max(0, thresholds['ideal_temp_min'] - t / temp_factor) + 
                          max(0, t / temp_factor - thresholds['ideal_temp_max'])
            )
        elif temp_type == 'max':
            # Heat stress - how much max temp exceeds stress threshold
            result_df[f'{temp_col}_heat_stress'] = result_df[temp_col].apply(
                lambda t: max(0, t / temp_factor - thresholds['stress_temp'])
            )
        elif temp_type == 'min':
            # Frost risk - how much min temp goes below frost threshold
            result_df[f'{temp_col}_frost_risk'] = result_df[temp_col].apply(
                lambda t: max(0, thresholds['frost_temp'] - t / temp_factor)
            )
    
    # Process each precipitation column
    for precip_col in precip_columns:
        # Check if precipitation might be in tenths of mm (PRISM format)
        if 'normal' in precip_col.lower():
            precip_factor = 10.0
        else:
            precip_factor = 1.0
        
        # If we have growing season total precipitation
        if 'total' in precip_col.lower() or 'sum' in precip_col.lower() or 'season' in precip_col.lower():
            # Deviation from ideal range
            result_df[f'{precip_col}_ideal_deviation'] = result_df[precip_col].apply(
                lambda p: max(0, thresholds['ideal_precip_min'] - p / precip_factor) + 
                          max(0, p / precip_factor - thresholds['ideal_precip_max'])
            )
            
            # Drought risk
            result_df[f'{precip_col}_drought_risk'] = result_df[precip_col].apply(
                lambda p: max(0, thresholds['ideal_precip_min'] - p / precip_factor) / 
                          thresholds['ideal_precip_min'] if p > 0 else 1.0
            )
            
            # Excess moisture risk
            result_df[f'{precip_col}_excess_risk'] = result_df[precip_col].apply(
                lambda p: max(0, p / precip_factor - thresholds['ideal_precip_max']) / 
                          thresholds['ideal_precip_max'] if thresholds['ideal_precip_max'] > 0 else 0.0
            )
    
    return result_df

def create_climate_features_for_modeling(climate_data, state_level=True, crop_type=None):
    """
    Create a comprehensive set of climate features for modeling.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with basic climate data
        state_level (bool): Whether data is at state level
        crop_type (str, optional): Crop type to filter features for
        
    Returns:
        pd.DataFrame: DataFrame with engineered climate features
    """
    # Make a copy to avoid modifying the original
    result_df = climate_data.copy()
    
    # Calculate basic climate features
    logger.info("Calculating heat stress features")
    result_df = calculate_heat_stress_features(result_df)
    
    logger.info("Calculating drought features")
    result_df = calculate_drought_features(result_df)
    
    logger.info("Calculating frost risk features")
    result_df = calculate_frost_risk_features(result_df)
    
    logger.info("Calculating water balance features")
    result_df = calculate_water_balance_features(result_df)
    
    # If crop type is specified, create crop-specific features
    if crop_type:
        logger.info(f"Calculating crop-specific features for {crop_type}")
        
        # Filter by growing season
        if 'day' in result_df.columns:
            result_df = filter_climate_data_by_growing_season(result_df, crop_type)
        
        # Calculate growing season statistics
        gs_stats = calculate_growing_season_statistics(result_df, crop_type)
        
        # Merge with result (if we have state-level statistics)
        if state_level and 'state_code' in result_df.columns:
            # Group by state and year
            result_stats = pd.DataFrame()
            
            for state in result_df['state_code'].unique():
                for year in result_df['year'].unique():
                    state_year_data = result_df[
                        (result_df['state_code'] == state) & 
                        (result_df['year'] == year)
                    ]
                    
                    if state_year_data.empty:
                        continue
                    
                    # Calculate statistics for this state-year
                    state_gs_stats = calculate_growing_season_statistics(state_year_data, crop_type, state)
                    
                    # Add state and year
                    state_gs_stats_df = pd.DataFrame([{**{'state_code': state, 'year': year}, **state_gs_stats}])
                    
                    # Append to results
                    result_stats = pd.concat([result_stats, state_gs_stats_df], ignore_index=True)
            
            # Set as result
            result_df = result_stats
        
        # Create crop-climate interactions
        result_df = create_climate_crop_interactions(result_df, crop_type)
    
    return result_df

def main():
    """Test functionality."""
    logger.info("Testing PRISM feature engineering functionality...")
    
    # Create sample climate data
    import calendar
    
    sample_data = []
    
    # Daily data for a year in two states
    year = 2017
    for state_code in ['IA', 'IL']:  # Iowa and Illinois
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            
            for day in range(1, days_in_month + 1):
                # Simulate temperature and precipitation data
                # Temperature follows seasonal pattern
                tmean = 15 + 15 * np.sin((month - 1) / 12 * 2 * np.pi - np.pi/2) + np.random.normal(0, 2)
                tmin = tmean - 5 + np.random.normal(0, 1)
                tmax = tmean + 5 + np.random.normal(0, 1)
                
                # Precipitation has seasonal pattern with more in spring/summer
                ppt_prob = 0.3 + 0.2 * np.sin((month - 1) / 12 * 2 * np.pi)
                ppt = np.random.exponential(5) if np.random.random() < ppt_prob else 0
                
                sample_data.append({
                    'year': year,
                    'month': month,
                    'day': day,
                    'state_code': state_code,
                    'tmean': tmean,
                    'tmin': tmin,
                    'tmax': tmax,
                    'ppt': ppt
                })
    
    climate_df = pd.DataFrame(sample_data)
    
    # Test feature engineering functions
    crop_type = 'CORN'
    
    logger.info(f"Testing heat stress features")
    heat_stress_df = calculate_heat_stress_features(climate_df)
    new_cols = [col for col in heat_stress_df.columns if col not in climate_df.columns]
    logger.info(f"Added heat stress features: {new_cols}")
    
    logger.info(f"Testing drought features")
    drought_df = calculate_drought_features(heat_stress_df)
    new_cols = [col for col in drought_df.columns if col not in heat_stress_df.columns]
    logger.info(f"Added drought features: {new_cols}")
    
    logger.info(f"Testing frost risk features")
    frost_df = calculate_frost_risk_features(drought_df)
    new_cols = [col for col in frost_df.columns if col not in drought_df.columns]
    logger.info(f"Added frost risk features: {new_cols}")
    
    logger.info(f"Testing water balance features")
    water_df = calculate_water_balance_features(frost_df)
    new_cols = [col for col in water_df.columns if col not in frost_df.columns]
    logger.info(f"Added water balance features: {new_cols}")
    
    logger.info(f"Testing crop-climate interactions for {crop_type}")
    crop_df = create_climate_crop_interactions(water_df, crop_type)
    new_cols = [col for col in crop_df.columns if col not in water_df.columns]
    logger.info(f"Added crop interaction features: {new_cols}")
    
    logger.info(f"Testing comprehensive feature creation for {crop_type}")
    final_df = create_climate_features_for_modeling(climate_df, crop_type=crop_type)
    logger.info(f"Final features have shape: {final_df.shape}")
    
    logger.info("PRISM feature engineering testing complete")

if __name__ == "__main__":
    main()