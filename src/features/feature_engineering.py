#!/usr/bin/env python3
"""
Feature engineering module for climate and agricultural data.
Creates derived features from raw climate data that are relevant for crop yield prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_growing_degree_days(df, base_temp=10, max_temp=30):
    """
    Calculate Growing Degree Days (GDD) from temperature data.
    GDD is a measure of heat accumulation used to predict plant and pest development.
    
    Args:
        df (pd.DataFrame): DataFrame with temperature data
        base_temp (float): Base temperature below which plant growth is minimal (default: 10°C)
        max_temp (float): Maximum temperature cap for calculation (default: 30°C)
        
    Returns:
        pd.DataFrame: DataFrame with additional GDD columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the necessary columns exist
    if 'DLY-TMIN-NORMAL' not in result_df.columns or 'DLY-TMAX-NORMAL' not in result_df.columns:
        raise ValueError("Temperature columns not found in DataFrame")
    
    # Convert from tenths of degrees if needed (NOAA format)
    tmin = result_df['DLY-TMIN-NORMAL'] / 10 if result_df['DLY-TMIN-NORMAL'].max() > 100 else result_df['DLY-TMIN-NORMAL']
    tmax = result_df['DLY-TMAX-NORMAL'] / 10 if result_df['DLY-TMAX-NORMAL'].max() > 100 else result_df['DLY-TMAX-NORMAL']
    
    # Calculate daily GDD
    # GDD = max(0, (min(Tmax, max_temp) + max(Tmin, base_temp)) / 2 - base_temp)
    result_df['GDD'] = np.maximum(0, 
                                (np.minimum(tmax, max_temp) + np.maximum(tmin, base_temp)) / 2 - base_temp)
    
    # If date column exists, calculate cumulative GDD by year and station
    if 'DATE' in result_df.columns:
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(result_df['DATE']):
            result_df['DATE'] = pd.to_datetime(result_df['DATE'])
        
        # Add year column
        result_df['YEAR'] = result_df['DATE'].dt.year
        
        # Calculate cumulative GDD for each station and year
        result_df['CUMULATIVE_GDD'] = result_df.groupby(['STATION', 'YEAR'])['GDD'].cumsum()
    
    return result_df

def calculate_frost_free_period(df):
    """
    Calculate frost-free period from temperature data.
    The frost-free period is the number of days between the last spring frost and first fall frost.
    
    Args:
        df (pd.DataFrame): DataFrame with daily temperature data
        
    Returns:
        pd.DataFrame: DataFrame with frost dates and frost-free period
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the necessary columns exist
    if 'DLY-TMIN-NORMAL' not in result_df.columns:
        raise ValueError("Minimum temperature column not found in DataFrame")
    
    if 'DATE' not in result_df.columns:
        raise ValueError("Date column not found in DataFrame")
    
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(result_df['DATE']):
        result_df['DATE'] = pd.to_datetime(result_df['DATE'])
    
    # Convert from tenths of degrees if needed (NOAA format)
    tmin = result_df['DLY-TMIN-NORMAL'] / 10 if result_df['DLY-TMIN-NORMAL'].max() > 100 else result_df['DLY-TMIN-NORMAL']
    
    # Add month and day columns
    result_df['MONTH'] = result_df['DATE'].dt.month
    result_df['DAY'] = result_df['DATE'].dt.day
    
    # Define frost threshold (0°C)
    frost_threshold = 0
    
    # Flag days with frost
    result_df['HAS_FROST'] = tmin <= frost_threshold
    
    # Group by station
    stations = result_df['STATION'].unique()
    
    # Create a list to store frost date info
    frost_info = []
    
    for station in stations:
        station_data = result_df[result_df['STATION'] == station].sort_values('DATE')
        
        # Find last spring frost (last frost day before July 1)
        spring_data = station_data[station_data['DATE'] < f"{station_data['DATE'].dt.year[0]}-07-01"]
        spring_frost_days = spring_data[spring_data['HAS_FROST']].sort_values('DATE')
        
        if not spring_frost_days.empty:
            last_spring_frost = spring_frost_days.iloc[-1]
            last_spring_frost_date = last_spring_frost['DATE']
        else:
            last_spring_frost_date = None
        
        # Find first fall frost (first frost day after July 1)
        fall_data = station_data[station_data['DATE'] >= f"{station_data['DATE'].dt.year[0]}-07-01"]
        fall_frost_days = fall_data[fall_data['HAS_FROST']].sort_values('DATE')
        
        if not fall_frost_days.empty:
            first_fall_frost = fall_frost_days.iloc[0]
            first_fall_frost_date = first_fall_frost['DATE']
        else:
            first_fall_frost_date = None
        
        # Calculate frost-free period if both dates exist
        if last_spring_frost_date is not None and first_fall_frost_date is not None:
            frost_free_days = (first_fall_frost_date - last_spring_frost_date).days
        else:
            frost_free_days = None
        
        # Store the information
        frost_info.append({
            'STATION': station,
            'LAST_SPRING_FROST': last_spring_frost_date,
            'FIRST_FALL_FROST': first_fall_frost_date,
            'FROST_FREE_DAYS': frost_free_days
        })
    
    return pd.DataFrame(frost_info)

def calculate_precipitation_features(df):
    """
    Calculate precipitation-related features.
    
    Args:
        df (pd.DataFrame): DataFrame with precipitation data
        
    Returns:
        pd.DataFrame: DataFrame with additional precipitation features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if the necessary columns exist
    if 'DLY-PRCP-NORMAL' not in result_df.columns:
        raise ValueError("Precipitation column not found in DataFrame")
    
    if 'DATE' not in result_df.columns:
        raise ValueError("Date column not found in DataFrame")
    
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(result_df['DATE']):
        result_df['DATE'] = pd.to_datetime(result_df['DATE'])
    
    # Convert from tenths of mm if needed (NOAA format)
    prcp = result_df['DLY-PRCP-NORMAL'] / 10 if result_df['DLY-PRCP-NORMAL'].max() > 1000 else result_df['DLY-PRCP-NORMAL']
    
    # Create month column
    result_df['MONTH'] = result_df['DATE'].dt.month
    result_df['YEAR'] = result_df['DATE'].dt.year
    
    # Calculate monthly precipitation
    monthly_prcp = result_df.groupby(['STATION', 'YEAR', 'MONTH'])['DLY-PRCP-NORMAL'].sum().reset_index()
    monthly_prcp = monthly_prcp.rename(columns={'DLY-PRCP-NORMAL': 'MONTHLY_PRCP'})
    
    # Define growing season (April to September in Northern Hemisphere)
    growing_season_months = [4, 5, 6, 7, 8, 9]
    
    # Calculate growing season precipitation
    growing_season_prcp = monthly_prcp[monthly_prcp['MONTH'].isin(growing_season_months)]
    growing_season_total = growing_season_prcp.groupby(['STATION', 'YEAR'])['MONTHLY_PRCP'].sum().reset_index()
    growing_season_total = growing_season_total.rename(columns={'MONTHLY_PRCP': 'GROWING_SEASON_PRCP'})
    
    # Calculate number of wet days (days with precipitation > 1mm)
    result_df['WET_DAY'] = prcp > 1.0
    wet_days = result_df.groupby(['STATION', 'YEAR'])['WET_DAY'].sum().reset_index()
    wet_days = wet_days.rename(columns={'WET_DAY': 'WET_DAYS_COUNT'})
    
    # Calculate consecutive dry days
    result_df['DRY_DAY'] = prcp <= 1.0
    
    # Create a list to store consecutive dry day info
    dry_spell_info = []
    
    for station in result_df['STATION'].unique():
        station_data = result_df[result_df['STATION'] == station].sort_values('DATE')
        
        # Reset consecutive counts at the beginning of each year
        station_data['YEAR_CHANGE'] = station_data['YEAR'].diff().fillna(0) != 0
        
        # Calculate consecutive dry days
        station_data['DRY_SPELL_START'] = (station_data['DRY_DAY'] & 
                                           (~station_data['DRY_DAY'].shift(1).fillna(False) | 
                                            station_data['YEAR_CHANGE']))
        
        station_data['DRY_SPELL_GROUP'] = station_data['DRY_SPELL_START'].cumsum()
        
        # Count consecutive dry days in each group
        dry_spells = station_data[station_data['DRY_DAY']].groupby(['STATION', 'YEAR', 'DRY_SPELL_GROUP']).size().reset_index()
        dry_spells = dry_spells.rename(columns={0: 'CONSECUTIVE_DRY_DAYS'})
        
        # Find the maximum dry spell for each station and year
        max_dry_spells = dry_spells.groupby(['STATION', 'YEAR'])['CONSECUTIVE_DRY_DAYS'].max().reset_index()
        max_dry_spells = max_dry_spells.rename(columns={'CONSECUTIVE_DRY_DAYS': 'MAX_CONSECUTIVE_DRY_DAYS'})
        
        dry_spell_info.append(max_dry_spells)
    
    # Combine the dry spell information
    max_dry_spells_df = pd.concat(dry_spell_info) if dry_spell_info else pd.DataFrame()
    
    # Return the precipitation features
    return {
        'monthly_precipitation': monthly_prcp,
        'growing_season_precipitation': growing_season_total,
        'wet_days': wet_days,
        'max_consecutive_dry_days': max_dry_spells_df
    }

def create_climate_features(climate_df):
    """
    Create all climate-related features from raw climate data.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with climate data
        
    Returns:
        pd.DataFrame: DataFrame with all climate features
    """
    # Calculate growing degree days
    gdd_df = calculate_growing_degree_days(climate_df)
    
    # Calculate frost-free period
    frost_df = calculate_frost_free_period(climate_df)
    
    # Calculate precipitation features
    precip_features = calculate_precipitation_features(climate_df)
    
    # Monthly and seasonal aggregations
    if 'MONTH' not in climate_df.columns and 'DATE' in climate_df.columns:
        climate_df['MONTH'] = pd.to_datetime(climate_df['DATE']).dt.month
    
    # Define seasons
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    
    if 'MONTH' in climate_df.columns:
        climate_df['SEASON'] = climate_df['MONTH'].map(season_map)
    
    # Return all calculated features for further processing
    return {
        'climate_with_gdd': gdd_df,
        'frost_dates': frost_df,
        'precipitation_features': precip_features
    }

def merge_climate_crop_data(crop_df, climate_features, region_mapping=None):
    """
    Merge climate features with crop yield data.
    This requires mapping climate stations to crop production regions.
    
    Args:
        crop_df (pd.DataFrame): DataFrame with crop yield data
        climate_features (dict): Dictionary of DataFrames with climate features
        region_mapping (pd.DataFrame, optional): DataFrame mapping stations to crop regions
        
    Returns:
        pd.DataFrame: Merged DataFrame with crop and climate data
    """
    # This is a simplified version assuming national-level data
    # For a real implementation, you would need to map climate stations to 
    # agricultural regions and aggregate the climate data accordingly
    
    # Extract the key features
    gdd_df = climate_features['climate_with_gdd']
    frost_df = climate_features['frost_dates']
    precip_features = climate_features['precipitation_features']
    
    # Aggregate climate data to annual national averages
    # In a real implementation, this would use region_mapping to match stations to crop regions
    
    # Annual GDD
    annual_gdd = gdd_df.groupby('YEAR')['GDD'].mean().reset_index()
    annual_gdd = annual_gdd.rename(columns={'GDD': 'AVG_GDD'})
    
    # Frost-free period
    national_frost = frost_df.groupby('STATION')['FROST_FREE_DAYS'].mean().reset_index()
    avg_frost_free = national_frost['FROST_FREE_DAYS'].mean()
    
    # Growing season precipitation
    growing_season_prcp = precip_features['growing_season_precipitation']
    national_prcp = growing_season_prcp.groupby('YEAR')['GROWING_SEASON_PRCP'].mean().reset_index()
    
    # Merge with crop data
    result_df = crop_df.merge(annual_gdd, on='YEAR', how='left')
    result_df = result_df.merge(national_prcp, on='YEAR', how='left')
    
    # Add average frost-free days (this would be year-specific in a real implementation)
    result_df['AVG_FROST_FREE_DAYS'] = avg_frost_free
    
    return result_df

def engineer_features_for_modeling(merged_df):
    """
    Engineer final features for modeling.
    
    Args:
        merged_df (pd.DataFrame): Merged DataFrame with crop and climate data
        
    Returns:
        pd.DataFrame: DataFrame with engineered features for modeling
    """
    # Make a copy to avoid modifying the original
    result_df = merged_df.copy()
    
    # Calculate lagged yield features (previous census yields)
    for crop in result_df['CROP'].unique():
        crop_data = result_df[result_df['CROP'] == crop].sort_values('YEAR')
        
        # Calculate lagged yield
        crop_data['PREV_YIELD'] = crop_data['YIELD'].shift(1)
        
        # Calculate yield change
        crop_data['YIELD_CHANGE'] = crop_data['YIELD'] - crop_data['PREV_YIELD']
        crop_data['YIELD_CHANGE_PCT'] = crop_data['YIELD_CHANGE'] / crop_data['PREV_YIELD'] * 100
        
        # Update the result dataframe
        result_df.loc[result_df['CROP'] == crop, 'PREV_YIELD'] = crop_data['PREV_YIELD']
        result_df.loc[result_df['CROP'] == crop, 'YIELD_CHANGE'] = crop_data['YIELD_CHANGE']
        result_df.loc[result_df['CROP'] == crop, 'YIELD_CHANGE_PCT'] = crop_data['YIELD_CHANGE_PCT']
    
    # Feature interactions
    if 'AVG_GDD' in result_df.columns and 'GROWING_SEASON_PRCP' in result_df.columns:
        # GDD x Precipitation interaction (heat and moisture combined effect)
        result_df['GDD_PRCP_INTERACTION'] = result_df['AVG_GDD'] * result_df['GROWING_SEASON_PRCP']
    
    # Normalize the features
    for feature in ['AVG_GDD', 'GROWING_SEASON_PRCP', 'GDD_PRCP_INTERACTION']:
        if feature in result_df.columns:
            feature_mean = result_df[feature].mean()
            feature_std = result_df[feature].std()
            result_df[f'{feature}_NORM'] = (result_df[feature] - feature_mean) / feature_std
    
    return result_df

if __name__ == "__main__":
    # Test code to verify functionality
    import os
    
    # Sample paths
    sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample")
    usda_sample_path = os.path.join(sample_data_dir, "usda_crop_data_sample.csv")
    noaa_sample_path = os.path.join(sample_data_dir, "noaa_climate_normals_sample.csv")
    
    # Load sample data
    try:
        usda_data = pd.read_csv(usda_sample_path)
        noaa_data = pd.read_csv(noaa_sample_path)
        
        # Convert date column
        noaa_data['DATE'] = pd.to_datetime(noaa_data['DATE'])
        
        # Create climate features
        climate_features = create_climate_features(noaa_data)
        
        # Merge data
        merged_data = merge_climate_crop_data(usda_data, climate_features)
        
        # Engineer features
        final_features = engineer_features_for_modeling(merged_data)
        
        print("Feature engineering successful!")
        print(f"Resulting DataFrame shape: {final_features.shape}")
        
    except FileNotFoundError:
        print("Sample data files not found. Please run the data download scripts first.")