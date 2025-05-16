#!/usr/bin/env python3
"""
Temporal alignment module for climate and agricultural data.
Aligns climate data with crop-specific growing seasons and agricultural census years.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import calendar

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

# Define growing seasons for major field crops
# This is a simplified version. In a real application, these should be state-specific
CROP_GROWING_SEASONS = {
    'CORN': {
        'start_month': 4,  # April
        'start_day': 15,
        'end_month': 9,    # September
        'end_day': 30,
        'critical_periods': [
            {'name': 'planting', 'start_month': 4, 'start_day': 15, 'end_month': 5, 'end_day': 31},
            {'name': 'vegetative', 'start_month': 6, 'start_day': 1, 'end_month': 7, 'end_day': 15},
            {'name': 'reproductive', 'start_month': 7, 'start_day': 16, 'end_month': 8, 'end_day': 31},
            {'name': 'maturation', 'start_month': 9, 'start_day': 1, 'end_month': 9, 'end_day': 30}
        ]
    },
    'SOYBEANS': {
        'start_month': 5,  # May
        'start_day': 1,
        'end_month': 10,   # October
        'end_day': 15,
        'critical_periods': [
            {'name': 'planting', 'start_month': 5, 'start_day': 1, 'end_month': 5, 'end_day': 31},
            {'name': 'vegetative', 'start_month': 6, 'start_day': 1, 'end_month': 7, 'end_day': 31},
            {'name': 'reproductive', 'start_month': 8, 'start_day': 1, 'end_month': 9, 'end_day': 15},
            {'name': 'maturation', 'start_month': 9, 'start_day': 16, 'end_month': 10, 'end_day': 15}
        ]
    },
    'WHEAT': {
        # Winter wheat (most common)
        'start_month': 9,  # September (planting)
        'start_day': 1,
        'end_month': 7,    # July (harvest)
        'end_day': 15,
        'critical_periods': [
            {'name': 'fall_planting', 'start_month': 9, 'start_day': 1, 'end_month': 10, 'end_day': 31},
            {'name': 'winter_dormancy', 'start_month': 11, 'start_day': 1, 'end_month': 2, 'end_day': 28},
            {'name': 'spring_growth', 'start_month': 3, 'start_day': 1, 'end_month': 5, 'end_day': 31},
            {'name': 'grain_fill', 'start_month': 6, 'start_day': 1, 'end_month': 7, 'end_day': 15}
        ]
    },
    'COTTON': {
        'start_month': 4,  # April
        'start_day': 1,
        'end_month': 10,   # October
        'end_day': 31,
        'critical_periods': [
            {'name': 'planting', 'start_month': 4, 'start_day': 1, 'end_month': 5, 'end_day': 15},
            {'name': 'vegetative', 'start_month': 5, 'start_day': 16, 'end_month': 7, 'end_day': 15},
            {'name': 'flowering', 'start_month': 7, 'start_day': 16, 'end_month': 8, 'end_day': 31},
            {'name': 'boll_development', 'start_month': 9, 'start_day': 1, 'end_month': 10, 'end_day': 31}
        ]
    },
    'RICE': {
        'start_month': 3,  # March
        'start_day': 15,
        'end_month': 9,    # September
        'end_day': 15,
        'critical_periods': [
            {'name': 'planting', 'start_month': 3, 'start_day': 15, 'end_month': 4, 'end_day': 30},
            {'name': 'vegetative', 'start_month': 5, 'start_day': 1, 'end_month': 6, 'end_day': 30},
            {'name': 'reproductive', 'start_month': 7, 'start_day': 1, 'end_month': 8, 'end_day': 15},
            {'name': 'maturation', 'start_month': 8, 'start_day': 16, 'end_month': 9, 'end_day': 15}
        ]
    }
}

# Regional adjustments (days to add/subtract from standard dates)
# Positive values mean later planting/harvest, negative values mean earlier
REGIONAL_ADJUSTMENTS = {
    'CORN': {
        'Northern': {'start': 15, 'end': -10},  # Later planting, earlier harvest
        'Central': {'start': 0, 'end': 0},      # Standard dates
        'Southern': {'start': -15, 'end': 10}   # Earlier planting, later harvest
    },
    'SOYBEANS': {
        'Northern': {'start': 10, 'end': -10},
        'Central': {'start': 0, 'end': 0},
        'Southern': {'start': -10, 'end': 5}
    },
    'WHEAT': {
        'Northern': {'start': 0, 'end': 10},    # Later harvest in north
        'Central': {'start': 0, 'end': 0},
        'Southern': {'start': 0, 'end': -10}    # Earlier harvest in south
    },
    'COTTON': {
        'Northern': {'start': 20, 'end': -15},
        'Central': {'start': 10, 'end': -5},
        'Southern': {'start': 0, 'end': 0}
    },
    'RICE': {
        'Northern': {'start': 15, 'end': -10},
        'Central': {'start': 5, 'end': -5},
        'Southern': {'start': 0, 'end': 0}
    }
}

# State region mappings
STATE_REGIONS = {
    # Northern
    'ND': 'Northern', 'SD': 'Northern', 'MN': 'Northern', 'WI': 'Northern',
    'MI': 'Northern', 'MT': 'Northern', 'ID': 'Northern', 'WA': 'Northern',
    
    # Central
    'IA': 'Central', 'IL': 'Central', 'IN': 'Central', 'OH': 'Central', 
    'NE': 'Central', 'KS': 'Central', 'MO': 'Central', 'KY': 'Central',
    'CO': 'Central', 'WY': 'Central', 'UT': 'Central', 'NV': 'Central',
    'NY': 'Central', 'PA': 'Central', 'WV': 'Central', 'VA': 'Central',
    
    # Southern
    'TX': 'Southern', 'OK': 'Southern', 'AR': 'Southern', 'LA': 'Southern',
    'MS': 'Southern', 'AL': 'Southern', 'GA': 'Southern', 'SC': 'Southern',
    'NC': 'Southern', 'TN': 'Southern', 'FL': 'Southern', 'CA': 'Southern',
    'AZ': 'Southern', 'NM': 'Southern'
}

def get_adjusted_growing_season(crop, state_code):
    """
    Get the adjusted growing season dates for a specific crop and state.
    
    Args:
        crop (str): Crop name (e.g., 'CORN')
        state_code (str): State code (e.g., 'IA')
        
    Returns:
        dict: Dictionary with adjusted start and end dates
    """
    # Get base growing season
    base_season = CROP_GROWING_SEASONS.get(crop, None)
    if not base_season:
        logger.warning(f"No growing season defined for {crop}")
        return None
    
    # Get region for state
    region = STATE_REGIONS.get(state_code, 'Central')  # Default to Central if not found
    
    # Get regional adjustments
    regional_adj = REGIONAL_ADJUSTMENTS.get(crop, {}).get(region, {'start': 0, 'end': 0})
    
    # Create adjusted dates
    start_month = base_season['start_month']
    start_day = base_season['start_day']
    end_month = base_season['end_month']
    end_day = base_season['end_day']
    
    # Create date objects
    # Note: We use a leap year (2020) to handle February 29
    start_date = datetime(2020, start_month, start_day)
    end_date = datetime(2020, end_month, end_day)
    
    # Apply adjustments
    start_date = start_date + timedelta(days=regional_adj['start'])
    end_date = end_date + timedelta(days=regional_adj['end'])
    
    # Handle adjustments that cross month boundaries
    adjusted_start_month = start_date.month
    adjusted_start_day = start_date.day
    adjusted_end_month = end_date.month
    adjusted_end_day = end_date.day
    
    # Adjust critical periods too
    adjusted_critical_periods = []
    for period in base_season.get('critical_periods', []):
        cp_start_date = datetime(2020, period['start_month'], period['start_day'])
        cp_end_date = datetime(2020, period['end_month'], period['end_day'])
        
        # Apply proportional adjustments based on period's position in growing season
        if period['name'] == 'planting' or period['name'] == 'fall_planting':
            cp_start_date = cp_start_date + timedelta(days=regional_adj['start'])
            # End date of planting is adjusted proportionally (less than full adjustment)
            cp_end_date = cp_end_date + timedelta(days=int(regional_adj['start'] * 0.5))
        elif period['name'] == 'maturation' or period['name'] == 'grain_fill' or period['name'] == 'boll_development':
            # Start date of final period is adjusted proportionally
            cp_start_date = cp_start_date + timedelta(days=int(regional_adj['end'] * 0.5))
            cp_end_date = cp_end_date + timedelta(days=regional_adj['end'])
        else:
            # Middle periods get smaller adjustments
            cp_start_date = cp_start_date + timedelta(days=int(regional_adj['start'] * 0.3))
            cp_end_date = cp_end_date + timedelta(days=int(regional_adj['end'] * 0.3))
        
        adjusted_critical_periods.append({
            'name': period['name'],
            'start_month': cp_start_date.month,
            'start_day': cp_start_date.day,
            'end_month': cp_end_date.month,
            'end_day': cp_end_date.day
        })
    
    return {
        'start_month': adjusted_start_month,
        'start_day': adjusted_start_day,
        'end_month': adjusted_end_month,
        'end_day': adjusted_end_day,
        'critical_periods': adjusted_critical_periods
    }

def get_growing_season_days(crop, state_code):
    """
    Get a list of all days in the growing season for a specific crop and state.
    
    Args:
        crop (str): Crop name
        state_code (str): State code
        
    Returns:
        list: List of (month, day) tuples representing days in the growing season
    """
    adjusted_season = get_adjusted_growing_season(crop, state_code)
    if not adjusted_season:
        return []
    
    # Extract season information
    start_month = adjusted_season['start_month']
    start_day = adjusted_season['start_day']
    end_month = adjusted_season['end_month']
    end_day = adjusted_season['end_day']
    
    # Special case: if season spans across years (e.g., winter wheat)
    # For simplicity, we use a non-leap year for consistent month lengths
    year = 2019
    
    start_date = datetime(year, start_month, start_day)
    
    # If end month is earlier in the year than start month, it's in the next year
    if end_month < start_month:
        end_date = datetime(year + 1, end_month, end_day)
    else:
        end_date = datetime(year, end_month, end_day)
    
    # Generate list of days
    days_list = []
    current_date = start_date
    
    while current_date <= end_date:
        days_list.append((current_date.month, current_date.day))
        current_date += timedelta(days=1)
    
    return days_list

def get_critical_period_days(crop, state_code, period_name):
    """
    Get a list of all days in a specific critical period for a crop and state.
    
    Args:
        crop (str): Crop name
        state_code (str): State code
        period_name (str): Name of the critical period
        
    Returns:
        list: List of (month, day) tuples representing days in the critical period
    """
    adjusted_season = get_adjusted_growing_season(crop, state_code)
    if not adjusted_season or 'critical_periods' not in adjusted_season:
        return []
    
    # Find the specified critical period
    period = None
    for p in adjusted_season['critical_periods']:
        if p['name'] == period_name:
            period = p
            break
    
    if not period:
        logger.warning(f"Critical period '{period_name}' not found for {crop}")
        return []
    
    # Extract period information
    start_month = period['start_month']
    start_day = period['start_day']
    end_month = period['end_month']
    end_day = period['end_day']
    
    # For simplicity, we use a non-leap year for consistent month lengths
    year = 2019
    
    start_date = datetime(year, start_month, start_day)
    
    # If end month is earlier in the year than start month, it's in the next year
    if end_month < start_month:
        end_date = datetime(year + 1, end_month, end_day)
    else:
        end_date = datetime(year, end_month, end_day)
    
    # Generate list of days
    days_list = []
    current_date = start_date
    
    while current_date <= end_date:
        days_list.append((current_date.month, current_date.day))
        current_date += timedelta(days=1)
    
    return days_list

def filter_climate_data_by_growing_season(climate_data, crop, state_code=None):
    """
    Filter climate data to include only days within the growing season for a specific crop.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data (must have month and day columns)
        crop (str): Crop name
        state_code (str, optional): State code. If None, uses standard growing season
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only growing season data
    """
    if 'month' not in climate_data.columns:
        logger.error("Climate data must include 'month' column")
        return climate_data
    
    # For daily data
    if 'day' in climate_data.columns:
        # Get growing season days
        if state_code:
            gs_days = get_growing_season_days(crop, state_code)
        else:
            # Use standard growing season for the crop
            base_season = CROP_GROWING_SEASONS.get(crop, None)
            if not base_season:
                logger.warning(f"No growing season defined for {crop}")
                return climate_data
            
            # Generate standard growing season days
            gs_days = get_growing_season_days(crop, 'IA')  # Use Iowa as default
        
        # Filter data
        return climate_data[climate_data.apply(
            lambda row: (row['month'], row['day']) in gs_days, axis=1
        )]
    
    # For monthly data, filter by growing season months
    else:
        base_season = CROP_GROWING_SEASONS.get(crop, None)
        if not base_season:
            logger.warning(f"No growing season defined for {crop}")
            return climate_data
        
        start_month = base_season['start_month']
        end_month = base_season['end_month']
        
        # If season spans across years
        if start_month > end_month:
            return climate_data[
                (climate_data['month'] >= start_month) | 
                (climate_data['month'] <= end_month)
            ]
        else:
            return climate_data[
                (climate_data['month'] >= start_month) & 
                (climate_data['month'] <= end_month)
            ]

def filter_climate_data_by_critical_period(climate_data, crop, period_name, state_code=None):
    """
    Filter climate data to include only days within a critical period for a specific crop.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data (must have month and day columns)
        crop (str): Crop name
        period_name (str): Name of the critical period
        state_code (str, optional): State code. If None, uses standard growing season
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only critical period data
    """
    if 'month' not in climate_data.columns:
        logger.error("Climate data must include 'month' column")
        return climate_data
    
    # For daily data
    if 'day' in climate_data.columns:
        # Get critical period days
        if state_code:
            cp_days = get_critical_period_days(crop, state_code, period_name)
        else:
            # Use standard critical period for the crop
            base_season = CROP_GROWING_SEASONS.get(crop, None)
            if not base_season or 'critical_periods' not in base_season:
                logger.warning(f"No critical periods defined for {crop}")
                return climate_data
            
            # Find the specified critical period
            period = None
            for p in base_season['critical_periods']:
                if p['name'] == period_name:
                    period = p
                    break
            
            if not period:
                logger.warning(f"Critical period '{period_name}' not found for {crop}")
                return climate_data
            
            # Generate standard critical period days
            cp_days = get_critical_period_days(crop, 'IA', period_name)  # Use Iowa as default
        
        # Filter data
        return climate_data[climate_data.apply(
            lambda row: (row['month'], row['day']) in cp_days, axis=1
        )]
    
    # For monthly data, it's harder to filter by critical period precisely
    # We'll just include months that overlap with the critical period
    else:
        base_season = CROP_GROWING_SEASONS.get(crop, None)
        if not base_season or 'critical_periods' not in base_season:
            logger.warning(f"No critical periods defined for {crop}")
            return climate_data
        
        # Find the specified critical period
        period = None
        for p in base_season['critical_periods']:
            if p['name'] == period_name:
                period = p
                break
        
        if not period:
            logger.warning(f"Critical period '{period_name}' not found for {crop}")
            return climate_data
        
        start_month = period['start_month']
        end_month = period['end_month']
        
        # If period spans across years
        if start_month > end_month:
            return climate_data[
                (climate_data['month'] >= start_month) | 
                (climate_data['month'] <= end_month)
            ]
        else:
            return climate_data[
                (climate_data['month'] >= start_month) & 
                (climate_data['month'] <= end_month)
            ]

def calculate_growing_season_statistics(climate_data, crop, state_code=None):
    """
    Calculate statistics for growing season climate variables.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data
        crop (str): Crop name
        state_code (str, optional): State code
        
    Returns:
        dict: Dictionary with growing season statistics
    """
    # Filter data to growing season
    gs_data = filter_climate_data_by_growing_season(climate_data, crop, state_code)
    
    if gs_data.empty:
        logger.warning(f"No growing season data available for {crop} in {state_code or 'default region'}")
        return {}
    
    # Calculate statistics for each climate variable
    statistics = {}
    
    # Check what variables are available
    numeric_columns = gs_data.select_dtypes(include=[np.number]).columns
    
    # Variables of interest
    var_mapping = {
        'tmin': 'temperature_min',
        'tmax': 'temperature_max',
        'tmean': 'temperature_mean',
        'ppt': 'precipitation',
        'DLY-TMIN-NORMAL': 'temperature_min',
        'DLY-TMAX-NORMAL': 'temperature_max',
        'DLY-TMEAN-NORMAL': 'temperature_mean',
        'DLY-PRCP-NORMAL': 'precipitation'
    }
    
    for col in numeric_columns:
        # Skip certain columns
        if col in ['month', 'day', 'year', 'latitude', 'longitude']:
            continue
        
        # Get standardized name
        var_name = var_mapping.get(col, col)
        
        # Calculate statistics
        if 'precipitation' in var_name.lower() or 'ppt' in var_name.lower():
            # For precipitation, we're interested in total
            statistics[f"{var_name}_total"] = gs_data[col].sum()
            statistics[f"{var_name}_max_daily"] = gs_data[col].max()
            
            # Count days with precipitation
            if 'day' in gs_data.columns:
                statistics[f"days_with_rain"] = (gs_data[col] > 0).sum()
        else:
            # For temperature and other variables
            statistics[f"{var_name}_mean"] = gs_data[col].mean()
            statistics[f"{var_name}_min"] = gs_data[col].min()
            statistics[f"{var_name}_max"] = gs_data[col].max()
            
            # Calculate GDD if this is temperature
            if 'temp' in var_name.lower():
                # Use 10°C as base temperature
                base_temp = 10
                # Cap at 30°C for max temp
                if 'max' in var_name.lower():
                    gdd_values = gs_data[col].apply(lambda x: max(0, min(x, 30) - base_temp))
                else:
                    gdd_values = gs_data[col].apply(lambda x: max(0, x - base_temp))
                
                statistics[f"gdd_base{base_temp}"] = gdd_values.sum()
    
    # Process by critical periods
    for period in CROP_GROWING_SEASONS.get(crop, {}).get('critical_periods', []):
        period_name = period['name']
        period_data = filter_climate_data_by_critical_period(climate_data, crop, period_name, state_code)
        
        if period_data.empty:
            continue
        
        for col in numeric_columns:
            # Skip certain columns
            if col in ['month', 'day', 'year', 'latitude', 'longitude']:
                continue
            
            # Get standardized name
            var_name = var_mapping.get(col, col)
            
            # Calculate statistics
            if 'precipitation' in var_name.lower() or 'ppt' in var_name.lower():
                # For precipitation, we're interested in total
                statistics[f"{period_name}_{var_name}_total"] = period_data[col].sum()
            else:
                # For temperature and other variables
                statistics[f"{period_name}_{var_name}_mean"] = period_data[col].mean()
    
    return statistics

def align_climate_data_with_census_years(climate_data, state_level=True):
    """
    Align climate data with USDA Census years by calculating multi-year averages.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data
        state_level (bool): Whether data is at state level
        
    Returns:
        pd.DataFrame: DataFrame with climate data aligned to census years
    """
    if 'year' not in climate_data.columns:
        logger.error("Climate data must include 'year' column")
        return climate_data
    
    # Create empty DataFrame to store results
    aligned_data = pd.DataFrame()
    
    for census_year in CENSUS_YEARS:
        # For each census year, use 3-year average centered on the year
        # (unless it's too close to the edge of our data)
        years_to_include = [census_year - 1, census_year, census_year + 1]
        years_to_include = [y for y in years_to_include if y in climate_data['year'].unique()]
        
        if not years_to_include:
            logger.warning(f"No climate data available for census year {census_year}")
            continue
        
        # Filter to relevant years
        period_data = climate_data[climate_data['year'].isin(years_to_include)]
        
        if state_level:
            # Group by state and calculate averages
            group_cols = ['state_code', 'variable']
        else:
            # National level
            group_cols = ['variable']
        
        # Calculate averages for all numeric columns
        numeric_cols = period_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day']]
        
        if not numeric_cols:
            logger.warning(f"No numeric columns to average for census year {census_year}")
            continue
        
        # Group and calculate averages
        period_avg = period_data.groupby(group_cols)[numeric_cols].mean().reset_index()
        
        # Add census year
        period_avg['census_year'] = census_year
        
        # Append to results
        aligned_data = pd.concat([aligned_data, period_avg], ignore_index=True)
    
    return aligned_data

def create_climate_indices(climate_data, temp_col='tmean', precip_col='ppt'):
    """
    Create climate indices useful for agricultural analysis.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data
        temp_col (str): Column name for temperature
        precip_col (str): Column name for precipitation
        
    Returns:
        pd.DataFrame: DataFrame with additional climate indices
    """
    # Make a copy to avoid modifying the original
    result_df = climate_data.copy()
    
    # Check if necessary columns exist
    if temp_col not in result_df.columns:
        logger.warning(f"Temperature column '{temp_col}' not found")
        return result_df
    
    if precip_col not in result_df.columns:
        logger.warning(f"Precipitation column '{precip_col}' not found")
        return result_df
    
    # Heat Stress Index (HSI) - count days above 30°C (86°F)
    if 'day' in result_df.columns:  # Only for daily data
        result_df['heat_stress_day'] = (result_df[temp_col] > 30).astype(int)
        
        # Frost/Cold Stress - count days below 0°C (32°F)
        result_df['frost_day'] = (result_df[temp_col] < 0).astype(int)
        
        # Dry Day - precipitation below 1mm
        result_df['dry_day'] = (result_df[precip_col] < 1).astype(int)
        
        # Wet Day - precipitation above 10mm
        result_df['wet_day'] = (result_df[precip_col] > 10).astype(int)
    
    # Calculate more indices at monthly level
    if 'month' in result_df.columns:
        # Aridity Index (simple version) - ratio of precipitation to temperature
        # Higher values indicate more moisture relative to temperature
        # Add small constant to avoid division by zero
        # Convert to absolute temperature if it's negative
        result_df['aridity_index'] = result_df[precip_col] / (np.abs(result_df[temp_col]) + 0.1)
        
        # Growing Degree Days (base 10°C)
        result_df['gdd_base10'] = result_df[temp_col].apply(lambda x: max(0, x - 10))
        
        # Cooling Degree Days (base 30°C) - measure of heat stress
        result_df['cdd_base30'] = result_df[temp_col].apply(lambda x: max(0, x - 30))
    
    return result_df

def align_with_crop_phenology(climate_data, crop, state_code=None):
    """
    Align climate data with crop phenology stages.
    
    Args:
        climate_data (pd.DataFrame): DataFrame with climate data
        crop (str): Crop name
        state_code (str, optional): State code
        
    Returns:
        pd.DataFrame: DataFrame with climate data aligned to phenology stages
    """
    # Get adjusted growing season
    adjusted_season = get_adjusted_growing_season(crop, state_code)
    
    if not adjusted_season or 'critical_periods' not in adjusted_season:
        logger.warning(f"No critical periods defined for {crop}")
        return climate_data
    
    # Create a new column for phenology stage
    result_df = climate_data.copy()
    result_df['phenology_stage'] = 'outside_growing_season'
    
    # Check if daily data
    if 'day' in result_df.columns:
        # Assign phenology stages based on critical periods
        for period in adjusted_season['critical_periods']:
            period_name = period['name']
            period_days = get_critical_period_days(crop, state_code or 'IA', period_name)
            
            # Assign this period to matching days
            for idx, row in result_df.iterrows():
                if (row['month'], row['day']) in period_days:
                    result_df.at[idx, 'phenology_stage'] = period_name
    
    # For monthly data, assign based on dominant period in each month
    else:
        for month in range(1, 13):
            # Count days in each period for this month
            period_counts = {}
            
            # For each critical period, count days in this month
            for period in adjusted_season['critical_periods']:
                period_name = period['name']
                period_days = get_critical_period_days(crop, state_code or 'IA', period_name)
                
                # Count days in this month
                days_in_month = [(m, d) for m, d in period_days if m == month]
                period_counts[period_name] = len(days_in_month)
            
            # If any period has days in this month
            if any(period_counts.values()):
                # Assign the period with the most days
                dominant_period = max(period_counts.items(), key=lambda x: x[1])
                if dominant_period[1] > 0:  # If there are any days
                    result_df.loc[result_df['month'] == month, 'phenology_stage'] = dominant_period[0]
    
    return result_df

def main():
    """Test functionality."""
    logger.info("Testing temporal alignment functionality...")
    
    # Create sample climate data
    sample_data = []
    
    # Daily data for a year
    year = 2017
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
                'state_code': 'IA',  # Iowa
                'tmean': tmean,
                'tmin': tmin,
                'tmax': tmax,
                'ppt': ppt
            })
    
    climate_df = pd.DataFrame(sample_data)
    
    # Test growing season filtering
    crop = 'CORN'
    state_code = 'IA'
    
    logger.info(f"Testing growing season filtering for {crop} in {state_code}")
    gs_data = filter_climate_data_by_growing_season(climate_df, crop, state_code)
    logger.info(f"Growing season data has {len(gs_data)} days out of {len(climate_df)} total days")
    
    # Test critical period filtering
    period_name = 'reproductive'
    logger.info(f"Testing critical period ({period_name}) filtering for {crop}")
    cp_data = filter_climate_data_by_critical_period(climate_df, crop, period_name, state_code)
    logger.info(f"Critical period data has {len(cp_data)} days")
    
    # Test growing season statistics
    logger.info("Testing growing season statistics calculation")
    gs_stats = calculate_growing_season_statistics(climate_df, crop, state_code)
    logger.info(f"Calculated {len(gs_stats)} statistics")
    
    # Test climate indices
    logger.info("Testing climate indices creation")
    indices_df = create_climate_indices(climate_df)
    new_cols = [col for col in indices_df.columns if col not in climate_df.columns]
    logger.info(f"Added {len(new_cols)} climate indices: {', '.join(new_cols)}")
    
    # Test phenology alignment
    logger.info("Testing phenology alignment")
    pheno_df = align_with_crop_phenology(climate_df, crop, state_code)
    phenology_counts = pheno_df['phenology_stage'].value_counts()
    logger.info(f"Phenology stage counts:\n{phenology_counts}")
    
    logger.info("Temporal alignment testing complete")

if __name__ == "__main__":
    main()