#!/usr/bin/env python3
"""
Module for processing PRISM climate data from GeoTIFF files.
Extracts climate data, performs spatial aggregation to state level,
and creates processed datasets for analysis.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "prism")
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim", "prism")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "prism")

# Path to US state boundaries shapefile 
# This would need to be downloaded from a source like the US Census Bureau
STATE_BOUNDARIES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   "data", "raw", "boundaries", "us_states.shp")

# USDA Census years
CENSUS_YEARS = [1997, 2002, 2007, 2012, 2017]

# Agricultural states (major crop-producing states)
AGRICULTURAL_STATES = [
    'IL', 'IA', 'NE', 'MN', 'IN', 'KS', 'SD', 'OH', 'ND', 'MO',  # Corn Belt & Northern Plains
    'AR', 'MS', 'LA', 'TX', 'AL', 'GA', 'SC', 'NC', 'TN',        # Southern states
    'CA', 'WA', 'OR', 'ID', 'CO', 'MT', 'WY', 'UT'               # Western states
]

# Define growing seasons for major crops
GROWING_SEASONS = {
    'CORN': {'start_month': 4, 'end_month': 9},     # April to September
    'SOYBEANS': {'start_month': 5, 'end_month': 10}, # May to October
    'WHEAT': {'start_month': 9, 'end_month': 6},     # September to June (winter wheat)
    'COTTON': {'start_month': 4, 'end_month': 9},    # April to September
    'RICE': {'start_month': 4, 'end_month': 9}       # April to September
}

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_state_boundaries():
    """
    Load US state boundaries from shapefile.
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing state boundaries
    """
    try:
        # Check if the shapefile exists
        if not os.path.exists(STATE_BOUNDARIES_PATH):
            logger.warning(f"State boundaries shapefile not found at {STATE_BOUNDARIES_PATH}")
            logger.warning("Using sample data instead of actual state boundaries")
            return create_sample_state_boundaries()
        
        # Load the shapefile
        state_boundaries = gpd.read_file(STATE_BOUNDARIES_PATH)
        logger.info(f"Loaded state boundaries with {len(state_boundaries)} states")
        return state_boundaries
    
    except Exception as e:
        logger.error(f"Error loading state boundaries: {e}")
        logger.warning("Using sample data instead of actual state boundaries")
        return create_sample_state_boundaries()

def create_sample_state_boundaries():
    """
    Create a sample GeoDataFrame with simplified state boundaries for testing.
    
    Returns:
        geopandas.GeoDataFrame: Sample GeoDataFrame with state boundaries
    """
    from shapely.geometry import Polygon
    
    # Create simplified polygons for a few agricultural states
    state_data = [
        {'STATE_CODE': 'IL', 'STATE_NAME': 'Illinois', 'geometry': Polygon([
            (-91.5, 37.0), (-89.0, 37.0), (-89.0, 42.5), (-91.5, 42.5)
        ])},
        {'STATE_CODE': 'IA', 'STATE_NAME': 'Iowa', 'geometry': Polygon([
            (-96.6, 40.4), (-91.0, 40.4), (-91.0, 43.5), (-96.6, 43.5)
        ])},
        {'STATE_CODE': 'NE', 'STATE_NAME': 'Nebraska', 'geometry': Polygon([
            (-104.0, 40.0), (-95.3, 40.0), (-95.3, 43.0), (-104.0, 43.0)
        ])},
        {'STATE_CODE': 'MN', 'STATE_NAME': 'Minnesota', 'geometry': Polygon([
            (-97.2, 43.5), (-91.4, 43.5), (-91.4, 49.4), (-97.2, 49.4)
        ])},
        {'STATE_CODE': 'IN', 'STATE_NAME': 'Indiana', 'geometry': Polygon([
            (-88.1, 37.8), (-84.8, 37.8), (-84.8, 41.8), (-88.1, 41.8)
        ])}
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(state_data, crs="EPSG:4326")
    logger.info(f"Created sample state boundaries with {len(gdf)} states")
    return gdf

def find_geotiff_files(variable, year=None, month=None):
    """
    Find GeoTIFF files for a specific variable, year, and month.
    
    Args:
        variable (str): Climate variable name (e.g., 'ppt', 'tmin')
        year (int, optional): Year to filter by
        month (int, optional): Month to filter by
        
    Returns:
        list: List of file paths matching the criteria
    """
    # Check if we're using sample data
    sample_dir = os.path.join(RAW_DATA_DIR, "sample", variable)
    if os.path.exists(sample_dir):
        pattern = f"{variable}_"
        if year is not None:
            pattern += f"{year}"
            if month is not None:
                pattern += f"{month:02d}"
        pattern += "*_sample.csv"
        
        files = glob.glob(os.path.join(sample_dir, pattern))
        logger.info(f"Found {len(files)} sample files for {variable}" + 
                  (f" in {year}" if year else "") + 
                  (f" month {month}" if month else ""))
        return files
    
    # Search for actual GeoTIFF files
    pattern = os.path.join(RAW_DATA_DIR, f"{variable}_*", "*.bil")
    
    # Filter by date if provided
    files = glob.glob(pattern)
    
    if year is not None:
        files = [f for f in files if str(year) in os.path.basename(f)]
        
        if month is not None:
            # Format month as 2 digits (e.g., "01" for January)
            month_str = f"{month:02d}"
            files = [f for f in files if str(year) + month_str in os.path.basename(f)]
    
    logger.info(f"Found {len(files)} GeoTIFF files for {variable}" + 
              (f" in {year}" if year else "") + 
              (f" month {month}" if month else ""))
    
    return files

def extract_date_from_filename(filename):
    """
    Extract year and month from a PRISM filename.
    
    Args:
        filename (str): PRISM filename
        
    Returns:
        tuple: (year, month) or (None, None) if date cannot be extracted
    """
    # For sample files
    if "_sample.csv" in filename:
        match = re.search(r'(\d{4})(\d{2})_sample', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
    
    # For actual PRISM files
    # Example: PRISM_tmin_stable_4kmM3_201701_bil.bil
    match = re.search(r'(\d{4})(\d{2})_', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None

def read_geotiff_data(file_path):
    """
    Read data from a GeoTIFF file.
    
    Args:
        file_path (str): Path to GeoTIFF file
        
    Returns:
        tuple: (data_array, metadata) or (None, None) if file cannot be read
    """
    # Check if this is a sample CSV file
    if file_path.endswith("_sample.csv"):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Read sample data from {file_path} with {len(df)} points")
            return df, {'is_sample': True}
        except Exception as e:
            logger.error(f"Error reading sample data: {e}")
            return None, None
    
    # Read actual GeoTIFF file
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            metadata = src.meta
            logger.info(f"Read GeoTIFF data from {file_path} with shape {data.shape}")
            return data, metadata
    except Exception as e:
        logger.error(f"Error reading GeoTIFF file {file_path}: {e}")
        return None, None

def aggregate_by_state(data, metadata, state_boundaries, is_sample=False):
    """
    Aggregate GeoTIFF data by state.
    
    Args:
        data: GeoTIFF data array or DataFrame for sample data
        metadata: GeoTIFF metadata
        state_boundaries (geopandas.GeoDataFrame): State boundary data
        is_sample (bool): Whether the data is sample data
        
    Returns:
        pd.DataFrame: Aggregated state-level data
    """
    if is_sample or (isinstance(metadata, dict) and metadata.get('is_sample')):
        # For sample data, aggregate by state
        if isinstance(data, pd.DataFrame):
            # Group by state and calculate mean
            state_data = data.groupby('state_code')['value'].agg(['mean', 'min', 'max', 'std']).reset_index()
            state_data = state_data.rename(columns={
                'mean': 'mean_value',
                'min': 'min_value',
                'max': 'max_value',
                'std': 'std_value'
            })
            logger.info(f"Aggregated sample data to {len(state_data)} states")
            return state_data
        else:
            logger.error("Sample data provided but not in DataFrame format")
            return pd.DataFrame()
    
    # For actual GeoTIFF data
    results = []
    
    try:
        for idx, state in state_boundaries.iterrows():
            # Get the state geometry
            geom = [mapping(state.geometry)]
            
            # Extract data within the state boundary
            with rasterio.open(file_path) as src:
                out_image, out_transform = mask(src, geom, crop=True)
                
                # Calculate statistics
                state_data = out_image[0]
                valid_data = state_data[state_data != src.nodata]
                
                if len(valid_data) > 0:
                    results.append({
                        'state_code': state['STATE_CODE'],
                        'mean_value': float(np.mean(valid_data)),
                        'min_value': float(np.min(valid_data)),
                        'max_value': float(np.max(valid_data)),
                        'std_value': float(np.std(valid_data))
                    })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        logger.error(f"Error aggregating data by state: {e}")
        return pd.DataFrame()

def process_variable_for_period(variable, year=None, month=None, state_boundaries=None):
    """
    Process all GeoTIFF files for a specific variable and time period.
    
    Args:
        variable (str): Climate variable name
        year (int, optional): Year to process
        month (int, optional): Month to process
        state_boundaries (geopandas.GeoDataFrame, optional): State boundary data
        
    Returns:
        pd.DataFrame: Processed state-level data
    """
    # Load state boundaries if not provided
    if state_boundaries is None:
        state_boundaries = load_state_boundaries()
    
    # Find relevant files
    files = find_geotiff_files(variable, year, month)
    
    if not files:
        logger.warning(f"No files found for {variable} in {year or 'all years'}, month {month or 'all months'}")
        return pd.DataFrame()
    
    # Process each file and combine results
    all_results = []
    
    for file_path in files:
        # Extract date from filename
        file_year, file_month = extract_date_from_filename(file_path)
        
        if file_year is None or file_month is None:
            logger.warning(f"Could not extract date from filename: {file_path}")
            continue
        
        # Read the data
        data, metadata = read_geotiff_data(file_path)
        
        if data is None:
            continue
        
        # Aggregate by state
        is_sample = file_path.endswith("_sample.csv")
        state_data = aggregate_by_state(data, metadata, state_boundaries, is_sample)
        
        # Add date information
        state_data['year'] = file_year
        state_data['month'] = file_month
        state_data['variable'] = variable
        
        all_results.append(state_data)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Processed {len(combined_results)} state-month records for {variable}")
        return combined_results
    else:
        logger.warning(f"No results to combine for {variable}")
        return pd.DataFrame()

def process_all_variables_for_period(variables, year=None, month=None):
    """
    Process multiple variables for a time period.
    
    Args:
        variables (list): List of climate variables
        year (int, optional): Year to process
        month (int, optional): Month to process
        
    Returns:
        pd.DataFrame: Combined processed data
    """
    state_boundaries = load_state_boundaries()
    all_results = []
    
    for variable in variables:
        variable_data = process_variable_for_period(variable, year, month, state_boundaries)
        all_results.append(variable_data)
    
    # Combine all variables
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Processed {len(combined_results)} records across {len(variables)} variables")
        
        # Save to interim directory
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(INTERIM_DATA_DIR, f"prism_state_data_{timestamp}.csv")
        combined_results.to_csv(output_file, index=False)
        logger.info(f"Saved interim data to {output_file}")
        
        return combined_results
    else:
        logger.warning("No data to combine")
        return pd.DataFrame()

def calculate_growing_season_statistics(data, year=None):
    """
    Calculate growing season statistics for each crop and state.
    
    Args:
        data (pd.DataFrame): Processed state-level climate data
        year (int, optional): Year to filter by
        
    Returns:
        pd.DataFrame: Growing season statistics
    """
    # Filter by year if provided
    if year is not None:
        data = data[data['year'] == year]
    
    # List to store growing season statistics
    growing_season_stats = []
    
    # Process each state
    for state_code in data['state_code'].unique():
        state_data = data[data['state_code'] == state_code]
        
        # Process each crop's growing season
        for crop, season in GROWING_SEASONS.items():
            start_month = season['start_month']
            end_month = season['end_month']
            
            # Handle seasons that span across years (e.g., winter wheat)
            if start_month > end_month:
                # For data in the first year of the growing season
                months_first_year = list(range(start_month, 13))
                # For data in the second year of the growing season
                months_second_year = list(range(1, end_month + 1))
                
                # We need both years of data
                if year is not None:
                    logger.warning(f"Cannot calculate complete growing season for {crop} in year {year} without data from {year+1}")
                    # Just use the months from the specified year
                    growing_season_months = months_first_year
                else:
                    growing_season_months = months_first_year + months_second_year
            else:
                growing_season_months = list(range(start_month, end_month + 1))
            
            # Filter data to growing season months
            season_data = state_data[state_data['month'].isin(growing_season_months)]
            
            # Calculate statistics by variable
            for variable in season_data['variable'].unique():
                var_data = season_data[season_data['variable'] == variable]
                
                if var_data.empty:
                    continue
                
                # Calculate statistics
                if variable == 'ppt':
                    # For precipitation, sum over the growing season
                    total = var_data['mean_value'].sum()
                    metric_name = f"growing_season_{variable}_total"
                    
                    growing_season_stats.append({
                        'state_code': state_code,
                        'year': var_data['year'].iloc[0] if year is None else year,
                        'crop': crop,
                        'metric': metric_name,
                        'value': total
                    })
                else:
                    # For temperature, calculate mean and extremes
                    mean_value = var_data['mean_value'].mean()
                    min_value = var_data['min_value'].min()
                    max_value = var_data['max_value'].max()
                    
                    # Add to results
                    growing_season_stats.append({
                        'state_code': state_code,
                        'year': var_data['year'].iloc[0] if year is None else year,
                        'crop': crop,
                        'metric': f"growing_season_{variable}_mean",
                        'value': mean_value
                    })
                    
                    growing_season_stats.append({
                        'state_code': state_code,
                        'year': var_data['year'].iloc[0] if year is None else year,
                        'crop': crop,
                        'metric': f"growing_season_{variable}_min",
                        'value': min_value
                    })
                    
                    growing_season_stats.append({
                        'state_code': state_code,
                        'year': var_data['year'].iloc[0] if year is None else year,
                        'crop': crop,
                        'metric': f"growing_season_{variable}_max",
                        'value': max_value
                    })
            
            # Calculate growing degree days (if temperature data is available)
            if 'tmean' in season_data['variable'].values or ('tmin' in season_data['variable'].values and 'tmax' in season_data['variable'].values):
                # If mean temperature is available
                if 'tmean' in season_data['variable'].values:
                    tmean_data = season_data[season_data['variable'] == 'tmean']
                    temps = tmean_data['mean_value'].values
                # Otherwise, calculate from min and max
                else:
                    tmin_data = season_data[season_data['variable'] == 'tmin']
                    tmax_data = season_data[season_data['variable'] == 'tmax']
                    
                    # Match months for min and max temp
                    tmin_df = tmin_data[['month', 'mean_value']].rename(columns={'mean_value': 'tmin'})
                    tmax_df = tmax_data[['month', 'mean_value']].rename(columns={'mean_value': 'tmax'})
                    temp_df = pd.merge(tmin_df, tmax_df, on='month')
                    
                    # Calculate mean temperature
                    temp_df['tmean'] = (temp_df['tmin'] + temp_df['tmax']) / 2
                    temps = temp_df['tmean'].values
                
                # Calculate GDD with base 10°C
                gdd_base10 = sum(max(0, t - 10) for t in temps if not np.isnan(t))
                
                growing_season_stats.append({
                    'state_code': state_code,
                    'year': season_data['year'].iloc[0] if year is None else year,
                    'crop': crop,
                    'metric': 'growing_degree_days_base10',
                    'value': gdd_base10
                })
    
    # Convert to DataFrame
    gs_df = pd.DataFrame(growing_season_stats)
    
    if not gs_df.empty:
        # Pivot to wide format
        gs_wide = gs_df.pivot_table(
            index=['state_code', 'year', 'crop'],
            columns='metric',
            values='value'
        ).reset_index()
        
        logger.info(f"Calculated growing season statistics for {len(gs_wide)} state-year-crop combinations")
        
        # Save to processed directory
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(PROCESSED_DATA_DIR, f"prism_growing_season_stats_{timestamp}.csv")
        gs_wide.to_csv(output_file, index=False)
        logger.info(f"Saved growing season statistics to {output_file}")
        
        return gs_wide
    else:
        logger.warning("No growing season statistics to calculate")
        return pd.DataFrame()

def process_data_for_census_years(variables):
    """
    Process climate data for USDA Census years.
    
    Args:
        variables (list): List of climate variables
        
    Returns:
        dict: Dictionary with processed data for each census year
    """
    results = {}
    
    for year in CENSUS_YEARS:
        logger.info(f"Processing climate data for census year {year}")
        
        # Process data for all months in the year
        year_data = process_all_variables_for_period(variables, year)
        
        if not year_data.empty:
            # Calculate growing season statistics
            gs_stats = calculate_growing_season_statistics(year_data, year)
            
            results[year] = {
                'monthly_data': year_data,
                'growing_season_stats': gs_stats
            }
    
    return results

def merge_with_usda_data(climate_data, usda_data_path):
    """
    Merge climate data with USDA crop yield data.
    
    Args:
        climate_data (dict): Dictionary with processed climate data by year
        usda_data_path (str): Path to USDA crop data CSV
        
    Returns:
        pd.DataFrame: Merged dataset with climate and yield data
    """
    try:
        # Load USDA data
        usda_df = pd.read_csv(usda_data_path)
        logger.info(f"Loaded USDA data with {len(usda_df)} records")
        
        # List to store merged data
        merged_data = []
        
        # Process each crop type
        for crop in usda_df['CROP'].unique():
            crop_data = usda_df[usda_df['CROP'] == crop]
            
            for _, row in crop_data.iterrows():
                year = row['YEAR']
                
                # Skip if we don't have climate data for this year
                if year not in climate_data:
                    continue
                
                # Get growing season stats for this crop
                gs_stats = climate_data[year]['growing_season_stats']
                crop_gs_stats = gs_stats[gs_stats['crop'] == crop]
                
                # For national-level USDA data, average across all agricultural states
                national_climate = crop_gs_stats.drop(columns=['state_code', 'crop']).groupby('year').mean().reset_index()
                
                if national_climate.empty:
                    continue
                
                # Combine with yield data
                merged_row = row.to_dict()
                
                # Add climate variables
                for col in national_climate.columns:
                    if col != 'year':
                        merged_row[col] = national_climate[col].iloc[0]
                
                merged_data.append(merged_row)
        
        # Convert to DataFrame
        merged_df = pd.DataFrame(merged_data)
        
        if not merged_df.empty:
            # Save to processed directory
            timestamp = datetime.now().strftime("%Y%m%d")
            output_file = os.path.join(PROCESSED_DATA_DIR, f"crop_climate_merged_{timestamp}.csv")
            merged_df.to_csv(output_file, index=False)
            logger.info(f"Saved merged crop-climate data to {output_file} with {len(merged_df)} records")
            
            return merged_df
        else:
            logger.warning("No data to merge")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error merging with USDA data: {e}")
        return pd.DataFrame()

def main():
    """Main execution function."""
    logger.info("Starting PRISM climate data processing...")
    
    ensure_directories()
    
    # Define variables to process
    variables = ['ppt', 'tmin', 'tmax', 'tmean']
    
    # Process data for a 2017 (most recent census year) as a test
    logger.info("Processing data for 2017 as a test")
    test_data = process_all_variables_for_period(variables, 2017)
    
    if not test_data.empty:
        # Calculate growing season statistics
        gs_stats = calculate_growing_season_statistics(test_data, 2017)
        
        # Sample path to USDA data (replace with actual path)
        usda_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "data", "sample", "usda_crop_data_sample.csv")
        
        # Merge with USDA data if available
        if os.path.exists(usda_data_path):
            logger.info("Merging with USDA data")
            merged_data = merge_with_usda_data({2017: {'growing_season_stats': gs_stats}}, usda_data_path)
    
    # Uncomment the below line to process data for all census years
    # census_data = process_data_for_census_years(variables)
    
    logger.info("PRISM climate data processing complete.")

if __name__ == "__main__":
    main()