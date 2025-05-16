#!/usr/bin/env python3
"""
Script to download PRISM climate data.
Retrieves temperature, precipitation, and growing degree days data from PRISM Climate Group's web service.
"""

import os
import requests
import zipfile
import io
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PRISM_BASE_URL = "https://services.nacse.org/prism/data/get"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "prism")
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim", "prism")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "prism")

# PRISM data variables
PRISM_VARIABLES = {
    'ppt': 'Precipitation (mm)',
    'tmin': 'Minimum Temperature (°C)',
    'tmax': 'Maximum Temperature (°C)',
    'tmean': 'Mean Temperature (°C)',
    'tdmean': 'Dew Point Temperature (°C)',
    'vpdmin': 'Vapor Pressure Deficit Minimum (hPa)',
    'vpdmax': 'Vapor Pressure Deficit Maximum (hPa)'
}

# Normals currently available (1991-2020)
NORMALS_YEAR_RANGE = "1991-2020"

# Agricultural states (major crop-producing states)
AGRICULTURAL_STATES = [
    'IL', 'IA', 'NE', 'MN', 'IN', 'KS', 'SD', 'OH', 'ND', 'MO',  # Corn Belt & Northern Plains
    'AR', 'MS', 'LA', 'TX', 'AL', 'GA', 'SC', 'NC', 'TN',        # Southern states
    'CA', 'WA', 'OR', 'ID', 'CO', 'MT', 'WY', 'UT'               # Western states
]

# USDA Census years
CENSUS_YEARS = [1997, 2002, 2007, 2012, 2017]

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_prism_data(variable, date, resolution="4km", region="us", stable=True):
    """
    Download PRISM climate data for a specific variable, date, and resolution.
    
    Args:
        variable (str): Climate variable to download (e.g., 'ppt', 'tmin', 'tmax')
        date (str): Date in format 'YYYYMM' for monthly or 'YYYYMMDD' for daily
        resolution (str): Spatial resolution ('4km' or '800m')
        region (str): Geographic region ('us' for Continental US)
        stable (bool): Whether to request the stable version of the data
        
    Returns:
        str: Path to downloaded and extracted data directory, or None if download failed
    """
    # Construct URL
    stability = "stable" if stable else "provisional"
    url = f"{PRISM_BASE_URL}/{region}/{resolution}/{variable}/{date}?mode={stability}"
    
    logger.info(f"Downloading PRISM data: {variable} for {date} at {resolution} resolution")
    
    try:
        # Download the data
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory for extracted files
        output_dir = os.path.join(RAW_DATA_DIR, f"{variable}_{date}")
        if os.path.exists(output_dir):
            logger.info(f"Data already exists at {output_dir}, skipping download")
            return output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract zip file contents
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info(f"Data successfully downloaded and extracted to {output_dir}")
        return output_dir
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        return None
    except zipfile.BadZipFile:
        logger.error(f"Received invalid zip file for {variable} {date}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def download_monthly_data_for_year(year, variables=None):
    """
    Download monthly PRISM data for an entire year.
    
    Args:
        year (int): Year to download data for
        variables (list): List of variables to download, or None for all variables
        
    Returns:
        dict: Dictionary with variables as keys and lists of downloaded directories as values
    """
    if variables is None:
        variables = list(PRISM_VARIABLES.keys())
    
    results = {var: [] for var in variables}
    
    for var in variables:
        for month in range(1, 13):
            date = f"{year}{month:02d}"
            output_dir = download_prism_data(var, date)
            
            if output_dir:
                results[var].append(output_dir)
            
            # Be kind to the server
            time.sleep(1)
    
    return results

def download_normals(variables=None, period="monthly", resolution="4km"):
    """
    Download PRISM climate normals.
    
    Args:
        variables (list): List of variables to download, or None for all variables
        period (str): Time period for normals ('monthly', 'annual', etc.)
        resolution (str): Spatial resolution ('4km' or '800m')
        
    Returns:
        dict: Dictionary with variables as keys and lists of downloaded directories as values
    """
    if variables is None:
        variables = list(PRISM_VARIABLES.keys())
    
    results = {var: [] for var in variables}
    
    for var in variables:
        if period == "monthly":
            for month in range(1, 13):
                # Normals use a specific format
                date = f"{NORMALS_YEAR_RANGE}/monthly/{month:02d}"
                output_dir = download_prism_data(var, date, resolution=resolution)
                
                if output_dir:
                    results[var].append(output_dir)
                
                # Be kind to the server
                time.sleep(1)
        elif period == "annual":
            date = f"{NORMALS_YEAR_RANGE}/annual"
            output_dir = download_prism_data(var, date, resolution=resolution)
            
            if output_dir:
                results[var].append(output_dir)
        
    return results

def download_data_for_census_years(variables=None):
    """
    Download monthly data for USDA Census years.
    
    Args:
        variables (list): List of variables to download, or None for all variables
        
    Returns:
        dict: Dictionary with years as keys and downloaded data info as values
    """
    if variables is None:
        variables = list(PRISM_VARIABLES.keys())
    
    results = {}
    
    for year in CENSUS_YEARS:
        logger.info(f"Downloading data for census year {year}")
        results[year] = download_monthly_data_for_year(year, variables)
    
    return results

def create_sample_data():
    """
    Create sample PRISM data files for testing and development.
    Uses predefined sample values instead of actual API calls.
    """
    ensure_directories()
    
    # Create a sample directory structure that mimics what would be downloaded
    sample_dir = os.path.join(RAW_DATA_DIR, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample CSV data (this won't be in the real GeoTIFF format, but is for testing)
    variables = ['ppt', 'tmin', 'tmax', 'tmean']
    years = [2017]  # Just use the most recent census year for the sample
    
    for var in variables:
        var_dir = os.path.join(sample_dir, var)
        os.makedirs(var_dir, exist_ok=True)
        
        for year in years:
            for month in range(1, 13):
                # Create sample CSV data
                data = []
                
                # Generate sample grid data for 5 states (normally this would be raster data)
                for state, state_code in [
                    ('Illinois', 'IL'), 
                    ('Iowa', 'IA'), 
                    ('Nebraska', 'NE'),
                    ('Minnesota', 'MN'),
                    ('Indiana', 'IN')
                ]:
                    # Generate 10 sample points for each state
                    for i in range(10):
                        if var == 'ppt':
                            # Generate seasonal precipitation pattern
                            # Higher in spring/summer, lower in fall/winter
                            monthly_factors = [0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.2, 1.1, 0.9, 0.7, 0.6, 0.5]
                            value = 50 + 30 * monthly_factors[month-1] + np.random.normal(0, 10)
                        elif var == 'tmin':
                            # Generate seasonal temperature pattern
                            # Higher in summer, lower in winter
                            monthly_factors = [-15, -12, -5, 2, 8, 13, 15, 14, 10, 3, -5, -10]
                            value = monthly_factors[month-1] + np.random.normal(0, 2)
                        elif var == 'tmax':
                            # Generate seasonal temperature pattern
                            # Higher in summer, lower in winter
                            monthly_factors = [0, 2, 10, 18, 25, 30, 32, 31, 25, 18, 8, 2]
                            value = monthly_factors[month-1] + np.random.normal(0, 2)
                        elif var == 'tmean':
                            # Generate seasonal temperature pattern
                            # Higher in summer, lower in winter
                            monthly_factors = [-8, -5, 3, 10, 16, 22, 24, 22, 18, 10, 1, -5]
                            value = monthly_factors[month-1] + np.random.normal(0, 2)
                        
                        data.append({
                            'state': state,
                            'state_code': state_code,
                            'grid_id': f"{state_code}_{i+1}",
                            'latitude': 40 + np.random.normal(0, 2),
                            'longitude': -90 + np.random.normal(0, 5),
                            'value': value
                        })
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame(data)
                output_file = os.path.join(var_dir, f"{var}_{year}{month:02d}_sample.csv")
                df.to_csv(output_file, index=False)
                
                logger.info(f"Created sample data file: {output_file}")
    
    # Create processed sample file that combines monthly data by state
    processed_data = []
    for year in years:
        for var in variables:
            for state_code in ['IL', 'IA', 'NE', 'MN', 'IN']:
                # Create monthly values
                for month in range(1, 13):
                    if var == 'ppt':
                        # Generate seasonal precipitation pattern
                        monthly_factors = [0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.2, 1.1, 0.9, 0.7, 0.6, 0.5]
                        value = 50 + 30 * monthly_factors[month-1] + np.random.normal(0, 5)
                    elif var == 'tmin':
                        # Generate seasonal temperature pattern
                        monthly_factors = [-15, -12, -5, 2, 8, 13, 15, 14, 10, 3, -5, -10]
                        value = monthly_factors[month-1] + np.random.normal(0, 1)
                    elif var == 'tmax':
                        # Generate seasonal temperature pattern
                        monthly_factors = [0, 2, 10, 18, 25, 30, 32, 31, 25, 18, 8, 2]
                        value = monthly_factors[month-1] + np.random.normal(0, 1)
                    elif var == 'tmean':
                        # Generate seasonal temperature pattern
                        monthly_factors = [-8, -5, 3, 10, 16, 22, 24, 22, 18, 10, 1, -5]
                        value = monthly_factors[month-1] + np.random.normal(0, 1)
                    
                    processed_data.append({
                        'year': year,
                        'month': month,
                        'state_code': state_code,
                        'variable': var,
                        'value': value
                    })
    
    # Create DataFrame and save to CSV
    processed_df = pd.DataFrame(processed_data)
    processed_file = os.path.join(PROCESSED_DATA_DIR, "prism_state_monthly_sample.csv")
    processed_df.to_csv(processed_file, index=False)
    
    logger.info(f"Created processed sample data file: {processed_file}")
    
    # Create growing season statistics
    growing_season_data = []
    
    for year in years:
        for state_code in ['IL', 'IA', 'NE', 'MN', 'IN']:
            # Growing season months (April-September)
            growing_months = list(range(4, 10))
            
            # Calculate statistics
            gs_tmin = np.mean([row['value'] for row in processed_data 
                              if row['year'] == year and row['state_code'] == state_code 
                              and row['variable'] == 'tmin' and row['month'] in growing_months])
            
            gs_tmax = np.mean([row['value'] for row in processed_data 
                              if row['year'] == year and row['state_code'] == state_code 
                              and row['variable'] == 'tmax' and row['month'] in growing_months])
            
            gs_tavg = (gs_tmin + gs_tmax) / 2
            
            gs_ppt = np.sum([row['value'] for row in processed_data 
                            if row['year'] == year and row['state_code'] == state_code 
                            and row['variable'] == 'ppt' and row['month'] in growing_months])
            
            # Calculate growing degree days (base 10°C)
            gdd_base10 = sum(max(0, (min(30, t) - 10)) for t in 
                             [row['value'] for row in processed_data 
                              if row['year'] == year and row['state_code'] == state_code 
                              and row['variable'] == 'tmean' and row['month'] in growing_months])
            
            # Estimate frost dates and growing season length
            frost_free_days = 150 + np.random.randint(-10, 11)  # Random value around 150 days
            
            growing_season_data.append({
                'year': year,
                'state_code': state_code,
                'growing_season_tmin': gs_tmin,
                'growing_season_tmax': gs_tmax,
                'growing_season_tavg': gs_tavg,
                'growing_season_ppt': gs_ppt,
                'growing_degree_days_base10': gdd_base10,
                'frost_free_days': frost_free_days
            })
    
    # Create DataFrame and save to CSV
    growing_season_df = pd.DataFrame(growing_season_data)
    growing_season_file = os.path.join(PROCESSED_DATA_DIR, "prism_growing_season_sample.csv")
    growing_season_df.to_csv(growing_season_file, index=False)
    
    logger.info(f"Created growing season sample data file: {growing_season_file}")

def main():
    """Main execution function."""
    logger.info("Starting PRISM climate data download...")
    
    ensure_directories()
    
    # Create sample data for testing
    logger.info("Creating sample data files...")
    create_sample_data()
    
    # Uncomment the below lines to use actual API calls
    # Specify variables to download
    # variables = ['ppt', 'tmin', 'tmax']
    
    # # Download climate normals
    # logger.info("Downloading monthly climate normals...")
    # normals_results = download_normals(variables=variables)
    
    # # Download data for USDA Census years
    # logger.info("Downloading data for USDA Census years...")
    # census_results = download_data_for_census_years(variables=variables)
    
    logger.info("PRISM data download and sample creation complete.")

if __name__ == "__main__":
    main()