#!/usr/bin/env python3
"""
Script to download USDA Census of Agriculture Data for field crops.
Uses USDA QuickStats API to fetch yield and production data for major field crops.
"""

import os
import json
import time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
USDA_API_KEY = os.getenv("USDA_API_KEY", "")  # Get API key from environment variable
USDA_API_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim")

# List of field crops to include
FIELD_CROPS = [
    'CORN', 
    'SOYBEANS',
    'WHEAT',
    'COTTON',
    'RICE',
    'BARLEY',
    'SORGHUM',
    'OATS'
]

# Years for the Census of Agriculture
CENSUS_YEARS = [1997, 2002, 2007, 2012, 2017]

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

def fetch_usda_data(crop, year):
    """
    Fetch data from USDA QuickStats API for a specific crop and year.
    
    Args:
        crop (str): Crop name (e.g., 'CORN')
        year (int): Census year
        
    Returns:
        dict: JSON response from the API
    """
    if not USDA_API_KEY:
        raise ValueError("USDA API key is required. Please set USDA_API_KEY environment variable.")
    
    params = {
        'key': USDA_API_KEY,
        'source_desc': 'CENSUS',
        'commodity_desc': crop,
        'year': year,
        'freq_desc': 'ANNUAL',
        'domain_desc': 'TOTAL',
        'agg_level_desc': 'NATIONAL',
        'format': 'JSON',
    }
    
    response = requests.get(USDA_API_BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {crop} in {year}: {response.status_code}")
        return None

def parse_usda_response(response_data):
    """
    Parse USDA API response and extract relevant fields.
    
    Args:
        response_data (dict): JSON response from the USDA API
        
    Returns:
        list: List of dictionaries with parsed data
    """
    if not response_data or 'data' not in response_data:
        return []
    
    parsed_data = []
    
    for item in response_data['data']:
        try:
            # Extract only the fields we need
            data_point = {
                'YEAR': int(item.get('year', 0)),
                'CROP': item.get('commodity_desc', ''),
                'CATEGORY': item.get('statisticcat_desc', ''),
                'VALUE': float(item.get('Value', 0)),
                'UNIT': item.get('unit_desc', ''),
            }
            parsed_data.append(data_point)
        except (ValueError, KeyError) as e:
            print(f"Error parsing data point: {e}")
            continue
    
    return parsed_data

def download_all_crop_data():
    """
    Download data for all specified crops and years.
    
    Returns:
        pd.DataFrame: Combined dataframe with all crop data
    """
    ensure_directories()
    
    all_data = []
    
    for crop in FIELD_CROPS:
        for year in CENSUS_YEARS:
            print(f"Fetching data for {crop} in {year}...")
            response_data = fetch_usda_data(crop, year)
            
            if response_data:
                parsed_data = parse_usda_response(response_data)
                all_data.extend(parsed_data)
            
            # Be kind to the API with a short delay between requests
            time.sleep(0.5)
    
    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d")
        raw_file_path = os.path.join(RAW_DATA_DIR, f"usda_crop_data_raw_{timestamp}.csv")
        df.to_csv(raw_file_path, index=False)
        print(f"Raw data saved to {raw_file_path}")
        
        return df
    else:
        print("No data was retrieved.")
        return pd.DataFrame()

def process_crop_data(df):
    """
    Process raw crop data to create a clean dataset for analysis.
    
    Args:
        df (pd.DataFrame): Raw crop data
        
    Returns:
        pd.DataFrame: Processed crop data with yield, harvested acres, and production
    """
    if df.empty:
        return df
    
    # Create a list to store processed records
    processed_data = []
    
    # Group by year and crop
    for (year, crop), group in df.groupby(['YEAR', 'CROP']):
        record = {'YEAR': year, 'CROP': crop}
        
        # Extract yield, harvested acres, and production data
        for _, row in group.iterrows():
            category = row['CATEGORY']
            value = row['VALUE']
            
            if 'YIELD' in category.upper():
                record['YIELD'] = value
            elif 'AREA HARVESTED' in category.upper():
                record['HARVESTED_ACRES'] = value
            elif 'PRODUCTION' in category.upper():
                record['PRODUCTION'] = value
        
        # Only add complete records
        if all(k in record for k in ['YIELD', 'HARVESTED_ACRES', 'PRODUCTION']):
            processed_data.append(record)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Save processed data
    if not processed_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d")
        processed_file_path = os.path.join(INTERIM_DATA_DIR, f"usda_crop_data_processed_{timestamp}.csv")
        processed_df.to_csv(processed_file_path, index=False)
        print(f"Processed data saved to {processed_file_path}")
    
    return processed_df

def main():
    """Main execution function."""
    print("Starting USDA data download...")
    
    # Download raw data
    raw_data = download_all_crop_data()
    
    # Process data if raw data was downloaded
    if not raw_data.empty:
        processed_data = process_crop_data(raw_data)
        print(f"Downloaded and processed data for {len(processed_data)} crop-year combinations.")
    
    print("USDA data download complete.")

if __name__ == "__main__":
    main()