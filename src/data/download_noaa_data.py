#!/usr/bin/env python3
"""
Script to download NOAA Climate Normals data.
Retrieves temperature, precipitation, and freeze data from NOAA's Climate Data Online.
"""

import os
import time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
NOAA_API_TOKEN = os.getenv("NOAA_API_TOKEN", "")  # Get API token from environment variable
NOAA_API_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
INTERIM_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim")

# Climate variables to download
CLIMATE_VARIABLES = {
    'TMIN': 'Minimum temperature',
    'TMAX': 'Maximum temperature',
    'PRCP': 'Precipitation',
    'TAVG': 'Average temperature'
}

# Agricultural states (major crop-producing states)
AGRICULTURAL_STATES = [
    'IL', 'IA', 'NE', 'MN', 'IN', 'KS', 'SD', 'OH', 'ND', 'MO',  # Corn Belt & Northern Plains
    'AR', 'MS', 'LA', 'TX', 'AL', 'GA', 'SC', 'NC', 'TN',        # Southern states
    'CA', 'WA', 'OR', 'ID', 'CO', 'MT', 'WY', 'UT'               # Western states
]

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

def fetch_noaa_data(dataset_id, data_type_id, location_id, start_date, end_date):
    """
    Fetch climate data from NOAA API.
    
    Args:
        dataset_id (str): NOAA dataset identifier
        data_type_id (str): Climate variable identifier
        location_id (str): Location identifier
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        
    Returns:
        dict: JSON response from the API
    """
    if not NOAA_API_TOKEN:
        raise ValueError("NOAA API token is required. Please set NOAA_API_TOKEN environment variable.")
    
    headers = {
        'token': NOAA_API_TOKEN
    }
    
    params = {
        'datasetid': dataset_id,
        'datatypeid': data_type_id,
        'locationid': location_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'units': 'standard'
    }
    
    response = requests.get(f"{NOAA_API_BASE_URL}data", headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None

def fetch_stations(state):
    """
    Fetch weather stations for a particular state.
    
    Args:
        state (str): Two-letter state code
        
    Returns:
        list: List of station identifiers
    """
    if not NOAA_API_TOKEN:
        raise ValueError("NOAA API token is required. Please set NOAA_API_TOKEN environment variable.")
    
    headers = {
        'token': NOAA_API_TOKEN
    }
    
    params = {
        'locationid': f"FIPS:{state}",
        'limit': 1000,
        'datatypeid': 'NORMAL_DLY'  # For climate normals
    }
    
    response = requests.get(f"{NOAA_API_BASE_URL}stations", headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return [station['id'] for station in data.get('results', [])]
    else:
        print(f"Error fetching stations for {state}: {response.status_code}")
        return []

def download_climate_normals():
    """
    Download daily climate normals (1981-2010) for all specified states.
    
    Returns:
        pd.DataFrame: Combined dataframe with climate normals
    """
    ensure_directories()
    
    climate_data = []
    
    for state in AGRICULTURAL_STATES:
        print(f"Fetching stations for {state}...")
        stations = fetch_stations(state)
        
        if not stations:
            print(f"No stations found for {state}, skipping.")
            continue
        
        # Use a subset of stations to avoid overwhelming the API
        selected_stations = stations[:5]  # Adjust this number based on API limits
        
        for station in selected_stations:
            for variable, description in CLIMATE_VARIABLES.items():
                print(f"Fetching {description} normals for station {station}...")
                
                # For climate normals (1981-2010)
                data = fetch_noaa_data(
                    dataset_id='NORMAL_DLY',
                    data_type_id=f'DLY-{variable}-NORMAL',
                    location_id=station,
                    start_date='2010-01-01',  # Climate normals are indexed by a nominal year (2010)
                    end_date='2010-12-31'
                )
                
                if data and 'results' in data:
                    for result in data['results']:
                        try:
                            climate_data.append({
                                'STATION': result.get('station', ''),
                                'DATE': result.get('date', ''),
                                'VARIABLE': variable,
                                'VALUE': float(result.get('value', 0)),
                                'ATTRIBUTES': result.get('attributes', '')
                            })
                        except (ValueError, KeyError) as e:
                            print(f"Error parsing data point: {e}")
                            continue
                
                # Be kind to the API with a delay between requests
                time.sleep(1)
            
            print(f"Completed data fetch for station {station}")
    
    # Convert to DataFrame
    if climate_data:
        df = pd.DataFrame(climate_data)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d")
        raw_file_path = os.path.join(RAW_DATA_DIR, f"noaa_climate_normals_raw_{timestamp}.csv")
        df.to_csv(raw_file_path, index=False)
        print(f"Raw climate normals saved to {raw_file_path}")
        
        return df
    else:
        print("No climate data was retrieved.")
        return pd.DataFrame()

def process_climate_data(df):
    """
    Process raw climate data to create a clean dataset for analysis.
    
    Args:
        df (pd.DataFrame): Raw climate data
        
    Returns:
        pd.DataFrame: Processed climate data
    """
    if df.empty:
        return df
    
    # Add additional station information
    station_info = get_station_metadata(df['STATION'].unique())
    
    # Merge with station information
    if station_info is not None and not station_info.empty:
        df = df.merge(station_info, on='STATION', how='left')
    
    # Convert date string to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Create month and day columns for aggregation
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    
    # Pivot the data for easier analysis
    pivot_df = df.pivot_table(
        index=['STATION', 'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE'],
        columns='VARIABLE',
        values='VALUE'
    ).reset_index()
    
    # Rename columns
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={
        'TMIN': 'DLY-TMIN-NORMAL',
        'TMAX': 'DLY-TMAX-NORMAL',
        'PRCP': 'DLY-PRCP-NORMAL',
        'TAVG': 'DLY-TAVG-NORMAL'
    })
    
    # Calculate monthly and seasonal aggregates
    monthly_df = calculate_monthly_aggregates(df)
    seasonal_df = calculate_seasonal_aggregates(monthly_df)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save daily data
    if not pivot_df.empty:
        daily_file_path = os.path.join(INTERIM_DATA_DIR, f"noaa_daily_normals_{timestamp}.csv")
        pivot_df.to_csv(daily_file_path, index=False)
        print(f"Processed daily normals saved to {daily_file_path}")
    
    # Save monthly data
    if not monthly_df.empty:
        monthly_file_path = os.path.join(INTERIM_DATA_DIR, f"noaa_monthly_normals_{timestamp}.csv")
        monthly_df.to_csv(monthly_file_path, index=False)
        print(f"Processed monthly normals saved to {monthly_file_path}")
    
    # Save seasonal data
    if not seasonal_df.empty:
        seasonal_file_path = os.path.join(INTERIM_DATA_DIR, f"noaa_seasonal_normals_{timestamp}.csv")
        seasonal_df.to_csv(seasonal_file_path, index=False)
        print(f"Processed seasonal normals saved to {seasonal_file_path}")
    
    return pivot_df

def get_station_metadata(station_ids):
    """
    Fetch metadata for a list of stations.
    
    Args:
        station_ids (list): List of station identifiers
        
    Returns:
        pd.DataFrame: DataFrame with station metadata
    """
    if not NOAA_API_TOKEN:
        raise ValueError("NOAA API token is required. Please set NOAA_API_TOKEN environment variable.")
    
    headers = {
        'token': NOAA_API_TOKEN
    }
    
    station_data = []
    
    for station_id in station_ids:
        response = requests.get(f"{NOAA_API_BASE_URL}stations/{station_id}", headers=headers)
        
        if response.status_code == 200:
            station = response.json()
            station_data.append({
                'STATION': station.get('id', ''),
                'STATION_NAME': station.get('name', ''),
                'LATITUDE': station.get('latitude', 0),
                'LONGITUDE': station.get('longitude', 0),
                'ELEVATION': station.get('elevation', 0)
            })
        else:
            print(f"Error fetching metadata for station {station_id}: {response.status_code}")
        
        # Be kind to the API with a delay between requests
        time.sleep(0.5)
    
    return pd.DataFrame(station_data) if station_data else None

def calculate_monthly_aggregates(df):
    """
    Calculate monthly aggregates from daily climate data.
    
    Args:
        df (pd.DataFrame): Daily climate data
        
    Returns:
        pd.DataFrame: Monthly climate data aggregates
    """
    if df.empty:
        return pd.DataFrame()
    
    # Group by station and month
    monthly_agg = df.groupby(['STATION', 'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'MONTH', 'VARIABLE']).agg({
        'VALUE': [
            ('MEAN', 'mean'),
            ('MIN', 'min'),
            ('MAX', 'max'),
            ('SUM', 'sum')
        ]
    }).reset_index()
    
    # Flatten the column names
    monthly_agg.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in monthly_agg.columns]
    
    return monthly_agg

def calculate_seasonal_aggregates(monthly_df):
    """
    Calculate seasonal aggregates from monthly climate data.
    
    Args:
        monthly_df (pd.DataFrame): Monthly climate data
        
    Returns:
        pd.DataFrame: Seasonal climate data aggregates
    """
    if monthly_df.empty:
        return pd.DataFrame()
    
    # Define seasons (meteorological)
    seasons = {
        'WINTER': [12, 1, 2],
        'SPRING': [3, 4, 5],
        'SUMMER': [6, 7, 8],
        'FALL': [9, 10, 11]
    }
    
    # Create a copy to avoid modifying the original
    df = monthly_df.copy()
    
    # Add season column
    df['SEASON'] = df['MONTH'].apply(
        lambda m: next(season for season, months in seasons.items() if m in months)
    )
    
    # Group by station and season
    seasonal_agg = df.groupby(['STATION', 'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'SEASON', 'VARIABLE']).agg({
        'VALUE_MEAN': 'mean',
        'VALUE_MIN': 'min',
        'VALUE_MAX': 'max',
        'VALUE_SUM': 'sum'
    }).reset_index()
    
    return seasonal_agg

def create_sample_data():
    """
    Create sample data files for testing and development.
    Uses predefined sample values instead of actual API calls.
    """
    ensure_directories()
    
    # Sample USDA data
    usda_sample = pd.DataFrame({
        'YEAR': [1997, 2002, 2007, 2012, 2017] * 8,
        'CROP': ['CORN'] * 5 + ['SOYBEANS'] * 5 + ['WHEAT'] * 5 + ['COTTON'] * 5 + 
                ['RICE'] * 5 + ['BARLEY'] * 5 + ['SORGHUM'] * 5 + ['OATS'] * 5,
        'YIELD': [
            # Corn yields
            127.0, 129.3, 151.1, 123.4, 176.6,
            # Soybean yields
            38.9, 42.0, 41.7, 39.6, 49.1,
            # Wheat yields
            39.5, 35.0, 40.2, 46.2, 46.4,
            # Cotton yields
            673.0, 665.0, 879.0, 887.0, 905.0,
            # Rice yields
            5894.0, 6578.0, 7185.0, 7449.0, 7507.0,
            # Barley yields
            58.0, 55.0, 60.0, 67.0, 73.0,
            # Sorghum yields
            69.0, 50.0, 73.0, 50.0, 72.0,
            # Oats yields
            60.0, 56.0, 60.0, 61.0, 61.0
        ],
        'HARVESTED_ACRES': [
            # Corn acres
            72700000, 69330000, 86520000, 87375000, 82733000,
            # Soybean acres
            69110000, 72497000, 63915000, 76104000, 89522000,
            # Wheat acres
            62840000, 45518000, 51011000, 48759000, 37589000,
            # Cotton acres
            13180000, 12456000, 10489000, 9153000, 10941000,
            # Rice acres
            3130000, 3197000, 2758000, 2699000, 2463000,
            # Barley acres
            6706000, 4142000, 3521000, 3467000, 2566000,
            # Sorghum acres
            9158000, 6750000, 6769000, 4986000, 5049000,
            # Oats acres
            2844000, 1986000, 1509000, 1078000, 814000
        ],
        'PRODUCTION': [
            # Corn production
            9207000000, 8966800000, 13073900000, 10780300000, 14609400000,
            # Soybean production
            2689000000, 2756800000, 2677100000, 3015000000, 4392000000,
            # Wheat production
            2481000000, 1605900000, 2051000000, 2252000000, 1741000000,
            # Cotton production
            8700000, 8182000, 9221000, 8122000, 9903000,
            # Rice production
            184300000, 210300000, 198300000, 200000000, 185000000,
            # Barley production
            389000000, 227000000, 212000000, 233000000, 186000000,
            # Sorghum production
            631000000, 342000000, 492000000, 249000000, 364000000,
            # Oats production
            169000000, 116000000, 91000000, 64000000, 50000000
        ]
    })
    
    # Sample NOAA climate data
    noaa_sample = pd.DataFrame({
        'STATION': ['GHCND:USC00327027'] * 4,
        'STATION_NAME': ['PETERSBURG 2 N ND US'] * 4,
        'ELEVATION': [466.3] * 4,
        'LATITUDE': [48.0355] * 4,
        'LONGITUDE': [-98.01] * 4,
        'DATE': pd.to_datetime(['2010-01-01', '2010-01-02', '2010-07-01', '2010-07-02']),
        'DLY-TMIN-NORMAL': [-33, -35, 45, 47],
        'DLY-TMAX-NORMAL': [145, 144, 256, 258],
        'DLY-PRCP-NORMAL': [2, 4, 115, 98]
    })
    
    # Save sample data
    sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample")
    os.makedirs(sample_data_dir, exist_ok=True)
    
    usda_sample_path = os.path.join(sample_data_dir, "usda_crop_data_sample.csv")
    noaa_sample_path = os.path.join(sample_data_dir, "noaa_climate_normals_sample.csv")
    
    usda_sample.to_csv(usda_sample_path, index=False)
    noaa_sample.to_csv(noaa_sample_path, index=False)
    
    print(f"Sample USDA data saved to {usda_sample_path}")
    print(f"Sample NOAA data saved to {noaa_sample_path}")

def main():
    """Main execution function."""
    print("Starting NOAA Climate Normals download...")
    
    # Create sample data for testing (comment this out for actual API usage)
    print("Creating sample data files...")
    create_sample_data()
    
    # Uncomment the below lines to use actual API calls
    # climate_data = download_climate_normals()
    # if not climate_data.empty:
    #     processed_data = process_climate_data(climate_data)
    #     print("NOAA Climate Normals download and processing complete.")
    # else:
    #     print("No climate data was retrieved. Check your API token and internet connection.")
    
    print("NOAA data processing complete.")

if __name__ == "__main__":
    main()