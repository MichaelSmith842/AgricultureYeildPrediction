#!/usr/bin/env python3
"""
Spatial aggregation module for PRISM climate data.
Provides functions to aggregate gridded climate data to state and county levels,
with focus on agricultural regions.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box
import logging
from pathlib import Path

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

# Path to boundary files
BOUNDARIES_DIR = os.path.join(RAW_DATA_DIR, "boundaries")
STATE_BOUNDARIES_PATH = os.path.join(BOUNDARIES_DIR, "us_states.shp")
COUNTY_BOUNDARIES_PATH = os.path.join(BOUNDARIES_DIR, "us_counties.shp")
CROPLAND_MASK_PATH = os.path.join(BOUNDARIES_DIR, "cropland_mask.tif")

# Agricultural states (major crop-producing states)
AGRICULTURAL_STATES = [
    'IL', 'IA', 'NE', 'MN', 'IN', 'KS', 'SD', 'OH', 'ND', 'MO',  # Corn Belt & Northern Plains
    'AR', 'MS', 'LA', 'TX', 'AL', 'GA', 'SC', 'NC', 'TN',        # Southern states
    'CA', 'WA', 'OR', 'ID', 'CO', 'MT', 'WY', 'UT'               # Western states
]

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(BOUNDARIES_DIR, exist_ok=True)

def load_state_boundaries(ag_states_only=True):
    """
    Load US state boundaries from shapefile.
    
    Args:
        ag_states_only (bool): Whether to filter for agricultural states only
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing state boundaries
    """
    try:
        # Check if the shapefile exists
        if not os.path.exists(STATE_BOUNDARIES_PATH):
            logger.warning(f"State boundaries shapefile not found at {STATE_BOUNDARIES_PATH}")
            logger.warning("Creating sample state boundaries")
            return create_sample_state_boundaries(ag_states_only)
        
        # Load the shapefile
        state_boundaries = gpd.read_file(STATE_BOUNDARIES_PATH)
        
        # Filter for agricultural states if requested
        if ag_states_only:
            state_boundaries = state_boundaries[
                state_boundaries['STATE_CODE'].isin(AGRICULTURAL_STATES)
            ]
        
        logger.info(f"Loaded state boundaries with {len(state_boundaries)} states")
        return state_boundaries
    
    except Exception as e:
        logger.error(f"Error loading state boundaries: {e}")
        logger.warning("Creating sample state boundaries")
        return create_sample_state_boundaries(ag_states_only)

def create_sample_state_boundaries(ag_states_only=True):
    """
    Create a sample GeoDataFrame with simplified state boundaries for testing.
    
    Args:
        ag_states_only (bool): Whether to include only agricultural states
        
    Returns:
        geopandas.GeoDataFrame: Sample GeoDataFrame with state boundaries
    """
    from shapely.geometry import Polygon
    
    # Create simplified polygons for agricultural states
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
    
    # Add some non-agricultural states if needed
    if not ag_states_only:
        state_data.extend([
            {'STATE_CODE': 'NY', 'STATE_NAME': 'New York', 'geometry': Polygon([
                (-79.8, 40.5), (-73.9, 40.5), (-73.9, 45.0), (-79.8, 45.0)
            ])},
            {'STATE_CODE': 'ME', 'STATE_NAME': 'Maine', 'geometry': Polygon([
                (-71.1, 43.1), (-67.0, 43.1), (-67.0, 47.5), (-71.1, 47.5)
            ])}
        ])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(state_data, crs="EPSG:4326")
    logger.info(f"Created sample state boundaries with {len(gdf)} states")
    return gdf

def load_county_boundaries(state_codes=None):
    """
    Load US county boundaries from shapefile.
    
    Args:
        state_codes (list): List of state codes to filter by
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing county boundaries
    """
    try:
        # Check if the shapefile exists
        if not os.path.exists(COUNTY_BOUNDARIES_PATH):
            logger.warning(f"County boundaries shapefile not found at {COUNTY_BOUNDARIES_PATH}")
            logger.warning("Creating sample county boundaries")
            return create_sample_county_boundaries(state_codes)
        
        # Load the shapefile
        county_boundaries = gpd.read_file(COUNTY_BOUNDARIES_PATH)
        
        # Filter by state if requested
        if state_codes:
            county_boundaries = county_boundaries[
                county_boundaries['STATE_CODE'].isin(state_codes)
            ]
        
        logger.info(f"Loaded county boundaries with {len(county_boundaries)} counties")
        return county_boundaries
    
    except Exception as e:
        logger.error(f"Error loading county boundaries: {e}")
        logger.warning("Creating sample county boundaries")
        return create_sample_county_boundaries(state_codes)

def create_sample_county_boundaries(state_codes=None):
    """
    Create a sample GeoDataFrame with simplified county boundaries for testing.
    
    Args:
        state_codes (list): List of state codes to include
        
    Returns:
        geopandas.GeoDataFrame: Sample GeoDataFrame with county boundaries
    """
    from shapely.geometry import Polygon
    
    # Set default state codes if not provided
    if not state_codes:
        state_codes = ['IL', 'IA', 'NE', 'MN', 'IN']
    elif not isinstance(state_codes, list):
        state_codes = [state_codes]
    
    # Create sample counties for selected states
    county_data = []
    
    # Counties for Illinois
    if 'IL' in state_codes:
        county_data.extend([
            {'COUNTY_NAME': 'Champaign', 'STATE_CODE': 'IL', 'FIPS': '17019', 'geometry': Polygon([
                (-88.4, 39.9), (-87.9, 39.9), (-87.9, 40.4), (-88.4, 40.4)
            ])},
            {'COUNTY_NAME': 'McLean', 'STATE_CODE': 'IL', 'FIPS': '17113', 'geometry': Polygon([
                (-89.2, 40.3), (-88.6, 40.3), (-88.6, 40.8), (-89.2, 40.8)
            ])}
        ])
    
    # Counties for Iowa
    if 'IA' in state_codes:
        county_data.extend([
            {'COUNTY_NAME': 'Story', 'STATE_CODE': 'IA', 'FIPS': '19169', 'geometry': Polygon([
                (-93.8, 41.8), (-93.3, 41.8), (-93.3, 42.2), (-93.8, 42.2)
            ])},
            {'COUNTY_NAME': 'Polk', 'STATE_CODE': 'IA', 'FIPS': '19153', 'geometry': Polygon([
                (-93.8, 41.4), (-93.3, 41.4), (-93.3, 41.8), (-93.8, 41.8)
            ])}
        ])
    
    # Counties for Nebraska
    if 'NE' in state_codes:
        county_data.extend([
            {'COUNTY_NAME': 'Lancaster', 'STATE_CODE': 'NE', 'FIPS': '31109', 'geometry': Polygon([
                (-96.9, 40.6), (-96.5, 40.6), (-96.5, 41.0), (-96.9, 41.0)
            ])},
            {'COUNTY_NAME': 'Douglas', 'STATE_CODE': 'NE', 'FIPS': '31055', 'geometry': Polygon([
                (-96.4, 41.2), (-95.9, 41.2), (-95.9, 41.4), (-96.4, 41.4)
            ])}
        ])
    
    # Counties for Minnesota
    if 'MN' in state_codes:
        county_data.extend([
            {'COUNTY_NAME': 'Hennepin', 'STATE_CODE': 'MN', 'FIPS': '27053', 'geometry': Polygon([
                (-93.6, 44.9), (-93.2, 44.9), (-93.2, 45.2), (-93.6, 45.2)
            ])},
            {'COUNTY_NAME': 'Ramsey', 'STATE_CODE': 'MN', 'FIPS': '27123', 'geometry': Polygon([
                (-93.2, 44.9), (-92.9, 44.9), (-92.9, 45.1), (-93.2, 45.1)
            ])}
        ])
    
    # Counties for Indiana
    if 'IN' in state_codes:
        county_data.extend([
            {'COUNTY_NAME': 'Marion', 'STATE_CODE': 'IN', 'FIPS': '18097', 'geometry': Polygon([
                (-86.3, 39.6), (-85.9, 39.6), (-85.9, 40.0), (-86.3, 40.0)
            ])},
            {'COUNTY_NAME': 'Hamilton', 'STATE_CODE': 'IN', 'FIPS': '18057', 'geometry': Polygon([
                (-86.2, 40.0), (-85.8, 40.0), (-85.8, 40.2), (-86.2, 40.2)
            ])}
        ])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(county_data, crs="EPSG:4326")
    logger.info(f"Created sample county boundaries with {len(gdf)} counties")
    return gdf

def load_cropland_mask():
    """
    Load cropland mask raster to identify agricultural areas.
    If mask is not available, return None.
    
    Returns:
        rasterio dataset or None: Cropland mask raster
    """
    if not os.path.exists(CROPLAND_MASK_PATH):
        logger.warning(f"Cropland mask not found at {CROPLAND_MASK_PATH}")
        return None
    
    try:
        mask_raster = rasterio.open(CROPLAND_MASK_PATH)
        logger.info(f"Loaded cropland mask with shape {mask_raster.shape}")
        return mask_raster
    except Exception as e:
        logger.error(f"Error loading cropland mask: {e}")
        return None

def extract_state_data_from_geotiff(geotiff_path, state_boundaries, state_code=None, use_cropland_mask=True):
    """
    Extract climate data for a specific state or all states from a GeoTIFF file.
    
    Args:
        geotiff_path (str): Path to GeoTIFF file
        state_boundaries (geopandas.GeoDataFrame): State boundary data
        state_code (str, optional): State code to extract, or None for all states
        use_cropland_mask (bool): Whether to apply cropland mask to focus on agricultural areas
        
    Returns:
        pd.DataFrame: DataFrame with extracted climate data statistics by state
    """
    # Check if this is a sample CSV file (used for testing)
    if geotiff_path.endswith("_sample.csv"):
        try:
            df = pd.read_csv(geotiff_path)
            
            # Filter by state if requested
            if state_code:
                df = df[df['state_code'] == state_code]
            
            # Group by state and calculate statistics
            state_stats = df.groupby('state_code')['value'].agg([
                ('mean_value', 'mean'),
                ('min_value', 'min'),
                ('max_value', 'max'),
                ('std_value', 'std'),
                ('count', 'count')
            ]).reset_index()
            
            logger.info(f"Extracted sample data for {len(state_stats)} states from {geotiff_path}")
            return state_stats
        
        except Exception as e:
            logger.error(f"Error extracting sample data: {e}")
            return pd.DataFrame()
    
    # Process actual GeoTIFF file
    try:
        # Load cropland mask if requested
        cropland_mask = None
        if use_cropland_mask:
            cropland_mask = load_cropland_mask()
        
        # Open the GeoTIFF file
        with rasterio.open(geotiff_path) as src:
            # Filter state boundaries if requested
            if state_code:
                boundaries = state_boundaries[state_boundaries['STATE_CODE'] == state_code]
            else:
                boundaries = state_boundaries
            
            # Extract data for each state
            results = []
            
            for idx, state in boundaries.iterrows():
                state_code = state['STATE_CODE']
                
                # Get the state geometry
                geom = [mapping(state.geometry)]
                
                # Extract data within the state boundary
                out_image, out_transform = mask(src, geom, crop=True, nodata=src.nodata)
                state_data = out_image[0]
                
                # Apply cropland mask if available
                if cropland_mask is not None:
                    # We would need to reproject the cropland mask to match the climate data
                    # This is a complex operation that would require more code
                    # For simplicity, we'll skip the actual implementation here
                    logger.info(f"Would apply cropland mask for {state_code} (not implemented in sample)")
                
                # Calculate statistics (ignoring nodata values)
                valid_data = state_data[state_data != src.nodata]
                
                if len(valid_data) > 0:
                    results.append({
                        'state_code': state_code,
                        'mean_value': float(np.mean(valid_data)),
                        'min_value': float(np.min(valid_data)),
                        'max_value': float(np.max(valid_data)),
                        'std_value': float(np.std(valid_data)),
                        'count': len(valid_data)
                    })
            
            state_stats = pd.DataFrame(results)
            logger.info(f"Extracted data for {len(state_stats)} states from {geotiff_path}")
            return state_stats
    
    except Exception as e:
        logger.error(f"Error extracting data from GeoTIFF {geotiff_path}: {e}")
        return pd.DataFrame()

def extract_county_data_from_geotiff(geotiff_path, county_boundaries, state_code=None, use_cropland_mask=True):
    """
    Extract climate data for counties from a GeoTIFF file.
    
    Args:
        geotiff_path (str): Path to GeoTIFF file
        county_boundaries (geopandas.GeoDataFrame): County boundary data
        state_code (str, optional): State code to extract counties from, or None for all
        use_cropland_mask (bool): Whether to apply cropland mask to focus on agricultural areas
        
    Returns:
        pd.DataFrame: DataFrame with extracted climate data statistics by county
    """
    # For sample CSV files, simulate county-level data
    if geotiff_path.endswith("_sample.csv"):
        try:
            df = pd.read_csv(geotiff_path)
            
            # Filter by state if requested
            if state_code:
                df = df[df['state_code'] == state_code]
            
            # Create sample county-level data
            # In a real implementation, this would use actual county polygons
            county_data = []
            
            for state_code in df['state_code'].unique():
                state_df = df[df['state_code'] == state_code]
                
                # Get counties for this state
                state_counties = county_boundaries[county_boundaries['STATE_CODE'] == state_code]
                
                for _, county in state_counties.iterrows():
                    county_name = county['COUNTY_NAME']
                    county_fips = county['FIPS']
                    
                    # Simulate county values based on state values
                    # In reality, this would be based on spatial overlap
                    base_value = state_df['value'].mean()
                    random_factor = np.random.uniform(0.9, 1.1)
                    
                    county_data.append({
                        'state_code': state_code,
                        'county_name': county_name,
                        'county_fips': county_fips,
                        'mean_value': base_value * random_factor,
                        'min_value': base_value * random_factor * 0.9,
                        'max_value': base_value * random_factor * 1.1,
                        'std_value': state_df['value'].std() * random_factor,
                        'count': 100  # Arbitrary count for samples
                    })
            
            county_stats = pd.DataFrame(county_data)
            logger.info(f"Created sample county statistics for {len(county_stats)} counties")
            return county_stats
        
        except Exception as e:
            logger.error(f"Error creating sample county data: {e}")
            return pd.DataFrame()
    
    # Process actual GeoTIFF file
    try:
        # Filter county boundaries if requested
        if state_code:
            counties = county_boundaries[county_boundaries['STATE_CODE'] == state_code]
        else:
            counties = county_boundaries
        
        # Load cropland mask if requested
        cropland_mask = None
        if use_cropland_mask:
            cropland_mask = load_cropland_mask()
        
        # Open the GeoTIFF file
        with rasterio.open(geotiff_path) as src:
            # Extract data for each county
            results = []
            
            for idx, county in counties.iterrows():
                state_code = county['STATE_CODE']
                county_name = county['COUNTY_NAME']
                county_fips = county['FIPS']
                
                # Get the county geometry
                geom = [mapping(county.geometry)]
                
                # Extract data within the county boundary
                try:
                    out_image, out_transform = mask(src, geom, crop=True, nodata=src.nodata)
                    county_data = out_image[0]
                    
                    # Apply cropland mask if available
                    if cropland_mask is not None:
                        # Similar to state-level, actual implementation would be more complex
                        logger.info(f"Would apply cropland mask for {county_name} (not implemented in sample)")
                    
                    # Calculate statistics (ignoring nodata values)
                    valid_data = county_data[county_data != src.nodata]
                    
                    if len(valid_data) > 0:
                        results.append({
                            'state_code': state_code,
                            'county_name': county_name,
                            'county_fips': county_fips,
                            'mean_value': float(np.mean(valid_data)),
                            'min_value': float(np.min(valid_data)),
                            'max_value': float(np.max(valid_data)),
                            'std_value': float(np.std(valid_data)),
                            'count': len(valid_data)
                        })
                
                except Exception as e:
                    logger.warning(f"Error extracting data for county {county_name}: {e}")
                    continue
            
            county_stats = pd.DataFrame(results)
            logger.info(f"Extracted data for {len(county_stats)} counties from {geotiff_path}")
            return county_stats
    
    except Exception as e:
        logger.error(f"Error extracting county data from GeoTIFF {geotiff_path}: {e}")
        return pd.DataFrame()

def aggregate_data_to_cropping_regions(geotiff_path, state_boundaries, crop_type=None):
    """
    Aggregate climate data to crop-specific growing regions.
    This function identifies key growing regions for specific crops and 
    weights climate data based on crop production areas.
    
    Args:
        geotiff_path (str): Path to GeoTIFF file
        state_boundaries (geopandas.GeoDataFrame): State boundary data
        crop_type (str, optional): Crop type to focus on
        
    Returns:
        pd.DataFrame: DataFrame with climate data for crop regions
    """
    # For sample implementation, we'll create a simplified version
    # In a real implementation, this would use actual crop production maps
    
    # Define crop-specific regions
    crop_regions = {
        'CORN': ['IL', 'IA', 'NE', 'MN', 'IN', 'OH', 'MO'],
        'SOYBEANS': ['IL', 'IA', 'MN', 'IN', 'NE', 'OH', 'MO'],
        'WHEAT': ['KS', 'ND', 'MT', 'WA', 'OK', 'SD', 'TX'],
        'COTTON': ['TX', 'GA', 'MS', 'AR', 'AL', 'NC', 'LA'],
        'RICE': ['AR', 'CA', 'LA', 'MS', 'TX', 'MO']
    }
    
    # Create weights for each state based on typical production
    # In reality, these would be based on USDA production statistics
    crop_weights = {
        'CORN': {
            'IA': 0.25, 'IL': 0.20, 'NE': 0.15, 'MN': 0.10, 
            'IN': 0.10, 'OH': 0.10, 'MO': 0.05, 'OTHER': 0.05
        },
        'SOYBEANS': {
            'IL': 0.22, 'IA': 0.20, 'MN': 0.12, 'IN': 0.10, 
            'NE': 0.10, 'OH': 0.08, 'MO': 0.08, 'OTHER': 0.10
        },
        'WHEAT': {
            'KS': 0.25, 'ND': 0.18, 'MT': 0.12, 'WA': 0.10, 
            'OK': 0.08, 'SD': 0.08, 'TX': 0.07, 'OTHER': 0.12
        },
        'COTTON': {
            'TX': 0.40, 'GA': 0.14, 'MS': 0.10, 'AR': 0.10, 
            'AL': 0.08, 'NC': 0.07, 'LA': 0.06, 'OTHER': 0.05
        },
        'RICE': {
            'AR': 0.50, 'CA': 0.20, 'LA': 0.15, 'MS': 0.08, 
            'TX': 0.04, 'MO': 0.03
        }
    }
    
    # Extract state-level data
    state_data = extract_state_data_from_geotiff(geotiff_path, state_boundaries)
    
    if state_data.empty:
        return pd.DataFrame()
    
    # If no specific crop is provided, aggregate for all crops
    crops_to_process = [crop_type] if crop_type else list(crop_regions.keys())
    
    results = []
    
    for crop in crops_to_process:
        # Get key states for this crop
        key_states = crop_regions.get(crop, [])
        state_weights = crop_weights.get(crop, {})
        
        if not key_states:
            continue
        
        # Filter state data for key states
        crop_state_data = state_data[state_data['state_code'].isin(key_states)]
        
        if crop_state_data.empty:
            continue
        
        # Calculate weighted average for the crop region
        total_weight = 0
        weighted_sum = 0
        
        for _, row in crop_state_data.iterrows():
            state_code = row['state_code']
            state_value = row['mean_value']
            weight = state_weights.get(state_code, state_weights.get('OTHER', 0))
            
            weighted_sum += state_value * weight
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
        else:
            weighted_avg = np.nan
        
        results.append({
            'crop': crop,
            'climate_value': weighted_avg,
            'states_included': len(crop_state_data),
            'primary_region': ', '.join(key_states[:3]) + '...'  # Abbreviated list
        })
    
    return pd.DataFrame(results)

def create_agricultural_mask(state_boundaries, county_boundaries=None, cropland_mask_path=None):
    """
    Create a mask identifying agricultural areas for more accurate climate aggregation.
    
    Args:
        state_boundaries (geopandas.GeoDataFrame): State boundary data
        county_boundaries (geopandas.GeoDataFrame, optional): County boundary data
        cropland_mask_path (str, optional): Path to cropland mask raster
        
    Returns:
        dict: Agricultural mask data structure for use in aggregation
    """
    # This is a simplified implementation
    # In a real system, this would use USDA cropland data layer or similar
    
    # Create basic mask based on state boundaries
    ag_states = state_boundaries[state_boundaries['STATE_CODE'].isin(AGRICULTURAL_STATES)]
    
    # Define importance weights for agricultural states
    state_weights = {
        'IA': 1.0, 'IL': 1.0, 'NE': 0.9, 'MN': 0.9, 'IN': 0.9,  # Top corn/soybean states
        'KS': 0.9, 'ND': 0.8, 'SD': 0.8, 'OH': 0.8, 'MO': 0.8,  # More corn belt & wheat
        'AR': 0.8, 'MS': 0.7, 'LA': 0.7, 'TX': 0.7, 'GA': 0.7,  # Cotton & rice states
        'AL': 0.6, 'SC': 0.6, 'NC': 0.6, 'TN': 0.6,             # More southern states
        'CA': 0.8, 'WA': 0.7, 'OR': 0.6, 'ID': 0.7, 'CO': 0.6,  # Western states
        'MT': 0.7, 'WY': 0.5, 'UT': 0.5                         # Mountain states
    }
    
    # Add weights to states
    ag_mask = {}
    for _, state in ag_states.iterrows():
        state_code = state['STATE_CODE']
        ag_mask[state_code] = {
            'weight': state_weights.get(state_code, 0.5),
            'geometry': state.geometry
        }
    
    logger.info(f"Created agricultural mask with {len(ag_mask)} states")
    return ag_mask

def main():
    """Test functionality."""
    logger.info("Testing spatial aggregation functionality...")
    
    ensure_directories()
    
    # Load state boundaries
    state_boundaries = load_state_boundaries()
    
    # Load county boundaries
    county_boundaries = load_county_boundaries()
    
    # Get sample file path for testing
    sample_dir = os.path.join(RAW_DATA_DIR, "prism", "sample")
    if not os.path.exists(sample_dir):
        logger.warning(f"Sample directory not found at {sample_dir}")
        logger.warning("Please run download_prism_data.py first to create sample data")
        return
    
    var_dir = os.path.join(sample_dir, "ppt")
    if not os.path.exists(var_dir):
        logger.warning(f"Sample variable directory not found at {var_dir}")
        return
    
    sample_files = [f for f in os.listdir(var_dir) if f.endswith("_sample.csv")]
    if not sample_files:
        logger.warning("No sample files found")
        return
    
    sample_file = os.path.join(var_dir, sample_files[0])
    
    # Test state-level aggregation
    logger.info(f"Testing state-level aggregation with sample file: {sample_file}")
    state_data = extract_state_data_from_geotiff(sample_file, state_boundaries)
    
    if not state_data.empty:
        logger.info(f"Successfully aggregated data for {len(state_data)} states")
        
        # Test county-level aggregation
        logger.info("Testing county-level aggregation")
        county_data = extract_county_data_from_geotiff(sample_file, county_boundaries)
        
        if not county_data.empty:
            logger.info(f"Successfully aggregated data for {len(county_data)} counties")
        
        # Test crop region aggregation
        logger.info("Testing crop region aggregation")
        region_data = aggregate_data_to_cropping_regions(sample_file, state_boundaries)
        
        if not region_data.empty:
            logger.info(f"Successfully aggregated data for {len(region_data)} crop regions")
    
    logger.info("Spatial aggregation testing complete.")

if __name__ == "__main__":
    main()