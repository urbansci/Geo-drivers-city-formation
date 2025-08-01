"""
# -- coding: utf-8 --
Author: Weiyu Zhang
Date: 2024
Description: Calculates 'second nature geography' for each city. 
             For each location-year pair, this script counts the existing cities within a set of distance bands before the founding year.
             Here, China-pref refers to the Chinese prefecture-level cities, i.e., CHGIS v6 data.
             
"""

import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(dataset: str) -> tuple:
    """
    Load grid points and city points data for specified dataset.
    
    Args:
        dataset (str): Dataset type ('China', 'China-wall', 'China-pref', or 'Europe')
    
    Returns:
        tuple: (grid_points, city_points) as GeoDataFrames
    """
    if dataset == "China-pref":
        grid_points = gpd.read_file('/path/to/china_pref_grid_centers.shp')
        city_points = gpd.read_file('/path/to/v6_time_pref_pts_utf_wgs84.shp') # original CHGIS v6 data
        city_points.set_crs(epsg=4326, inplace=True)
    elif dataset == "Europe":
        grid_points = gpd.read_file('/path/to/eu_grid_centers.shp')
        city_points = gpd.read_file('/path/to/eu_cities.shp')
        city_points.set_crs(epsg=4326, inplace=True)
    
    return grid_points, city_points

def transform_to_projected_crs(grid_points: gpd.GeoDataFrame, city_points: gpd.GeoDataFrame, dataset: str) -> tuple:
    """
    Transform data to projected coordinate reference system for accurate distance calculations.
    
    Args:
        grid_points (gpd.GeoDataFrame): Grid points in geographic CRS
        city_points (gpd.GeoDataFrame): City points in geographic CRS
        dataset (str): Dataset identifier for CRS selection
    
    Returns:
        tuple: (grid_points, city_points) transformed to projected CRS
    """
    if dataset == "China":
        proj_string = '+proj=lcc +lat_1=25 +lat_2=47 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    else:  # Europe
        proj_string = '+proj=lcc +lat_1=43 +lat_2=62 +lon_0=15 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    
    grid_points = grid_points.to_crs(proj_string)
    city_points = city_points.to_crs(proj_string)
    
    return grid_points, city_points

def calculate_second_nature(
    grid_points: gpd.GeoDataFrame,
    city_points: gpd.GeoDataFrame,
    buffer_distances: list,
    country: str
) -> pd.DataFrame:
    """
    Calculate second nature geography by counting cities within buffer distances for each historical year.
    
    Args:
        grid_points (gpd.GeoDataFrame): Spatial grid points for analysis
        city_points (gpd.GeoDataFrame): Historical city data with birth years
        buffer_distances (list): List of buffer distances in meters
        country (str): Country identifier for temporal filtering logic
    
    Returns:
        pd.DataFrame: Grid points with city count columns for each year and buffer distance
    """
    # Ensure consistent birth column naming
    if "BEG_YR" in city_points.columns:
        city_points.rename(columns={'BEG_YR': 'birth'}, inplace=True)
    
    print(city_points["birth"].value_counts().sort_index())
    
    # Get all unique city birth years
    unique_years = sorted(city_points['birth'].unique())
    
    # Create result DataFrame
    grid_points_result = grid_points.copy()
    
    # Create column names for each buffer distance
    buffer_columns = [f'city_count_{int(dist/1000)}km' for dist in buffer_distances]
    
    # Process each city birth year
    for year in unique_years:
        logger.info(f"Processing year: {year}")
        
        # Process each buffer distance
        for dist, col in zip(buffer_distances, buffer_columns):
            col_name = f'Y{year}_{col}'
            logger.info(f"Processing buffer distance: {dist} meters")
            
            # Filter cities existing up to the current year
            if country == "Europe":
                city_buffers = city_points[(city_points['birth'] <= (year-100))].copy() 
            else:
                city_buffers = city_points[(city_points['birth'] < year)&(city_points['END_YR'] > (year))].copy()
            
            # Create buffer geometries
            city_buffers['geometry'] = city_buffers.geometry.buffer(dist)
            
            # Perform spatial join to find intersections
            joined = gpd.sjoin(
                grid_points_result[['geometry']],
                city_buffers[['geometry']],
                how='inner',
                predicate='intersects'
            )
            
            # Count cities per grid point
            counts = joined.reset_index().groupby('index').size()
            counts_aligned = counts.reindex(grid_points_result.index, fill_value=0)
            
            # Add results to DataFrame
            grid_points_result[col_name] = counts_aligned
            
            logger.info(f"Completed {col_name}")
    
    return grid_points_result

def save_results(result_df: pd.DataFrame, dataset: str):
    """
    Save calculation results to parquet file.
    
    Args:
        result_df (pd.DataFrame): Results DataFrame with city counts
        dataset (str): Dataset identifier for output path selection
    """
    # Prepare for saving
    result_df.rename(columns={'ID': 'id'}, inplace=True)
    result_df.drop(columns='geometry', inplace=True)
    
    # Define output paths
    if dataset == "Europe":
        output_parquet = '/path/to/005deg_grids_European_city_count.parquet'
    elif dataset == "China-pref":
        output_parquet = '/path/to/005deg_grids_Chinese_pref_city_count.parquet'

    # Save results
    result_df.to_parquet(output_parquet)
    logger.info(f"Results saved to {output_parquet}")

def main():
    """
    Main function to execute second nature geography calculation workflow.
    """
    # Set parameters
    dataset = "China-pref"  # Options: ["Europe", "China-pref"]
    buffer_distances = [20000, 50000, 100000]  # 20km, 50km, 100km buffer distances
    
    # Load data
    logger.info("Loading data...")
    grid_points, city_points = load_data(dataset)
    
    # Transform coordinate systems
    logger.info("Transforming coordinate systems...")
    grid_points, city_points = transform_to_projected_crs(grid_points, city_points, dataset)
    
    # Calculate second nature geography
    logger.info("Calculating second nature geography...")
    result_df = calculate_second_nature(grid_points, city_points, buffer_distances, dataset)
    
    # Save results
    logger.info("Saving results...")
    save_results(result_df, dataset)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()