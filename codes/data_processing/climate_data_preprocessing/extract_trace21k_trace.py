"""
# -- coding: utf-8 --
Author: Weiyu Zhang
Date: 2024
Description: A script to extract time-series data from
             NetCDF climate datasets (e.g., TraCE-21k).
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import box
import logging

# Configure the logging module for progress updates.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_regional_average_timeseries(
    nc_path: str, 
    polygon_gdf: gpd.GeoDataFrame, 
    var_name: str, 
    unit_conversion_offset: float
) -> tuple:
    """
    Calculates the average value of a variable within a polygon over a specified time range.

    Args:
        nc_path (str): The path to the NetCDF climate data.
        polygon_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the polygon for the study area.
        var_name (str): The name of the climate variable to analyze (e.g., 'TS', 'TMQ').
        unit_conversion_offset (float): A value to add for unit conversion (e.g., -273.15 for K to °C, 0 for no change).

    Returns:
        tuple: A tuple containing two numpy arrays: (calendar_years, data_series).
    """
    logger.info(f"Opening NetCDF file: {nc_path}")
    with nc.Dataset(nc_path, 'r') as dataset:
        lons = dataset.variables['lon'][:]
        lats = dataset.variables['lat'][:]
        # The time variable might need unit adjustments based on the dataset's metadata.
        # Here we assume it's in thousands of years (ka) before present (1950).
        times = dataset.variables['time'][:] * 1000
        
        logger.info("Creating spatial mask for the provided polygon...")
        # Create a boolean mask for grid cells inside the polygon.
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for i in range(len(lats)):
            for j in range(len(lons)):
                # This check can be accelerated with vectorized operations.
                if any(polygon_gdf.geometry.contains(box(lons[j]-0.5, lats[i]-0.5, lons[j]+0.5, lats[i]+0.5))):
                    mask[i, j] = True
        
        if not np.any(mask):
            logger.warning("The provided polygon does not overlap with the NetCDF grid. No data will be extracted.")
            return np.array([]), np.array([])

        logger.info("Extracting data for the defined time period and region...")
        # Define time range for analysis (e.g., 300 BCE to 1900 CE).
        calendar_years = 1950 + times
        time_mask = (calendar_years >= -300) & (calendar_years <= 1900)
        time_indices = np.where(time_mask)[0]
        
        # Extract data for each time step within the masked region.
        regional_avg_data = []
        for time_idx in time_indices:
            data_slice = dataset.variables[var_name][time_idx, :, :]
            masked_data = data_slice[mask]
            if masked_data.size > 0:
                mean_val = np.mean(masked_data) + unit_conversion_offset
                regional_avg_data.append(mean_val)
        
        return calendar_years[time_mask], np.array(regional_avg_data)

def save_timeseries_to_parquet(years: np.ndarray, data: np.ndarray, data_label: str, output_path: str):
    """
    Saves a time-series dataset to a Parquet file, including original and smoothed data.

    Args:
        years (np.ndarray): Array of years for the time series.
        data (np.ndarray): Array of corresponding data values.
        data_label (str): The base name for the data columns (e.g., 'temperature').
        output_path (str): The path to save the output Parquet file.
    """
    if years.size == 0 or data.size == 0:
        logger.warning("Input data is empty. Skipping file save.")
        return

    logger.info(f"Preparing and saving data to {output_path}")
    # Calculate a smoothed version of the data for convenience.
    # Assumes a 300-year window based on 10-year timesteps (30 points).
    smoothed_data = gaussian_filter1d(data, sigma=30/6)
    
    # Create a pandas DataFrame.
    df = pd.DataFrame({
        'year': years.astype(int),
        f'{data_label}_original': data,
        f'{data_label}_smoothed': smoothed_data,
        'period': ['BCE' if y < 0 else 'CE' for y in years]
    })
    
    # Save the DataFrame to the efficient Parquet format.
    df.to_parquet(output_path, index=False)
    logger.info("Time-series data saved successfully.")
    print("\nData Preview:")
    print(df.head())

def main():
    """
    Main execution block to run the climate data extraction pipeline.
    """
    # --- 1. Configuration ---
    # NOTE: Replace with the actual paths to your data files.
    CLIMATE_NC_PATH = "/path/to/your/trace21k.nc"
    REGION_SHP_PATH = "/path/to/your/china_boundary.shp"
    OUTPUT_PARQUET_PATH = "/path/to/your/output/china_temperature_series.parquet"
    
    # Analysis parameters
    VARIABLE_TO_EXTRACT = 'TS'  # e.g., 'TS' for surface temperature
    UNIT_CONVERSION_OFFSET = -273.15 # For Kelvin to Celsius
    DATA_LABEL = 'temperature' # For column naming in the output file

    # --- 2. Load Region of Interest ---
    logger.info(f"--- Loading region of interest from: {REGION_SHP_PATH} ---")
    try:
        region_polygon = gpd.read_file(REGION_SHP_PATH)
    except Exception as e:
        logger.error(f"Failed to load shapefile: {e}")
        return

    # --- 3. Regional Time-Series Extraction ---
    logger.info(f"--- Analyzing '{VARIABLE_TO_EXTRACT}' for the specified region ---")
    years, data = calculate_regional_average_timeseries(
        nc_path=CLIMATE_NC_PATH,
        polygon_gdf=region_polygon,
        var_name=VARIABLE_TO_EXTRACT,
        unit_conversion_offset=UNIT_CONVERSION_OFFSET
    )
    
    # --- 4. Save Results ---
    if data.size > 0:
        logger.info("--- Saving extracted time-series data ---")
        save_timeseries_to_parquet(years, data, DATA_LABEL, OUTPUT_PARQUET_PATH)
    else:
        logger.warning("No data was extracted, so no output file will be saved.")

    logger.info("--- Process Completed ---")

if __name__ == "__main__":
    main()