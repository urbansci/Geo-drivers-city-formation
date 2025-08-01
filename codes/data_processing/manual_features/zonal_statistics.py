"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: Extract manual features (bioclimatic variables and agricultural potential) 
             from raster data. First, it clips the raster data to grid cells defined in spatial grid shapefile,
             then calculates zonal statistics (mean) for each grid cell, and saves the results to
             a new shapefile with attributes.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.plot
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.wkt import loads


def read_raw_data(file_path, data_type):
    """
    Read raster data from file path.
    
    Args:
        file_path (str): Directory path containing the raster file
        data_type (str): Type/name of the raster file
    Returns:
        rasterio.DatasetReader: Opened raster dataset
    """
    raw_data = rasterio.open(os.path.join(file_path, data_type))
    return raw_data


def read_grid_data(file_path):
    """
    Read spatial grid data from shapefile.
    
    Args:
        file_path (str): Path to the grid shapefile
    Returns:
        geopandas.GeoDataFrame: Grid geometries and attributes
    """
    grid_data = gpd.read_file(file_path)
    return grid_data


def extract_left_upper_coordinate(polygon):
    """
    Extract the left-upper coordinate from a polygon.
    
    Args:
        polygon (shapely.geometry.Polygon): Input polygon geometry
    Returns:
        tuple: (x, y) coordinates of the left-upper corner
    """
    x, y = polygon.exterior.coords[0]
    return x, y


def clip_resize(grids, raw_path, types):
    """
    Extract statistical features from raster data within grid cells.
    
    Args:
        grids (str): Path to the grid shapefile
        raw_path (str): Base path to raster data
        types (list): List of data types/variables to process
    """
    # Load existing grid with attributes or create new one
    if os.path.exists(grids.replace(".shp", "_with_attributes.shp")):
        grid_data = gpd.read_file(grids.replace(".shp", "_with_attributes.shp"))
    else:
        grid_data = gpd.read_file(grids)
        
    # Standardize ID column name
    if "Id" in grid_data.columns:
        grid_data.rename(columns={"Id": "ID"}, inplace=True)
        
    # Process each variable type
    for data_type in types:
        print("Processing {}...".format(data_type))
        
        # Determine raster file path for WorldClim data
        if "wc" in data_type:
            raw_path = os.path.join(r"D:\Urban Niche\Data\GIS files\wclim2.1_30s_bio", data_type + ".tif")
            
        grid_data[data_type] = -1000
        raster_data = rasterio.open(raw_path)
        
        # Process each grid cell
        for index, grid in tqdm(grid_data.iterrows(), total=grid_data.shape[0], desc="Clipping and Resizing"):
            multipolygon = MultiPolygon([grid['geometry']])
            
            # Clip the raster data to grid cell
            out_image, _ = rasterio.mask.mask(raster_data, multipolygon.geoms, crop=True, all_touched=True)
            
            # Calculate statistics on the clipped image
            img_arr = np.array(out_image[0].flatten())
            
            # Remove invalid pixels (< -100)
            img_arr = img_arr[img_arr >= -100]
            
            if img_arr.shape[0] == 0:
                mean = np.nan
            else:
                mean = np.mean(img_arr)
            
            grid_data.loc[index, data_type] = mean
    
    print(grid_data)
    
    # Save updated grid with extracted features
    out_path = grids.replace(".shp", "_with_attributes.shp")
    grid_data.to_file(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="type of the clip data, default is built, can be built, light, dem, water")
    parser.add_argument('--path', '-p', type=str, 
                        default=r"D:\Urban Niche\Data\GIS files\test.tif")
    parser.add_argument('--grid_path', '-gp', type=str, 
                        default=r"D:\Urban Niche\Data\GIS files\Grid\Chinese_grids\005deg_Chinese_wallcity_centered_grids.shp")
    
    arg = parser.parse_args()

    # Process WorldClim bioclimatic variables (19 variables)
    if arg.type == "wc":
        types = ["wc_{}".format(i) for i in range(1, 20)]
        clip_resize(arg.grid_path, arg.path, types)
    # Process agricultural potential
    elif arg.type == "agri":
        types = [arg.type]
        clip_resize(arg.grid_path, arg.path, types)