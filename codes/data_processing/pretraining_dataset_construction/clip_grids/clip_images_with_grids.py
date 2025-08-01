"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: Unified raster data clipping tool for geographical analysis.
             Processes multiple data types (DEM, water, climate, built-up areas) by clipping
             raw raster datasets to grid cell extents, applying data-specific preprocessing,
             and saving as TIFF images for self-supervised learning and temporal analysis.
"""

import rasterio 
import argparse
from tqdm import tqdm
import rasterio.mask
import rasterio.plot
from PIL import Image
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.wkt import loads
from shapely.ops import cascaded_union
import math
import sys, os, time


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


def clip_resize(grids, raw_path, out_path, tar_size, data_type, year, resize=False, filter_built=False, clim_type=None, grid_data=None):
    """
    Clip raw raster data to grid cell extent and save as image tiles.
    
    Args:
        grids (str): Path to the grid shapefile
        raw_path (str): Path to the raw raster data
        out_path (str): Output directory for clipped images
        tar_size (int): Target image size for resizing
        data_type (str): Type of data being processed
        year (int): Year identifier for file naming
        resize (bool): Whether to resize images to target size
        filter_built (bool): Filter built-up areas above threshold
        clim_type (str): Climate data type (clim, paleo_clim, future_clim, current_clim)
        grid_data (gpd.GeoDataFrame): Pre-loaded grid data for efficiency
    """
    
    # Load raster data
    raster_data = rasterio.open(raw_path)
    print("=========> Raster data loaded")
    
    # Load grid data if not provided
    if grid_data is None:
        grid_data = gpd.read_file(grids)
    print("=========> Grid data loaded")
    print("=========> Type:", data_type)
    
    # Standardize ID column naming
    if "Id" in grid_data.columns:
        grid_data.rename(columns={"Id": "ID"}, inplace=True)
    elif "cid" in grid_data.columns:
        grid_data.rename(columns={"cid": "ID"}, inplace=True)
    elif "id" in grid_data.columns:
        grid_data.rename(columns={"id": "ID"}, inplace=True)

    binary = False    
    # Assert binary setting for non-water data types
    if data_type != "water":
        assert binary == False, "binary should be False when type is not water"

    # Create output directory based on data type
    if binary or data_type != "water":
        out_path = os.path.join(out_path, data_type)
        os.makedirs(out_path, exist_ok=True)
    else:
        out_path = os.path.join(out_path, data_type + "_not_binary")
        os.makedirs(out_path, exist_ok=True)
    
    # Process each grid cell
    for index, grid in tqdm(grid_data.iterrows(), total=grid_data.shape[0], desc="Clipping and Resizing"):
        try:
            # Extract bounding box coordinates from grid geometry
            geometry = grid['geometry']
            minx, miny, maxx, maxy = geometry.bounds

            # Convert geographic coordinates to raster indices
            row_min, col_min = raster_data.index(minx, maxy)  # Top-left
            row_max, col_max = raster_data.index(maxx, miny)  # Bottom-right

            # Create window for clipping
            window = rasterio.windows.Window.from_slices((row_min, row_max), (col_min, col_max))

            # Extract image within window
            out_image = raster_data.read(window=window)
            out_image = np.array(out_image, dtype=np.float32)
            
            # Handle NoData values for non-bioclimatic data
            if "bio" not in data_type:
                out_image[np.isnan(out_image)] = 0
                out_image[out_image < -3000] = 0
            
            img = Image.fromarray(out_image[0]).convert("F")
            
        except Exception as e:
            print("Error in clipping grid: ", grid["ID"])
            if "bio" in data_type:  # More detailed error info for climate data
                print(e)
            continue

        # Apply data-type specific processing
        if data_type == "water":
            # Process water data with thresholding
            if binary:
                if resize:
                    img = img.resize((tar_size, tar_size), Image.BILINEAR)
                img_arr = np.array(img)
                img_arr[img_arr > 80] = 255
                img_arr[img_arr <= 80] = 0
                img = Image.fromarray(img_arr).convert("1")
                
        elif "bio" in data_type:
            # Process bioclimatic variables with climate-specific logic
            if clim_type == "clim":
                # Process contemporary climate data with mean imputation
                img_arr = np.array(img)
                
                # Check if there are valid values
                if img_arr[img_arr > -100].size == 0:
                    # Remove existing file if no valid data
                    existing_file = os.path.join(out_path, "Y{year}_{id}.tiff".format(year=year, id=int(grid["ID"])))
                    if os.path.exists(existing_file):
                        print("Remove grid: ", grid["ID"])
                        os.remove(existing_file)
                    continue
                
                # Calculate mean of valid values and impute invalid ones
                mean_val = np.mean(img_arr[img_arr > -100])
                img_arr[img_arr < -100] = mean_val
                img = Image.fromarray(img_arr).convert("F")
                
                if resize:
                    img = img.resize((tar_size, tar_size), Image.BILINEAR)
                    
            elif clim_type in ["paleo_clim", "future_clim", "current_clim"]:
                # Process paleoclimate, future climate, or current climate data
                if resize:
                    img = img.resize((tar_size, tar_size), Image.BILINEAR)
            else:
                # Default climate processing (for backward compatibility)
                img_arr = np.array(img)
                if img_arr[img_arr > -100].size > 0:
                    mean_val = np.mean(img_arr[img_arr > -100])
                    img_arr[img_arr < -100] = mean_val
                    img = Image.fromarray(img_arr).convert("F")
                if resize:
                    img = img.resize((tar_size, tar_size), Image.BILINEAR)
        else:
            # Default processing for other data types (DEM, slope, etc.)
            if resize:
                img = img.resize((tar_size, tar_size), Image.BILINEAR)

        # Generate output filename
        _id = grid["ID"]
        file_path = os.path.join(out_path, "Y{year}_{id}".format(year=year, id=int(_id)))

        # Clean filename by removing invalid characters
        file_path = file_path.replace('|', '-')
        file_path = file_path.replace('?', '-') 
        file_path = file_path.replace('*', '-') 
        
        try:
            img.save(file_path + ".tiff") 
            continue
        except:
            print("Error in saving file: ", file_path)
            
    # Save filtered results if applicable
    if filter_built:
        grid_data.to_file("/path/to/output/filtered_grids.shp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ims", "--imgsize", type=int, default=160, 
                        help="Image size for clipped data, default is 160")
    parser.add_argument("type", type=str, default="built", 
                        help="Type of data to clip: built, water, dem, clim, paleo_clim, future_clim, current_clim")
    parser.add_argument('--out', '-o', type=str,
                        default="/path/to/output/directory")
    parser.add_argument('--grid_path', '-gp', type=str, 
                        default="/path/to/grid/shapefile.shp")
    parser.add_argument("-r", '--resize', type=bool, default=True)
    parser.add_argument("-y", '--year', type=int, default=2000)
    parser.add_argument("-ssp", '--ssp', type=str, default="ssp126")
    
    arg = parser.parse_args()
    
    # Determine data types to process
    types = [arg.type]
        
    # Process each data type
    for type_ in types:
        if type_ == "water":
            clip_resize(
                arg.grid_path,
                "/path/to/global/surface/water.vrt", 
                arg.out, 
                arg.imgsize, 
                type_, 
                1000,
                resize=arg.resize
            )
        elif type_ == "dem":
            clip_resize(
                arg.grid_path,
                "/path/to/global/dem.vrt",
                arg.out, 
                arg.imgsize, 
                type_, 
                1000,
                resize=arg.resize
            )
        elif type_ == "paleo_clim":
            # Process paleoclimate data (9 bioclimatic variables)
            bio_ids = [1, 4, 12, 15, 7, 8, 9, 18, 19]
            arg.out = os.path.join(arg.out, str(arg.year//100))
            os.makedirs(arg.out, exist_ok=True)
            grid_data = gpd.read_file(arg.grid_path)
            print("=========> Grid data loaded")

            for bio_id in bio_ids:
                # Select appropriate paleoclimate dataset based on year
                raw_path = "/path/to/chelsa/trace21k/CHELSA_TraCE21k_bio{bio}_{yr}_V1.0.tif".format(
                        bio=str(bio_id).zfill(2), yr=str(arg.year//100))  
                clip_resize(
                    arg.grid_path,
                    raw_path,
                    arg.out, 
                    arg.imgsize, 
                    "bio_{}".format(bio_id), 
                    arg.year,
                    resize=arg.resize,
                    clim_type="paleo_clim",
                    grid_data=grid_data
                )
                
        elif type_ == "future_clim":
            # Process future climate projections (9 bioclimatic variables)
            ssp = arg.ssp
            bio_ids = [1, 4, 12, 15, 7, 8, 9, 18, 19]
            arg.out = os.path.join(arg.out, ssp)
            print("Dealing with future climate SSP: ", ssp)
            os.makedirs(arg.out, exist_ok=True)
            grid_data = gpd.read_file(arg.grid_path)
            print("=========> Grid data loaded")
            
            for bio_id in bio_ids:
                raw_path = "/path/to/chelsa/future/{s}/ensemble_mean_bio{bio}.tif".format(
                    s=ssp, bio=bio_id)
                clip_resize(
                    arg.grid_path,
                    raw_path,
                    arg.out, 
                    arg.imgsize, 
                    "bio_{}".format(bio_id), 
                    arg.year,
                    resize=arg.resize,
                    clim_type="future_clim",
                    grid_data=grid_data
                )
                
        elif type_ == "current_clim":
            # Process current climate data (9 bioclimatic variables)
            bio_ids = [1, 4, 12, 15, 7, 8, 9, 18, 19]
            print("Dealing with current climate")
            os.makedirs(arg.out, exist_ok=True)
            grid_data = gpd.read_file(arg.grid_path)
            print("=========> Grid data loaded")
            
            for bio_id in bio_ids:
                raw_path = "/path/to/chelsa/current/CHELSA_bio{bio}_1981-2010_V.2.1.tif".format(bio=bio_id)
                clip_resize(
                    arg.grid_path,
                    raw_path,
                    arg.out, 
                    arg.imgsize, 
                    "bio_{}".format(bio_id), 
                    arg.year,
                    resize=arg.resize,
                    clim_type="current_clim",
                    grid_data=grid_data
                )
        else: 
            raise NotImplementedError("Data type not implemented")