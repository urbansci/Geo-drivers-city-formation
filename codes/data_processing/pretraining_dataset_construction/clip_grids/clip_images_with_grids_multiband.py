"""
# -*- coding: utf-8 -*-
Author: Junjie Yang, Weiyu Zhang
Date: 2024
Description: Clip and stack climate variables into multi-band TIFF images.
             Designed for climate data processing where multiple bioclimatic variables
             need to be combined into single multi-channel images for efficient
             self-supervised learning model training and inference.
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
import tifffile


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


def clip_resize_multiband(grids, bio_ids, ssp, out_path, tar_size, year, resize=False, clim_type=None, grid_data=None):
    """
    Clip multiple climate variables and stack them into multi-band TIFF images.
    
    Args:
        grids (str): Path to the grid shapefile
        bio_ids (list): List of bioclimatic variable IDs to process
        ssp (str): SSP scenario name for future climate data
        out_path (str): Output directory for multi-band images
        tar_size (int): Target image size for resizing
        year (int): Year identifier for file naming
        resize (bool): Whether to resize images to target size
        clim_type (str): Climate data type (future_clim, current_clim)
        grid_data (gpd.GeoDataFrame): Pre-loaded grid data for efficiency
    """
    
    print("Checking whether data is loaded")
    if grid_data is None:
        grid_data = gpd.read_file(grids)
    print("=========> Grid data loaded")
    
    # Create output directory
    os.makedirs(out_path, exist_ok=True)
    
    # Standardize ID column naming
    if "id" in grid_data.columns:
        grid_data = grid_data.rename(columns={"id": "ID"})
    
    # Load all raster datasets for different bioclimatic variables
    raster_data_list = []
    if clim_type == "future_clim":
        for bio_id in bio_ids:
            # Use 3-model ensemble for future climate projections
            raw_path = "/path/to/chelsa/future/ensemble_3models/{s}/ensemble_mean_bio{bio}.tif".format(
                s=ssp, bio=bio_id)
            raster_data_list.append(rasterio.open(raw_path))
            
    elif clim_type == "current_clim":
        for bio_id in bio_ids:
            raw_path = "/path/to/chelsa/current/CHELSA_bio{bio}_1981-2010_V.2.1.tif".format(bio=bio_id)
            raster_data_list.append(rasterio.open(raw_path))
    else:
        raise NotImplementedError("Only future_clim and current_clim are supported")

    # Process each grid cell
    for index, grid in tqdm(grid_data.iterrows(), total=grid_data.shape[0], desc="Clipping and Stacking"):
        img_list = []
        
        # Clip each bioclimatic variable for current grid cell
        for raster_data in raster_data_list:
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
                img = Image.fromarray(out_image[0]).convert("F")

                # Apply climate-specific processing
                if clim_type == "clim":
                    raise NotImplementedError("Single-band clim processing not supported in multiband function")
                elif clim_type in ["paleo_clim", "future_clim", "current_clim"]:
                    if resize:
                        img = img.resize((tar_size, tar_size), Image.BILINEAR)
                else:
                    assert False, "clim_type should be paleo_clim, future_clim, or current_clim"
                
                img_list.append(img)
                
            except Exception as e:
                print("Error in clipping and resizing grid: ", grid["ID"])
                print(e)
                continue

        # Stack all bioclimatic variables into multi-band image
        try:
            # Convert PIL Images to numpy arrays
            img_list = [np.array(img) for img in img_list]
            
            # Stack arrays along the last axis (channels)
            array_multi_channel = np.stack(img_list, axis=-1)
            
            # Generate output filename
            _id = grid["ID"]
            file_path = os.path.join(out_path, "Y{year}_{id}".format(year=year, id=int(_id)))

            # Clean filename by removing invalid characters
            file_path = file_path.replace('|', '-')
            file_path = file_path.replace('?', '-') 
            file_path = file_path.replace('*', '-') 
            
            # Save as multi-band TIFF using tifffile
            tifffile.imwrite(
                file_path + ".tiff",
                array_multi_channel,
                photometric="minisblack",
                planarconfig="contig"
            )
            
        except Exception as e:
            print('Error in stacking and writing grid: ', grid["ID"])
            print(e)
            continue


if __name__ == "__main__":
    print('Starting multi-band climate data processing')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ims", "--imgsize", type=int, default=24, 
                        help="Image size for clipped data, default is 24")
    parser.add_argument("type", type=str, default="current_clim", 
                        help="Type of climate data: future_clim, current_clim")
    parser.add_argument('--out', '-o', type=str,
                        default="/path/to/output/directory")
    parser.add_argument('--grid_path', '-gp', type=str, 
                        default="/path/to/grid/shapefile.shp")
    parser.add_argument("-r", '--resize', type=bool, default=True)
    parser.add_argument("-y", '--year', type=int, default=2000)
    parser.add_argument("-ssp", '--ssp', type=str, default="ssp126")
    
    arg = parser.parse_args()
    print("arg.resize:", arg.resize)
    
    # Determine data types to process
    types = []
    if arg.type == "all":
        types = ["future_clim", "current_clim"]
    else:
        types = [arg.type]

    print(f"arg.type: {arg.type}")
    print(f"types: {types}")
    
    # Process each climate data type
    for type_ in types:
        if type_ == "future_clim":
            ssp = arg.ssp
            # Bioclimatic variables in numerical order
            bio_ids = [1, 4, 7, 8, 9, 12, 15, 18, 19]
            print("Processing bioclimatic variables:", bio_ids)
            
            arg.out = os.path.join(arg.out, ssp)
            print("Dealing with future climate SSP: ", ssp)
            os.makedirs(arg.out, exist_ok=True)
            
            print("Loading grid data from:", arg.grid_path)
            grid_data = gpd.read_file(arg.grid_path)
            print("=========> Grid data loaded")
            
            clip_resize_multiband(
                arg.grid_path,
                bio_ids,
                ssp,
                arg.out, 
                arg.imgsize, 
                arg.year,
                resize=arg.resize,
                clim_type="future_clim",
                grid_data=grid_data
            )
            
        elif type_ == "current_clim":
            # Bioclimatic variables in numerical order
            bio_ids = [1, 4, 7, 8, 9, 12, 15, 18, 19]
            print("Processing bioclimatic variables:", bio_ids)
            
            print("Dealing with current climate")
            os.makedirs(arg.out, exist_ok=True)
            
            print("Loading grid data from:", arg.grid_path)
            grid_data = gpd.read_file(arg.grid_path)
            print("=========> Grid data loaded")
            
            clip_resize_multiband(
                arg.grid_path,
                bio_ids,
                None,  # No SSP scenario for current climate
                arg.out, 
                arg.imgsize, 
                arg.year,
                resize=arg.resize,
                clim_type="current_clim",
                grid_data=grid_data
            )
        else: 
            raise NotImplementedError("Data type not implemented")