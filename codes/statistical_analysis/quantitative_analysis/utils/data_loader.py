# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: Loading and preparing geographical datasets for data preprocessing and classification. 
             This script is mainly used in temporal analysis, please replace the path variables to climate embeddings 
             and agricultural data path in the corresponding load functions.
             
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoDataLoader:
    """Geographical data loader for handling multiple data sources."""
    
    def __init__(
        self, 
        base_path: str = None, 
        config: FeatureConfig = FeatureConfig(), 
        data_file_name: str = "CN_attribute_dem_clim_augmented.parquet"
    ):
        """
        Initialize the data loader.
        
        Args:
            base_path (str): Root directory path for data
            config (FeatureConfig): Feature configuration object
            data_file_name (str): Name of the main data file
        """
        self.base_path = Path(base_path)
        self.config = config
        self.data_file_name = data_file_name
    
    def load_data(self, model_name: str) -> pd.DataFrame:
        """
        Load merged data.
        
        Args:
            model_name (str): Model name for locating embedding data files
            
        Returns:
            pd.DataFrame: Complete merged dataset
        """
        logger.info("Starting data loading...")
        
        # 1. Load embedding data
        embedding_path = self.base_path / "embeddings" / model_name / self.data_file_name
        logger.info(f"Loading embedding data: {embedding_path}")
        combined_df = pd.read_parquet(embedding_path)
        
        logger.info(f"Data loading completed, final dataset shape: {combined_df.shape}")
        return combined_df
    
    def load_agri_data(
        self, 
        combined_df: pd.DataFrame, 
        data_type: str = None, 
        path: str = None, 
        year: int = None
    ) -> pd.DataFrame:
        """
        Load agricultural data.
        
        Args:
            combined_df (pd.DataFrame): Existing combined dataframe
            data_type (str): Data type, can be 'pref', 'county', or 'eu'
            path (str): Custom path to agricultural data file
            year (int): Year for temporal filtering
            
        Returns:
            pd.DataFrame: Dataframe with agricultural data merged
            
        Raises:
            ValueError: If data_type is not valid
        """
        if path is None:
            if data_type == 'pref':
                path = "/path/to/Chinese_pref/agri_gaez.parquet"
            elif data_type == 'county':
                path = "/path/to/Chinese_county/agri_gaez.parquet"
            elif data_type == 'eu':
                path = "/path/to/Euro/agri_gaez.parquet"
            else:
                raise ValueError("Invalid data_type parameter, please specify 'pref', 'county', or 'eu'")
        
        logger.info(f"Loading {data_type} data: {path}")
        agri_data = pd.read_parquet(path)
        
        if year is not None:
            if year <= 1500:
                agri_data = agri_data[["id", "pre"]]
                agri_data.rename(columns={"pre": "agri"}, inplace=True)
            else:
                agri_data = agri_data[["id", "post"]]
                agri_data.rename(columns={"post": "agri"}, inplace=True)
        
        if "agri" in combined_df.columns:
            combined_df.drop(columns=["agri"], inplace=True)
        combined_df = combined_df.merge(agri_data, on='id', how='inner')

        return combined_df

    def load_climate_data(
        self, 
        combined_df: pd.DataFrame, 
        data_type: str = None, 
        year: int = None, 
        path: str = None
    ) -> pd.DataFrame:
        """
        
        Args:
            combined_df (pd.DataFrame): Existing combined dataframe
            data_type (str): Data type, can be 'pref', 'county', or 'eu'
            year (int): Year for climate data
            path (str): Custom path to climate data file
            
        Returns:
            pd.DataFrame: Dataframe with climate data merged
            
        Raises:
            ValueError: If data_type is not valid
        """
        if path is None:
            if data_type == 'pref':
                path = f"/path/to/Chinese_pref/paleo_clim_Y{year}.parquet"
            elif data_type == 'eu':
                path = f"/path/to/Euro/paleo_clim_Y{year}.parquet"
            else:
                raise ValueError("Invalid data_type parameter, please specify 'pref', 'county', or 'eu'")

        logger.info(f"Loading {data_type} data: {path}")
        climate_data = pd.read_parquet(path)
        
        # Standardize ID column name
        if "ID" in climate_data.columns:
            climate_data.rename(columns={"ID": "id"}, inplace=True)
        
        bio_cols = [col for col in climate_data.columns if col.startswith("bio")]
        if bio_cols:
            combined_df.drop(columns=bio_cols, inplace=True, errors='ignore')
        combined_df = combined_df.merge(climate_data, on='id', how='inner')

        return combined_df

    def load_dem_water_embeddings(self, combined_df: pd.DataFrame, name: str = None) -> pd.DataFrame:
        """
        Load DEM & water body embeddings.
        
        Args:
            combined_df (pd.DataFrame): Existing combined dataframe
            name (str): Name identifier for the embedding model
            
        Returns:
            pd.DataFrame: Dataframe with DEM-water embeddings
            
        Raises:
            ValueError: If name parameter is not provided
        """
        if name is None:
            raise ValueError("Invalid name parameter, please specify embedding model name")
        
        path = f"/path/to/embeddings/{name}"
        logger.info(f"Loading DEM&Water embedding data: {path}")

        # Check if pre-processed parquet file exists
        parquet_path = f'{path}/embeddings_{name}.parquet'
        if os.path.exists(parquet_path):
            dem_embedding_df = pd.read_parquet(parquet_path)
            if combined_df.shape[0] == 0:
                return dem_embedding_df
            else:
                combined_df = combined_df.merge(dem_embedding_df, left_on='id', right_on='id', how='inner')
                return combined_df

        # Load from numpy and csv files
        embeddings = np.load(f'{path}/embeddings_{name}_0.npy')
        df = pd.read_csv(f'{path}/embeddings_{name}_0.csv')

        # Remove existing DEM embedding columns
        dem_emb_cols = [col for col in combined_df.columns if col.startswith("D")]
        if dem_emb_cols:
            combined_df.drop(columns=dem_emb_cols, inplace=True, errors='ignore')

        embedding_df = pd.DataFrame(embeddings)
        df.drop(columns=['mean', 'std'], inplace=True)
        dem_embedding_df = pd.concat([df, embedding_df], axis=1)

        # Rename embedding columns with 'D' prefix
        for i in range(0, 256):
            dem_embedding_df.rename(columns={i: f'D{i}'}, inplace=True)
        
        # Save processed data as parquet for future use
        if not os.path.exists(parquet_path):
            dem_embedding_df.to_parquet(parquet_path)

        if combined_df.shape[0] == 0:
            return dem_embedding_df
        else:
            combined_df = combined_df.merge(dem_embedding_df, left_on='id', right_on='id', how='inner')
            return combined_df

    def load_climate_embeddings(self, combined_df: pd.DataFrame, data_type: str, year: int) -> pd.DataFrame:
        """
        Load climate embedding data.
        
        Args:
            combined_df (pd.DataFrame): Existing combined dataframe
            data_type (str): Data type ('pref', 'county', or 'eu')
            year (int): Year for climate embeddings
            
        Returns:
            pd.DataFrame: Dataframe with climate embeddings
        """

        if data_type == 'pref':
            path = f"/path/to/historical/climate/embeddings"
            name = f"embedding_name"
        elif data_type == 'eu':
            path = f"/path/to/historical/climate/embeddings"
            name = f"embedding_name"

        embeddings = np.load(f'{path}/embeddings_{name}_0.npy')
        df = pd.read_csv(f'{path}/embeddings_{name}_0.csv')

        # Remove existing climate embedding columns
        clim_emb_cols = [col for col in combined_df.columns if col.startswith("C")]
        if clim_emb_cols:
            combined_df.drop(columns=clim_emb_cols, inplace=True, errors='ignore')

        embedding_df = pd.DataFrame(embeddings)
        df.drop(columns=['mean', 'std'], inplace=True)
        clim_embedding_df = pd.concat([df, embedding_df], axis=1)

        # Rename embedding columns with 'C' prefix
        for i in range(0, 128):
            clim_embedding_df.rename(columns={i: f'C{i}'}, inplace=True)
        
        # Save processed data as parquet for future use
        parquet_path = f'{path}/embeddings_{name}.parquet'
        if not os.path.exists(parquet_path):
            clim_embedding_df.to_parquet(parquet_path)

        combined_df = combined_df.merge(clim_embedding_df, left_on='id', right_on='id', how='inner')

        return combined_df

    def check_data_files(self) -> bool:
        """
        Check if all required data files exist.
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        for path in self.config.data_paths.values():
            full_path = self.base_path / path
            if not full_path.exists():
                logger.error(f"Data file not found: {full_path}")
                return False
        return True