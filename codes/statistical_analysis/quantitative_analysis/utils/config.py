# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: Configuration classes for geographical city formation analysis preprocessing and feature handling.
"""


class PreprocessingConfig:
    """Configuration parameters for data preprocessing pipeline."""
    
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    UNDERSAMPLE_RATIO = 0.3
    COLS_TO_DROP = [
        'ID', 'geometry', 'id_1', 'is_sea', 'kmeans_clu', 
        'Shape_Leng', 'is_city', 'mean_y', 'std_x'
    ]
    COLS_OF_LABEL = ["city_id", "city", "birth", "id", "l_grid_id"]


class FeatureConfig:
    """Feature configuration class for managing column definitions and data paths."""
    
    def __init__(self):
        # Base columns to drop during preprocessing
        self.cols_to_drop = [
            'ID', 'geometry', 'id_1', 'is_sea', 'kmeans_clu',
            'Shape_Leng', 'is_city', 'mean_y', 'std_x'
        ]
        
        # Label-related columns
        self.cols_of_label = [
            "city_id", "city", "birth", "id", "l_grid_id", "year", "d_ml_river"
        ]
        
        # Data path configuration
        self.data_paths = {
            'pref_agri': "/path/to/Chinese_pref/agri_gaez.parquet",
            'county_agri': "/path/to/Chinese_county/agri_gaez.parquet",
            'pref_climate': "/path/to/Chinese_pref/paleo_clim_Y1900.parquet",
            'county_climate': "/path/to/Chinese_county/paleo_clim_Y1900.parquet"
        }