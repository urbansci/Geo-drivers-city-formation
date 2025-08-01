# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: Preprocessing pipeline for geographical city formation analysis. 
             Integrates data loading, feature processing, and dataset splitting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data_loader import GeoDataLoader
from .data_splitter_cityname import DataSplitterByCity
from .feature_processor import FeatureProcessor

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """
    Preprocessing pipeline that integrates data loading, feature processing, and dataset splitting.
    
    This pipeline handles the complete preprocessing workflow for geographical city formation analysis,
    including feature extraction from embeddings, city-based data splitting, and standardization.
    """
    
    def __init__(
        self,
        base_path: str,
        model_name: str,
        random_state: int = 42,
        test_ratio: float = 0.2,
        data_file_name: str = ""
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            base_path (str): Root directory path for data
            model_name (str): Model name for locating embedding data files
            data_file_name (str): Name of the data file to load
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            test_ratio (float, optional): Proportion of test set. Defaults to 0.2.
        """
        self.base_path = Path(base_path)
        self.model_name = model_name
        self.data_file_name = data_file_name
        self.random_state = random_state
        self.test_ratio = test_ratio
        
        # Initialize components
        self.config = FeatureConfig()
        self.data_loader = GeoDataLoader(
            base_path=base_path, 
            config=self.config,
            data_file_name=data_file_name
        )
        
        self.feature_processor = FeatureProcessor(config=self.config)
        self.splitter = DataSplitterByCity(random_state=random_state)
        
        # Store processed data
        self.processed_data = {}
        
    def run(self, standardize: bool = True) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Steps:
        1. Load data
        2. Generate features: separate X and y, create feature groups
        3. Split dataset: city-based train/test split
        4. Process features: standardization
        
        Args:
            standardize (bool): Whether to standardize features. Defaults to True.
            
        Returns:
            Dict[str, Any]: Dictionary containing processed data with keys:
                - x_train: Training set features
                - x_test: Test set features  
                - y_train: Training set labels
                - y_test: Test set labels
                - feature_groups: Feature grouping information
                - data_info: Data processing metadata
        """
        try:
            # Load data
            logger.info(f"Loading data: {self.model_name}")
            combined_df = self.data_loader.load_data(self.model_name)
            logger.info(f"Data loading completed, shape: {combined_df.shape}")
            
            return self._process_data(combined_df, standardize)
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def run_with_loaded_data(self, combined_df: pd.DataFrame, standardize: bool = True) -> Dict[str, Any]:
        """
        Run preprocessing pipeline with pre-loaded data (excluding data loading step).
        
        Steps:
        1. Generate features: separate X and y, create feature groups
        2. Split dataset: city-based train/test split  
        3. Process features: standardization
        
        Args:
            combined_df (pd.DataFrame): Pre-loaded combined dataframe
            standardize (bool): Whether to standardize features. Defaults to True.
            
        Returns:
            Dict[str, Any]: Dictionary containing processed data with same structure as run()
        """
        try:
            logger.info(f"Processing pre-loaded data, shape: {combined_df.shape}")
            return self._process_data(combined_df, standardize)
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def _process_data(self, combined_df: pd.DataFrame, standardize: bool) -> Dict[str, Any]:
        """
        Internal method to process data through the pipeline.
        
        Args:
            combined_df (pd.DataFrame): Input dataframe
            standardize (bool): Whether to standardize features
            
        Returns:
            Dict[str, Any]: Processed data dictionary
        """
        # Generate features
        logger.info("Generating features...")
        X, y, feature_groups = self.feature_processor.generate_features(combined_df)
        logger.info(f"Feature generation completed, number of features: {X.shape[1]}")
        
        # Split dataset by cities
        logger.info(f"Splitting dataset by city names, test ratio: {self.test_ratio}")
        x_train, x_test, y_train, y_test = self.splitter.split_by_cities(
            X, y, test_ratio=self.test_ratio
        )
        
        logger.info(f"Training set shape: {x_train.shape}")
        logger.info(f"Test set shape: {x_test.shape}")
        
        # Process features (standardization, etc.)
        logger.info("Processing features...")
        x_train_scaled, x_test_scaled, feature_info = self.feature_processor.prepare_features(
            x_train, x_test, 
            cols_to_drop=self.config.cols_of_label,
            standardize=standardize
        )
        
        # Store processed data
        self.processed_data = {
            'feature_info': feature_info,
            'x_train': x_train_scaled,
            'x_test': x_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_groups': feature_groups,
            'data_info': {
                'model_name': self.model_name,
                'test_ratio': self.test_ratio,
                'random_state': self.random_state,
                'train_cities': x_train['city'].nunique() if 'city' in x_train.columns else 0,
                'test_cities': x_test['city'].nunique() if 'city' in x_test.columns else 0,
                'total_features': x_train_scaled.shape[1],
                'standardized': standardize
            }
        }
        
        # Log processing results
        self._log_processing_results()
        
        return self.processed_data

    def _log_processing_results(self):
        """Log detailed information about processing results."""
        if not self.processed_data:
            logger.warning("No processing results available")
            return
            
        logger.info("\nData processing completed! Dataset information:")
        info = self.processed_data['data_info']
        logger.info(f"Model name: {info['model_name']}")
        logger.info(f"Training cities: {info['train_cities']}")
        logger.info(f"Test cities: {info['test_cities']}")
        logger.info(f"Total features: {info['total_features']}")
        logger.info(f"Standardized: {info['standardized']}")
        
        logger.info(f"\nTraining set shape: {self.processed_data['x_train'].shape}")
        logger.info(f"Test set shape: {self.processed_data['x_test'].shape}")
        
        # Log label distribution
        train_labels = pd.Series(self.processed_data['y_train']).value_counts()
        test_labels = pd.Series(self.processed_data['y_test']).value_counts()
        logger.info(f"Training set label distribution:\n{train_labels}")
        logger.info(f"Test set label distribution:\n{test_labels}")
        
        # Log feature group information
        logger.info("\nFeature group information:")
        for group_name, features in self.processed_data['feature_groups'].items():
            logger.info(f"{group_name}: {len(features)} features")
            # Show sample features (first 3)
            sample_features = features[:3] if len(features) >= 3 else features
            logger.info(f"Sample features: {sample_features}")
    
    def get_feature_groups(self) -> Dict[str, int]:
        """
        Get the number of features in each feature group.
        
        Returns:
            Dict[str, int]: Feature groups and their feature counts
            
        Raises:
            ValueError: If preprocessing pipeline hasn't been run yet
        """
        if not self.processed_data:
            raise ValueError("Please run the preprocessing pipeline first")
            
        return {
            group: len(features) 
            for group, features in self.processed_data['feature_groups'].items()
        }

    def get_feature_names_by_group(self, group_name: str) -> list:
        """
        Get feature names for a specific group.
        
        Args:
            group_name (str): Name of the feature group
            
        Returns:
            list: List of feature names in the specified group
            
        Raises:
            ValueError: If preprocessing hasn't been run or group doesn't exist
        """
        if not self.processed_data:
            raise ValueError("Please run the preprocessing pipeline first")
        
        if group_name not in self.processed_data['feature_groups']:
            available_groups = list(self.processed_data['feature_groups'].keys())
            raise ValueError(f"Group '{group_name}' not found. Available groups: {available_groups}")
            
        return self.processed_data['feature_groups'][group_name]

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the processed data.
        
        Returns:
            Dict[str, Any]: Summary statistics and information
        """
        if not self.processed_data:
            raise ValueError("Please run the preprocessing pipeline first")
        
        train_data = self.processed_data['x_train']
        test_data = self.processed_data['x_test']
        
        return {
            'dataset_info': self.processed_data['data_info'],
            'feature_groups': self.get_feature_groups(),
            'data_shapes': {
                'train_features': train_data.shape,
                'test_features': test_data.shape,
                'train_labels': len(self.processed_data['y_train']),
                'test_labels': len(self.processed_data['y_test'])
            },
            'class_distribution': {
                'train': pd.Series(self.processed_data['y_train']).value_counts().to_dict(),
                'test': pd.Series(self.processed_data['y_test']).value_counts().to_dict()
            }
        }


# Usage example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = PreprocessPipeline(
        base_path="/path/to/data",
        model_name="GEOS-213-CN-augmenteddem&water-256-499epoch",
        data_file_name="combined_data.parquet",
        random_state=42,
        test_ratio=0.2
    )
    
    # Run preprocessing pipeline
    processed_data = pipeline.run(standardize=True)
    
    # Get feature group information
    feature_groups = pipeline.get_feature_groups()
    print("Feature groups:", feature_groups)
    
    # Get comprehensive summary
    summary = pipeline.get_data_summary()
    print("Data summary:", summary)