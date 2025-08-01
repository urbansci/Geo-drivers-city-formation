# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: Feature processing module for geographical city formation analysis.
             Handles feature categorization, generation, and preprocessing including standardization.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FeatureConfig


class FeatureProcessor:
    """
    Feature processing class responsible for feature categorization, generation, and preprocessing.
    
    This class handles the complete feature processing workflow including:
    - Feature categorization (DEM embeddings, climate embeddings, attributes)
    - Feature matrix generation
    - Data standardization and preprocessing
    - Temporal data filtering
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize the feature processor.
        
        Args:
            config (FeatureConfig): Configuration object containing feature processing parameters
        """
        self.config = config
        self.scaler = StandardScaler()
    
    def categorize_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Categorize features into DEM embeddings, climate embeddings, and attribute features.
        
        Args:
            X (pd.DataFrame): Input feature dataframe
            
        Returns:
            Tuple[List[str], List[str], List[str], List[str], List[str]]: 
                - DEM embedding features
                - Climate embedding features  
                - Manual attribute features
                - DEM-water attribute features
                - Climate attribute features
        """
        dem_emb_list = []
        clim_emb_list = []
        attr_list = []
        
        # Categorize features based on column prefixes
        for col in X.columns:
            if col in self.config.cols_of_label:
                continue
            elif col in ['d_road', 'd_hub']:  # Skip road and hub distance features
                continue
            elif col.startswith("D"):  # DEM embedding features
                dem_emb_list.append(col)
            elif col.startswith("C"):  # Climate embedding features
                clim_emb_list.append(col)
            else:  # Other attribute features
                attr_list.append(col)
        
        # Define predefined attribute lists
        dem_water_attr_list = [
            "d_lake", "d_sea", "mean", "std", "d_river"
        ]
        
        clim_attr_list = [
            "bio_1", "bio_12", "bio_15", "bio_18", "bio_19", 
            "bio_4", "bio_7", "bio_8", "bio_9"
        ]
        
        attr_list = [
            'bio_1', 'bio_12', 'bio_15', 'bio_18', 'bio_19', 'bio_4', 'bio_7', 'bio_8', 'bio_9',
            'd_river', 'd_lake', 'd_sea', 'mean', 'std'
        ]
        
        return dem_emb_list, clim_emb_list, attr_list, dem_water_attr_list, clim_attr_list
    
    def generate_features(self, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """
        Generate feature matrix and labels from combined dataframe.
        
        Args:
            combined_df (pd.DataFrame): Combined input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
                - X: Feature matrix
                - y: Target labels
                - feature_groups: Dictionary mapping feature group names to feature lists
                
        Raises:
            ValueError: If 'is_city' column is missing from the dataset
        """
        # Check for required target column
        if 'is_city' not in combined_df.columns:
            raise ValueError("Missing 'is_city' label column in dataset")

        # Determine columns to keep
        cols_to_consider = combined_df.columns.difference(self.config.cols_to_drop)
        
        # Generate feature matrix and labels
        X = combined_df[cols_to_consider]
        y = combined_df['is_city']
        
        # Categorize features
        dem_emb_list, clim_emb_list, attr_list, dem_water_attr_list, clim_attr_list = self.categorize_features(X)
        
        # Create feature groups dictionary
        feature_groups = {
            'DEM&Water embedding': dem_emb_list,
            'Climate embedding': clim_emb_list,
            'Attribute': attr_list,
            'DEM&Water attribute': dem_water_attr_list,
            'Climate attribute': clim_attr_list,
            'All embedding': dem_emb_list + clim_emb_list,
            'All embedding&Agriculture': dem_emb_list + clim_emb_list + ['agri']
        }
        
        return X, y, feature_groups
    
    def prepare_features(
        self, 
        x_train: pd.DataFrame, 
        x_test: pd.DataFrame, 
        cols_to_drop: List[str] = None,
        standardize: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
        """
        Prepare features by removing unnecessary columns and applying standardization.
        
        Args:
            x_train (pd.DataFrame): Training feature dataframe
            x_test (pd.DataFrame): Test feature dataframe
            cols_to_drop (List[str], optional): Columns to drop from the datasets
            standardize (bool): Whether to apply standardization. Defaults to True.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]:
                - Processed training features
                - Processed test features  
                - Feature information dictionary (empty if not standardized)
                
        Raises:
            AssertionError: If training and test sets have inconsistent features
        """
        # Clean up columns to drop - remove non-existent columns
        if cols_to_drop:
            cols_to_drop = [col for col in cols_to_drop if col in x_train.columns]
            missing_cols = [col for col in cols_to_drop if col not in x_train.columns]
            if missing_cols:
                print(f"Columns not found in dataset (removed): {missing_cols}")
        
        # Drop unnecessary columns
        if cols_to_drop:
            x_train = x_train.drop(columns=cols_to_drop)
            x_test = x_test.drop(columns=cols_to_drop)
            
        # Ensure feature consistency between train and test sets
        assert set(x_train.columns) == set(x_test.columns), \
            "Training and test sets have inconsistent features"
        
        # Return early if no standardization requested
        if not standardize:
            return x_train, x_test, {}

        # Apply standardization
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Create feature information dictionary
        feature_info = {
            'feature_names': x_train.columns.tolist(),
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist()
        }

        # Convert back to DataFrames with original column names
        x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
        x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)
        
        return x_train_scaled, x_test_scaled, feature_info
    
    def process_all(
        self, 
        combined_df: pd.DataFrame,
        test_indices: List[int],
        additional_cols_to_drop: List[str] = None,
        standardize: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
        """
        Execute complete feature processing pipeline.
        
        Args:
            combined_df (pd.DataFrame): Input combined dataframe
            test_indices (List[int]): Indices for test set samples
            additional_cols_to_drop (List[str], optional): Additional columns to drop
            standardize (bool): Whether to apply standardization. Defaults to True.
            
        Returns:
            Tuple: Processed training features, test features, training labels, 
                   test labels, and feature groups
        """
        # Generate features and labels
        X, y, feature_groups = self.generate_features(combined_df)
        
        # Split into train and test sets
        train_mask = ~X.index.isin(test_indices)
        x_train = X[train_mask]
        x_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        
        # Prepare columns to drop
        cols_to_drop = self.config.cols_of_label.copy()
        if additional_cols_to_drop:
            cols_to_drop.extend(additional_cols_to_drop)
            
        # Process features
        x_train_scaled, x_test_scaled, _ = self.prepare_features(
            x_train, x_test, cols_to_drop=cols_to_drop, standardize=standardize
        )
        
        return x_train_scaled, x_test_scaled, y_train, y_test, feature_groups
    
    def filter_data(
        self,
        df: pd.DataFrame,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Filter data based on temporal range for historical analysis.
        
        This method handles temporal filtering by:
        1. Removing cities that were founded before the start year
        2. Marking cities founded after the end year as non-cities
        
        Args:
            df (pd.DataFrame): Input dataframe with 'birth' and 'is_city' columns
            start_year (int): Starting year for analysis period
            end_year (int): Ending year for analysis period
            
        Returns:
            pd.DataFrame: Filtered dataframe with temporal constraints applied
            
        Raises:
            KeyError: If required columns ('birth', 'is_city') are missing
        """
        # Validate required columns
        required_cols = ['birth', 'is_city']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Required columns missing: {missing_cols}")
        
        # Create copy to avoid modifying original data
        df_filtered = df.copy()
        
        # Remove cities founded before start_year
        pre_start_mask = (df_filtered['birth'] < start_year) & (df_filtered['is_city'] == 1)
        df_filtered = df_filtered[~pre_start_mask]
        
        # Mark cities founded after end_year as non-cities
        post_end_mask = (df_filtered['birth'] > end_year) & (df_filtered['is_city'] == 1)
        df_filtered.loc[post_end_mask, 'is_city'] = 0
        
        return df_filtered
    
    def get_feature_statistics(self, X: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate comprehensive statistics for features.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            Dict[str, any]: Dictionary containing feature statistics
        """
        stats = {
            'total_features': X.shape[1],
            'total_samples': X.shape[0],
            'missing_values': X.isnull().sum().to_dict(),
            'feature_types': X.dtypes.to_dict(),
            'numeric_summary': X.describe().to_dict()
        }
        
        # Add feature categorization statistics
        dem_emb, clim_emb, attr, dem_water_attr, clim_attr = self.categorize_features(X)
        stats['feature_counts'] = {
            'dem_embeddings': len(dem_emb),
            'climate_embeddings': len(clim_emb),
            'attributes': len(attr),
            'dem_water_attributes': len(dem_water_attr),
            'climate_attributes': len(clim_attr)
        }
        
        return stats


# Usage example
if __name__ == "__main__":
    # Example usage of FeatureProcessor
    from .config import FeatureConfig
    
    # Initialize configuration and processor
    config = FeatureConfig()
    processor = FeatureProcessor(config)
    
    # Example dataframe (replace with actual data)
    sample_data = pd.DataFrame({
        'D001': np.random.randn(100),  # DEM embedding
        'C001': np.random.randn(100),  # Climate embedding
        'bio_1': np.random.randn(100),  # Attribute
        'is_city': np.random.choice([0, 1], 100),  # Target
        'city': ['city_' + str(i) for i in range(100)]  # City names
    })
    
    # Generate features
    X, y, feature_groups = processor.generate_features(sample_data)
    print("Feature groups:", feature_groups)
    
    # Get feature statistics
    stats = processor.get_feature_statistics(X)
    print("Feature statistics:", stats)