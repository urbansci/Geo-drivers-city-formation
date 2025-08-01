# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: City-based data splitter for geographical dataset splitting. Because we assign positive samples
             by buffering city points, which induce severe spatial autocorrelation, we need to split the dataset
             by city names to ensure that training and test sets do not share the same city (neighbouring grids).
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitterByCity:
    """Geographical dataset splitter."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the city-based data splitter.
        
        Args:
            random_state (int): Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def split_by_cities(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset based on city names.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target variable
            test_ratio (float): Test set ratio
            
        Returns:
            Tuple: x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # 1. Identify positive and negative sample grids
        data_with_labels = pd.DataFrame({
            'id': X['id'], 
            'city': X['city'], 
            'label': y
        })

        # Save original index correspondence
        original_train_indices = x_train.index
        original_test_indices = x_test.index
        
        # Reset indices for safe processing
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Get valid city list
        unique_cities = data_with_labels[data_with_labels['label'] == 1]['city'].unique()
        
        # Create boolean mask instead of index list to avoid index issues
        if 'city_id' in x_test.columns:
            train_city_mask = x_test['city_id'] > -1
        else:
            train_city_mask = pd.Series([False] * len(x_test))
        
        # Use mask for data movement
        if train_city_mask.any():
            x_train = pd.concat([x_train, x_test[train_city_mask]], ignore_index=True)
            y_train = pd.concat([y_train, y_test[train_city_mask]], ignore_index=True)
            
            x_test = x_test[~train_city_mask].reset_index(drop=True)
            y_test = y_test[~train_city_mask].reset_index(drop=True)
        
        # Randomly sample cities
        num_test_cities = int(len(unique_cities) * test_ratio)
        test_cities = np.random.choice(unique_cities, num_test_cities, replace=False)
        
        # Use boolean mask to select test city data
        test_mask = x_train['city'].isin(test_cities)
        
        if test_mask.any():
            x_test = pd.concat([x_test, x_train[test_mask]], ignore_index=True)
            y_test = pd.concat([y_test, y_train[test_mask]], ignore_index=True)
            
            x_train = x_train[~test_mask].reset_index(drop=True)
            y_train = y_train[~test_mask].reset_index(drop=True)
        
        # Validate dataset integrity
        assert len(x_train) == len(y_train), "Training set length mismatch"
        assert len(x_test) == len(y_test), "Test set length mismatch"
        assert len(x_train) + len(x_test) == len(original_train_indices) + len(original_test_indices), \
               "Total number of samples changed"
        
        return x_train, x_test, y_train, y_test