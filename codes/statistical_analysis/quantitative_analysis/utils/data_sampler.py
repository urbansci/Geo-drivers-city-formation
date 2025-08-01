# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: Data sampling module for handling class imbalance in geographical city formation analysis.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


class DataSampler:
    """
    Data sampler for handling class imbalance using oversampling or undersampling strategies.
    """
    
    def __init__(self, random_state: int = 42, sampling_strategy: float = 0.1):
        """
        Initialize the data sampler.
        
        Args:
            random_state (int): Random seed for reproducible results
            sampling_strategy (float): Sampling strategy for undersampling
        """
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
        self.under_sampler = RandomUnderSampler(
            random_state=random_state, 
            sampling_strategy=sampling_strategy
        )
        
    def prepare_balanced_data(
        self, 
        processed_data: Dict,
        sampling_strategy: str = 'under',
        second_nature: List[str] = None,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
        """
        Prepare balanced feature group data.

        Args:
            processed_data (Dict): Processed data returned by pipeline.run()
            sampling_strategy (str): 'under' or 'over', choose undersampling or oversampling
            second_nature (List[str], optional): Additional second nature features to include
            
        Returns:
            Tuple containing:
                - Dict: Balanced data for each feature group
                  {
                      'Attribute': (X_train_balanced, X_test),
                      'All embedding': (X_train_balanced, X_test),
                      'DEM&Water embedding': (X_train_balanced, X_test),
                      'Climate embedding': (X_train_balanced, X_test)
                  }
                - y_balanced: Balanced training labels
                - y_test: Original test labels
        """
        x_train = processed_data['x_train']
        x_test = processed_data['x_test']
        y_train = processed_data['y_train']
        feature_groups = processed_data['feature_groups']
        
        # First apply sampling to the entire training set
        if sampling_strategy == 'under':
            X_balanced, y_balanced = self.under_sampler.fit_resample(x_train, y_train)
        else:
            X_balanced, y_balanced = self.smote.fit_resample(x_train, y_train)
        
        # Prepare feature combinations and split data
        combinations = {
            "Attribute": feature_groups['attributes'],
            "All embedding": feature_groups['dem_embeddings'] + feature_groups['climate_embeddings'],
            "DEM&Water embedding": feature_groups['dem_embeddings'],
            "Climate embedding": feature_groups['climate_embeddings'],
            "DEM&Water attribute": feature_groups['dem_water_attributes'],
            "Climate attribute": feature_groups['climate_attributes'],
            "Terrain-Water&Clim": feature_groups['dem_embeddings'] + feature_groups['climate_embeddings'],
            "Agriculture": ['pre', 'post']
        }

        # Add agricultural features if available
        if "agri" in x_train.columns:
            combinations['All embedding'] += ['agri']
        else:
            cols_to_add_agri = [col for col in x_train.columns if col in ['pre', 'post']]
            combinations['All embedding'] += cols_to_add_agri

        # Add second nature features if provided
        if second_nature is not None:
            combinations['Attribute'] += second_nature
            combinations['All embedding'] += second_nature
            combinations['DEM&Water embedding'] += second_nature
            combinations['Climate attribute'] += second_nature
            combinations['DEM&Water attribute'] += second_nature
            combinations['Climate embedding'] += second_nature
            
        balanced_data = {}
        for name, cols in combinations.items():
            logger.info(f"Processing feature group: {name}")
            X_train_group = X_balanced.loc[:, cols].copy()
            X_test_group = x_test.loc[:, cols].copy()
            balanced_data[name] = (X_train_group, X_test_group)
            
        return balanced_data, y_balanced, processed_data['y_test']