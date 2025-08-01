"""
This script uses trained XGBoost model to predict current urban potential
The input data are DEM and current climate embeddings
Author: Weiyu Zhang
Date: 2025-02-19
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import logging
from datetime import datetime
from typing import List, Tuple, Dict
from src.utils import PreprocessPipeline
from src.models import XGBoostTrainer, DataSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from glob import glob
import os
import pickle

# log configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_features(model_dir):
    """
    load single model and its corresponding feature information
    
    Args:
        model_dir: model directory
    Returns:
        (model, feature_info) 
    """
    model_path = os.path.join(model_dir, "model.joblib")
    feature_path = os.path.join(model_dir, "feature_info.csv")
    
    model = joblib.load(model_path)
    
    
    return model, feature_path


def standardize_features(new_data, feature_info_path):
    # load feature information
    feature_info = pd.read_csv(feature_info_path, names=['feature_paths', 'mean', 'std'], header=0)
    
    ordered_data = new_data[feature_info['feature_paths']]
    
    # standardization
    mean = np.array(feature_info['mean'])
    std = np.array(feature_info['std'])
    scaled_data = (ordered_data - mean) / std
    
    return scaled_data

def predict_with_model(model, feature_info_path, current_data):
    """
    use one model to predict
    
    Args:
        model: a trained model
        feature_info: the corresponding feature information
        current_data: current data
    Returns:
        current_prob: current urban potential
    """
    # standardization
    current_standardized = standardize_features(current_data, feature_info_path)
    
    # make sure the order of features is right
    feature_names = model.feature_names_in_
    current_standardized = current_standardized.reindex(columns=feature_names)
    
    # predict
    current_prob = model.predict_proba(current_standardized)[:, 1]
    
    return current_prob

def predict_models(
    model_base_path: str,
    dem_water_emb_path: str,
    current_clim_path: str,
    output_dir: str
):
    """
    make predictions for all models
    """
    # create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # load data
    logger.info("load dataset")
    logger.info(f"load DEM&Water representation data...")
    dem_water_embedding_df = pd.read_parquet(dem_water_emb_path)
    
    # load current climate data
    climate_current = pd.read_parquet(current_clim_path)
    
    # merge data
    data_current = climate_current.merge(dem_water_embedding_df, on='id', how='inner')
    id_list = data_current['id'].copy(deep=True)
    
    # make sure use consistent ID
    data_current = data_current.merge(id_list, on='id', how='inner')
    
    # reset index and delete id column for following prediction
    data_current_for_pred = data_current.drop(columns=['id']).reset_index(drop=True)
    
    # save all models' predictions
    all_current_probs = []
    
    # traverse all models
    for i in range(0, 300):
        model_dir = os.path.join(model_base_path, f"model_{i}")
        try:
            # load model and features
            model, feature_info_path = load_model_and_features(model_dir)
            
            # predict with the model
            current_prob = predict_with_model(
                model, feature_info_path, data_current_for_pred
            )
            
            all_current_probs.append(current_prob)
            
            logger.info(f"Get prediction of model {i + 1}")
            
            if i == 0:
                with open(output_dir + '/backup_' + str(i+1) + '.pickle', 'wb') as file:
                    pickle.dump(all_current_probs, file, protocol=pickle.HIGHEST_PROTOCOL)
            if (i + 1) % 50 == 0:
                with open(output_dir + '/backup_' + str(i+1) + '.pickle', 'wb') as file:
                    pickle.dump(all_current_probs, file, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Predictions of model {i + 1 - 10} - {i + 1} are saved!")
                
        except Exception as e:
            logger.error(f"Failed to predict with model {i}: {str(e)}")
    
    with open(output_dir + '/result_' + str(i+1) + '.pickle', 'wb') as file:
        pickle.dump(all_current_probs, file, protocol=pickle.HIGHEST_PROTOCOL)

# Set models' path
model_base_path = "/path/to/saved/model"
# Set embeddings' paths
dem_water_emb_path = "/path/to/dem&water/embeddings/merged_embeddings_dem_water_intID.parquet"
current_clim_path = "/path/to/current/climate/embeddings/merged_embeddings_current_clim_V2_intID.parquet"

# output path
output_dir = "/path/to/out/directory/{}_predict_current_n300_d4_CNprefEUtrain_GLOBALpred".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

comparison_df=predict_models(
    model_base_path=model_base_path,
    dem_water_emb_path=dem_water_emb_path,
    current_clim_path=current_clim_path,
    output_dir=output_dir
)