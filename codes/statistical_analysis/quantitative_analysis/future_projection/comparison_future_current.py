
"""
This script compares predictions of current and future urban potential (take SSP585 as an example)
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

def compare_predictions(
    dem_water_emb_path: str,
    current_clim_path: str,
    future_clim_path: str,
    output_dir: str
):
    """
    Compare current and future predictions
    """
    # create out directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # load current and future embeddings, and check if they are consistent
    logger.info("load dataset")
    logger.info(f"load DEM&Water representation data...")
    dem_water_embedding_df = pd.read_parquet(dem_water_emb_path)
    # load current and future climate data
    climate_current = pd.read_parquet(current_clim_path)
    climate_future = pd.read_parquet(future_clim_path)
    data_current = climate_current.merge(dem_water_embedding_df, on='id', how='inner')
    data_future = climate_future.merge(dem_water_embedding_df, on='id', how='inner')
    id_list = data_current['id'].copy(deep=True)
    # make sure use consistent ID
    data_future = data_future.merge(id_list, on='id', how='inner')
    data_current = data_current.merge(id_list, on='id', how='inner')
    # reset index and delete id column for following prediction
    data_current_for_pred = data_current.drop(columns=['id']).reset_index(drop=True)
    data_future_for_pred = data_future.drop(columns=['id']).reset_index(drop=True)
    if len(data_current_for_pred) != len(data_future_for_pred):
        print(len(data_current_for_pred), len(data_future_for_pred))
        raise ValueError("Current and future dataset have different number of rows!")
    
    # save all models' predictions
    all_current_probs = []
    all_future_probs = []

    # specify prediction pickle file path
    all_current_probs_path = "/path/to/current/prediction/pickle-file/directory/result_300.pickle"
    pickle_ssp = ssp
    all_future_probs_path = "/path/to/future/prediction/pickle-file/directory/result_300.pickle"


    # read pickle file
    with open(all_current_probs_path, "rb") as f:
        all_current_probs = pickle.load(f)
        print(type(all_current_probs))
        print(len(all_current_probs[0]))
    with open(all_future_probs_path, "rb") as f:
        all_future_probs = pickle.load(f)
        print(type(all_future_probs))
        print(len(all_future_probs[0]))
    
    # Calculate mean predicted probabilities
    current_prob_mean = np.mean(all_current_probs, axis=0)
    future_prob_mean = np.mean(all_future_probs, axis=0)
    
    # Calculate standard deviation of predicted probabilities
    current_prob_std = np.std(all_current_probs, axis=0)
    future_prob_std = np.std(all_future_probs, axis=0)
    
    # Create comparison dataframe
    logger.info("生成比较结果")
    comparison_df = pd.DataFrame({
        'id': id_list.tolist(),
        'current_probability_mean': current_prob_mean,
        'current_probability_std': current_prob_std,
        'future_probability_mean': future_prob_mean,
        'future_probability_std': future_prob_std,
        'probability_change_mean': future_prob_mean - current_prob_mean,
        'probability_change_std': np.sqrt(current_prob_std**2 + future_prob_std**2)
    })
    
    # save the result of comparison
    comparison_df.to_csv(output_path / 'prediction_comparison_ensemble.csv', index=True)
    
    # generate probability changes distribution plot
    plt.figure(figsize=(10, 6))
    prob_changes = future_prob_mean - current_prob_mean
    plt.hist(prob_changes, bins=50, edgecolor='black')
    plt.title('Distribution of Mean Probability Changes (Ensemble Prediction)')
    plt.xlabel('Probability Change (Future - Current)')
    plt.ylabel('Count')
    plt.savefig(output_path / 'probability_changes_ensemble.png')
    plt.close()
    
    # generate report
    report = [
        "# Comparison result of ensemble predictions\n",
        f"Number of models used: {len(all_current_probs)}",
        f"Total number of samples: {len(comparison_df)}",
        "\nProbability change statistics:",
        f"Mean change: {prob_changes.mean():.4f}",
        f"Maximum increase: {prob_changes.max():.4f}",
        f"Maximum decrease: {prob_changes.min():.4f}",
        f"Overall standard deviation: {prob_changes.std():.4f}",
        f"\nPrediction stability statistics: ",
        f"Mean standard deviation (current): {current_prob_std.mean():.4f}",
        f"Mean standard deviation (future): {future_prob_std.mean():.4f}"
    ]
    
    with open(output_path / 'comparison_report_ensemble.md', 'w') as f:
        f.write('\n'.join(report))
    
    # print main results
    logger.info(f"\n{'='*50}")
    logger.info("集成比较结果摘要：")
    logger.info(f"Number of models used: {len(all_current_probs)}")
    logger.info(f"Total number of samples: {len(comparison_df)}")
    logger.info(f"Mean change: {prob_changes.mean():.4f}")
    logger.info(f"Mean standard deviation (current/future): {current_prob_std.mean():.4f}/{future_prob_std.mean():.4f}")
    logger.info(f"Detailed results are saved to: {output_path}")
    logger.info('='*50)

    return comparison_df

# set paths
dem_water_emb_path = "/path/to/dem&water/embeddings/merged_embeddings_dem_water_intID.parquet"
current_clim_path = "/path/to/current/climate/embeddings/merged_embeddings_current_clim_V2_intID.parquet"
ssp = '585' # [126, 370, 585]
future_clim_path = "/path/to/future/climate/embeddings/merged_embeddings_future_clim_SSP585_intID.parquet"
print('ssp:', ssp)
print('future_clim_path:', future_clim_path)

output_dir = "/path/to/out/directory/{time}_comparision_result_SSP{ssp}_n300_d4_CNprefEUtrain_GLOBALpred".format(time=datetime.now().strftime("%Y%m%d_%H%M%S"), ssp=ssp)

comparison_df=compare_predictions(
    dem_water_emb_path=dem_water_emb_path,
    current_clim_path=current_clim_path,
    future_clim_path=future_clim_path,
    output_dir=output_dir
)

print(comparison_df.head())

# generate comparison result shapefile
import geopandas as gpd
# global grid shapfile
gdf = gpd.read_file('/path/to/0.05-deg-grid/shapefile/directory/005deg_global_intersect_land_nonwater.shp')
gdf.rename(columns={'ID': 'id'}, inplace=True)
gdf = gdf.merge(comparison_df, left_on='id', right_on='id', how='inner')
gdf.to_file(output_dir + '/result_n300_d4_CNprefEUtrain_GLOBALpred_' + ssp + '_300.shp')
print('Shapefile saved!')