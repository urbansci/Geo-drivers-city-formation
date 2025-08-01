"""
This script aims to train the XGBoost model, which uses DEM and climate embeddings to predict urban potential
Author: Weiyu Zhang
Date: 2025-02-19
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import random

def setup_logging(log_path: str = "logs"):
    # configure log
    log_dir = Path(log_path)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "model_training.log"),
            logging.StreamHandler()
        ]
    )

def bootstrap_sample_and_normalize(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # Generate bootstrap samples and perform normalization
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot, y_boot = X.iloc[indices], y.iloc[indices]

    # subsample 
    rus = RandomUnderSampler(
        sampling_strategy=0.1,
        random_state=random.randint(1, 1000000)
    )

    X_boot, y_boot = rus.fit_resample(X_boot, y_boot)
    
    # normalize the bootstrap sample
    scaler = StandardScaler()
    X_boot_scaled = pd.DataFrame(
        scaler.fit_transform(X_boot),
        columns=X_boot.columns
    )
    
    # save normalization parameters
    feature_info = pd.DataFrame({
        'feature_names': X_boot.columns.tolist(),
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    })
    
    return X_boot_scaled, y_boot, feature_info

def train_bootstrap_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_models: int = 100,
    base_config: dict = None,
) -> Tuple[List[XGBClassifier], List[pd.DataFrame]]:
    # Train multiple bootstrap models, each with its own random seed and normalization
    models = []
    feature_infos = []
    
    for i in range(n_models):
        # configure new random seed
        random_seed = random.randint(1, 1000000)
        np.random.seed(random_seed)
        
        # Bootstrap sample and normalization
        logging.info(f"Training model {i + 1}/{n_models}")
        X_boot_scaled, y_boot, feature_info = bootstrap_sample_and_normalize(X, y)
        
        logging.info(f"Completed bootstrap sampling and normalization")
        
        # update random seed
        model_config = base_config.copy()
        model_config['random_state'] = random.randint(1, 1000000)
        
        # train model
        model = XGBClassifier(**model_config)
        model.fit(X_boot_scaled, y_boot)
        
        # only retain feature information used by the model
        if len(feature_info) != 0:
            cols_to_save = model.feature_names_in_
            feature_info = feature_info.loc[
                feature_info['feature_names'].isin(cols_to_save), :
            ]
            
        models.append(model)
        feature_infos.append(feature_info)
        
        
        logging.info(f"Complete training model {i + 1}/{n_models}")
    
    return models, feature_infos

def save_bootstrap_models(
    models: List[XGBClassifier],
    feature_infos: List[pd.DataFrame],
    config: dict,
    save_dir: Path,
    model_name: str
):
    """save bootstrap model and normalization parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp}_CNprefEUtrain_GLOBALpred_subsample0.6_n300_d4"
    model_dir = save_dir / timestamp / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # save basic configurations
    with open(model_dir / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # save each model along with its corresponding feature information
    for i, (model, feature_info) in enumerate(zip(models, feature_infos)):
        # create subdirectory of each model
        model_subdir = model_dir / f"model_{i}"
        model_subdir.mkdir(exist_ok=True)
        
        #save model
        joblib.dump(model, model_subdir / "model.joblib")
        
        # save feature information
        if len(feature_info) != 0:
            feature_info.to_csv(model_subdir / "feature_info.csv", index=False)
    
    logging.info(f"All models and feature information have been saved to: {model_dir}")
    return model_dir

def predict_with_bootstrap_models(
    models: List[XGBClassifier],
    feature_infos: List[pd.DataFrame],
    X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    # make predictions using the bootstrap model ensemble
    probas = []
    
    for model, feature_info in zip(models, feature_infos):
        # process input data using the corresponding feature information
        scaler = StandardScaler()
        feature_names = feature_info['feature_names'].tolist()
        scaler.mean_ = np.array(feature_info['mean'])
        scaler.scale_ = np.array(feature_info['std'])
        
        # use only the features required by the model
        X_scaled = pd.DataFrame(
            scaler.transform(X[feature_names]),
            columns=feature_names
        )
        
        # predict
        prob = model.predict_proba(X_scaled)[:, 1]
        probas.append(prob)
    
    probas = np.array(probas)
    mean_proba = np.mean(probas, axis=0)
    std_proba = np.std(probas, axis=0)
    
    return mean_proba, std_proba

def plot_prediction_uncertainty(
    mean_proba: np.ndarray,
    std_proba: np.ndarray,
    save_path: Path
):
    # visualization of prediction uncertainty
    plt.figure(figsize=(10, 6))
    
    # plot prediction Probability Distribution
    plt.subplot(1, 2, 1)
    plt.hist(mean_proba, bins=50, edgecolor='black')
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    
    # plot prediction Uncertainty Distribution
    plt.subplot(1, 2, 2)
    plt.hist(std_proba, bins=50, edgecolor='black')
    plt.title('Prediction Uncertainty Distribution')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_uncertainty.png')
    plt.close()

def main():
    # configure log
    setup_logging("/path/to/save/log")
    logger = logging.getLogger(__name__)
    
    try:
        # basic configurations
        save_models = True
        base_dir = Path("/path/to/save/trained/models")

        save_model_dir = base_dir / "saved_models"
        results_dir = base_dir / "results"
        save_model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # data preprocess
        logger.info("Start data preprocessing...")
        
        # load CN-pref and EU embeddings and corresponding labels 
        dem_emb_cn = pd.read_parquet('/path/to/CN-pref/dem/embeddings/embeddings_GEOS-213-CN-pref-DEMWater-256dim-499epoch-fixed_seed.parquet')
        dem_emb_eu = pd.read_parquet('/path/to/EU/dem/embeddings/embeddings_GEOS-213-EU-Augmented-DEMWater-128-499epoch-fixed_seed.parquet')
        climate_emb_cn = pd.read_parquet('/path/to/CN-pref/climate/embeddings/embeddings_GEOS-275-CN-pref-current-Clim-128-1999epoch-Y2000.parquet')
        climate_emb_eu = pd.read_parquet('/path/to/EU/climate/embeddings/embeddings_GEOS-275-EU-current-Clim-128-1999epoch-Y2000.parquet')

        label_eu = pd.read_parquet('/path/to/EU/label/combined_df_compiled_10pt_rectified_city_clim_augmented.parquet')
        label_cn = pd.read_parquet('/path/to/CN-pref/label/CN_pref_attribute_dem_clim_augmented.parquet')
        label_eu = label_eu.loc[:, ['id', 'is_city']]
        label_cn = label_cn.loc[:, ['id', 'is_city']]

        # merge data
        emb_eu = dem_emb_eu.merge(climate_emb_eu, on='id')
        emb_cn = dem_emb_cn.merge(climate_emb_cn, on='id')
        
        data_eu = emb_eu.merge(label_eu, on='id')
        data_cn = emb_cn.merge(label_cn, on='id')
        
        combined_df = pd.concat([data_eu, data_cn], ignore_index=True)
        
        # prepare features and labels
        X = combined_df.drop(columns=['id', 'is_city'])
        y = combined_df['is_city']
        
        # XGBoost configuration
        configs = json.load(open('/path/to/xgboost_config.json', 'r'))
        config = configs['eu-cn']["All_embedding"]
        
        # train bootstrap model
        logger.info("Start training bootstrap model...")
        models, feature_infos = train_bootstrap_models(
            X=X,
            y=y,
            n_models=300,
            base_config=config
        )
        
        # save models
        model_dir = save_bootstrap_models(
            models=models,
            feature_infos=feature_infos,
            config=config,
            save_dir=save_model_dir,
            model_name="All_embedding_bootstrap"
        )
        logger.info("Processing completed!")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()