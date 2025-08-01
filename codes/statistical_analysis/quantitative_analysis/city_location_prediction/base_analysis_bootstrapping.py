# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2025.2.15
Description: Predict city locations via embeddings and manual features using XGBoost model. 
             Processes geographical embeddings and traditional features to predict urban formation patterns.
             args:
                -d, --dataset: Dataset to use (cn-walled, cn-pref, eu
                --feature_type: Type of features to use (All_embedding, Attribute, DEM_Water_embedding, Climate_embedding, All_embedding_Agriculture)
                --bootstrap_iterations: Number of bootstrap iterations for confidence intervals (default: 1000)
                --enable_shap: Enable SHAP analysis for model interpretation (default: False)
                --config_path: Path to hyperparameters configuration json file

Use cases:
    python main.py -d cn-walled --feature_type All_embedding --bootstrap_iterations 1000 --enable_shap
"""

import logging
from pathlib import Path
import os
import pandas as pd
import sys
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils import PreprocessPipeline
from utils.data_sampler import DataSampler
from models.xgboost_trainer import XGBoostTrainer

def setup_logging(log_path: str = "logs"):
    """
    Setup logging configuration for the training process.
    
    Args:
        log_path (str): Directory path for log files
    """
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

def load_config(config_path: str) -> dict:
    """
    Load hyperparameters configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration JSON file
        
    Returns:
        dict: Nested dictionary with dataset and model specific parameters
    """
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict

def main():
    """
    Main execution function for city formation prediction analysis.
    """
    # Setup logging and result paths
    setup_logging("./logs")
    logger = logging.getLogger(__name__)
    
    save_models = True
    base_dir = Path("./")
    save_model_dir = base_dir / "saved_models"
    results_dir = base_dir / "results"

    try:
        # Data preprocessing
        logger.info("Starting data preprocessing...")
        pipeline = PreprocessPipeline(
            base_path="",
            model_name="",
            data_file_name="",
            random_state=42,
            test_ratio=0.2,
        )
        
        # Parse command line arguments for dataset selection
        parser = argparse.ArgumentParser(description='City-presence classification')
        parser.add_argument('-d', '--dataset', type=str, required=True, 
                          default='cn-walled', help='Dataset: cn-walled, cn-pref, or eu')
        parser.add_argument('--feature_type', type=str, required=True,
                          help='Feature type: All_embedding, Attribute, DEM_Water_embedding, Climate_embedding, All_embedding_Agriculture')
        parser.add_argument('--bootstrap_iterations', type=int, default=1000,
                          help='Number of bootstrap iterations for confidence intervals (default: 1000)')
        parser.add_argument('--enable_shap', action='store_true', default=False,
                          help='Enable SHAP analysis for model interpretation')
        parser.add_argument('--disable_shap', dest='enable_shap', action='store_false',
                          help='Disable SHAP analysis (default)')
        parser.add_argument('--output_path', type=str, default='./results',)
        
        args = parser.parse_args()
        dataset = args.dataset
        feature_type = args.feature_type
        bootstrap_iterations = args.bootstrap_iterations
        enable_shap = args.enable_shap
        output_path = args.output_path
        
        
        if dataset not in ["cn-walled", "cn-pref", "eu"]:
            raise ValueError("Invalid dataset. Choose from: cn-walled, cn-pref, eu")
        if feature_type not in ["All_embedding", "Attribute", "DEM_Water_embedding", "Climate_embedding", "All_embedding_Agriculture"]:
            raise ValueError(f"Invalid feature type. Choose from: {list(feature_mapping.keys())}")
        

        # Load dataset based on argument
        if dataset == "cn-pref":
            combined_data = pd.read_parquet("/path/to/Chinese_pref/combined_df_in_polygon.parquet")
        elif dataset == "cn-walled":
            combined_data = pd.read_parquet("/path/to/Chinese_walled/combined_df.parquet")
        elif dataset == "eu":
            combined_data = pd.read_parquet("/path/to/Euro/combined_df.parquet")
            combined_data.drop(columns=['agri'], inplace=True)
        else:
            raise ValueError("Invalid dataset")

        # Process data with standardization
        processed_data = pipeline.run_with_loaded_data(combined_data, standardize=True)

        # Data sampling for balanced training
        logger.info("Performing data sampling...")
        sampler = DataSampler(random_state=42, sampling_strategy=0.1)
        balanced_data, y_train_balanced, y_test = sampler.prepare_balanced_data(
            processed_data,
            sampling_strategy='under'
        )
        
        # Model training and evaluation
        logger.info("Starting model training...")
        configs = load_config("./configs/configs.json") # configs of all models

        # Define single model type for analysis
        model_names = [feature_type]
            
        trainer = XGBoostTrainer()
        logger.info(f"Selected feature type: {feature_type}")
        logger.info(f"Actual feature name: {feature_type}")
        logger.info(f"Bootstrap iterations: {bootstrap_iterations}")
        logger.info(f"SHAP analysis: {'Enabled' if enable_shap else 'Disabled'}")
        
        datasets = [balanced_data[feature_type]]
        logger.info(f"Data format: {len(datasets)} dataset(s)")
        
        # Get model-specific config
        if dataset in configs and feature_type in configs[dataset]:
            model_config = configs[dataset][feature_type]
            logger.info(f"Using config for {dataset}/{feature_type}: {model_config}")
        else:
            logger.warning(f"No specific config found for {dataset}/{feature_type}, using default XGBoost parameters")
            model_config = {}
        
        # Train and evaluate models
        metrics = trainer.train_and_evaluate(
            datasets=datasets,
            model_names=model_names,
            y_train=y_train_balanced,
            y_test=y_test,
            bootstrapping=True,
            bootstrap_iterations=bootstrap_iterations, 
            SHAP=enable_shap,
            configs=configs
        )

        # Save results
        logger.info("Saving results...")
        
        # Compile results into DataFrame
        data = []
        for model in metrics.keys():
            row = {'model': model}
            for metric, values in metrics[model].items():
                row[str(metric)+"_mean"] = values['mean']
                row[str(metric)+"_ci_lower"] = values['ci_lower']
                row[str(metric)+"_ci_upper"] = values['ci_upper']
                if "shap" in str(metric):
                    print(str(metric), values['mean'])
            data.append(row)

        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        output_path = f'{output_path}/results_{dataset}_{feature_type}_bootstrap{bootstrap_iterations}_shap{enable_shap}.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info("Processing completed successfully!")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()