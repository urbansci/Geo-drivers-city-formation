# -*- coding: utf-8 -*-
"""
Author: Weiyu Zhang
Date: 2024
Description: XGBoost trainer for geographical city formation analysis with bootstrapping and SHAP analysis support.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    roc_curve
)
from sklearn.utils import resample
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelMetricsLogger:
    """Model metrics logger for saving evaluation results."""
    
    def __init__(self, base_path: str = "model_outputs"):
        """
        Initialize the metrics logger.
        
        Args:
            base_path (str): Base directory for saving outputs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def save_metrics(self, metrics: Dict, timestamp: Optional[str] = None):
        """
        Save all model metrics.
        
        Args:
            metrics (Dict): Dictionary containing model evaluation metrics
            timestamp (Optional[str]): Timestamp for output directory naming
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create output directory
        output_dir = self.base_path / timestamp
        output_dir.mkdir(exist_ok=True)
        
        # Save summary metrics CSV
        self._save_summary_csv(metrics, output_dir)
        
        # Save detailed text report
        self._save_detailed_report(metrics, output_dir)
        
        # Save confusion matrices
        self._save_confusion_matrices(metrics, output_dir)
        
    def _save_summary_csv(self, metrics: Dict, output_dir: Path):
        """Save main metrics summary CSV."""
        summary_data = []
        
        for model_name, model_metrics in metrics.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': model_metrics['accuracy'],
                'F1 Score': model_metrics['f1_score'],
                'MCC': model_metrics['mcc'],
                'PR-AUC': model_metrics['pr_info'][2],
                'ROC-AUC': model_metrics['roc_info'][2]
            })
            
        pd.DataFrame(summary_data).to_csv(
            output_dir / 'metrics_summary.csv',
            index=False
        )
        
    def _save_detailed_report(self, metrics: Dict, output_dir: Path):
        """Save detailed text report."""
        with open(output_dir / 'detailed_report.txt', 'w', encoding='utf-8') as f:
            for model_name, model_metrics in metrics.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'='*50}\n\n")
                
                # Basic metrics
                f.write(f"Accuracy: {model_metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {model_metrics['f1_score']:.4f}\n")
                f.write(f"MCC: {model_metrics['mcc']:.4f}\n")
                f.write(f"PR-AUC: {model_metrics['pr_info'][2]:.4f}\n")
                f.write(f"ROC-AUC: {model_metrics['roc_info'][2]:.4f}\n\n")
                
                # Classification report
                f.write("Classification Report:\n")
                f.write(f"{model_metrics['classification_report']}\n\n")
    
    def _save_confusion_matrices(self, metrics: Dict, output_dir: Path):
        """Save confusion matrices."""
        confusion_matrices = {
            model_name: model_metrics['confusion_matrix'].tolist()
            for model_name, model_metrics in metrics.items()
        }
        
        with open(output_dir / 'confusion_matrices.json', 'w') as f:
            json.dump(confusion_matrices, f, indent=2)


class XGBoostTrainer:
    """XGBoost model trainer for geographical analysis."""
    
    def __init__(self):
        """Initialize the XGBoost trainer."""
        self.config = {}
        self.models = {}
        self.metrics = {}
        
    def train_and_evaluate(
        self,
        datasets,
        model_names,
        y_train,
        y_test,
        bootstrapping=False,
        bootstrap_iterations=1000,
        configs=None,
        SHAP=False
    ) -> Dict:
        """
        Train and evaluate multiple models.
        
        Args:
            datasets: List of (x_train, x_test) tuples
            model_names: List of model names
            y_train: Training labels
            y_test: Test labels
            bootstrapping: Whether to use bootstrapping for evaluation
            bootstrap_iterations: Number of bootstrap iterations
            configs: Model configurations dictionary
            SHAP: Whether to compute SHAP values
            
        Returns:
            Dict: Dictionary containing evaluation metrics for all models
        """
        for i, (x_train, x_test) in enumerate(datasets):
            model_name = model_names[i]
            logger.info(f"Training model: {model_name}")
            
            # Configure model
            if configs is not None:
                self.config = (configs[model_name])
                self.config['nthread'] = 10
            else:
                raise ValueError("configs cannot be None, please provide model configurations.")
            model = XGBClassifier(**self.config)
            
            # Train model
            logger.info(f"Starting training for {model_name} model")
            logger.info(f"Training dataset size: {x_train.shape}")
            logger.info(f"Test dataset size: {x_test.shape}")
            model.fit(x_train, y_train)
            self.models[model_name] = model
            logger.info(f"{model_name} model training completed")

            # Predict and evaluate
            if bootstrapping:
                metrics = self._evaluate_model_bootstrapping(
                    model, x_test, y_test, SHAP=SHAP, bootstrap_iterations=bootstrap_iterations
                )
            else:
                metrics = self._evaluate_model(model, x_test, y_test)
            self.metrics[model_name] = metrics
            
            logger.info(f"{model_name} model evaluation completed")
            # Log evaluation results
            if not bootstrapping:
                self._log_metrics(model_name, metrics)
            
        return self.metrics
    
    def calculate_metrics_with_bootstrapping_multiprocessing(
        self, y_true, y_pred, y_pred_proba, x, model, 
        n_iterations=1000, confidence_level=0.95, 
        SHAP=False, n_jobs=30
    ):
        """
        Calculate metrics with stratified bootstrapping using multiprocessing.
        Optimized version: pre-compute SHAP values, extract needed rows for each bootstrap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            x: Feature data
            model: Trained model
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            SHAP: Whether to calculate SHAP values
            n_jobs: Number of parallel jobs
            
        Returns:
            Dict: Metrics with confidence intervals
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, f1_score
        from sklearn.utils import resample
        import shap
        from joblib import Parallel, delayed
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Convert input data to numpy arrays
        if isinstance(x, pd.DataFrame):
            feature_columns = x.columns
            x_values = x.values
        else:
            x_values = x
            
        y_true_values = np.asarray(y_true).flatten()
        y_pred_values = np.asarray(y_pred).flatten()
        y_pred_proba_values = np.asarray(y_pred_proba).flatten()
        
        n_samples = len(y_true_values)
        indices = np.arange(n_samples)
        
        # Pre-compute stratification information
        unique_classes = np.unique(y_true_values)
        class_indices = {c: indices[y_true_values == c] for c in unique_classes}
        
        # Pre-compute all SHAP values - key optimization
        all_shap_values = None
        print(f"x is DataFrame: {isinstance(x, pd.DataFrame)}")
        if SHAP and isinstance(x, pd.DataFrame):
            logger.info("Pre-computing SHAP values for all samples...")
            explainer = shap.TreeExplainer(model)
            all_explanation = explainer(x_values)
            all_shap_values = all_explanation.values
            
            print("SHAP values pre-computation completed, shape:", all_shap_values.shape)
            print("SHAP values example:", all_shap_values[:5, :5])  # Print first 5 samples, first 5 features
            
            # Pre-compute feature indices
            clim_emb_list_idx = np.array([i for i, col in enumerate(feature_columns) if col.startswith('C')])
            dem_emb_list_idx = np.array([i for i, col in enumerate(feature_columns) if col.startswith('D')])
            agri_emb_list_idx = np.array([i for i, col in enumerate(feature_columns) if col in ['pre', 'post']])
            
            logger.info("SHAP values pre-computation completed")
            
            logger.info(f"Climate features count: {len(clim_emb_list_idx)}")
            logger.info(f"DEM-water features count: {len(dem_emb_list_idx)}")
            logger.info(f"Agricultural features count: {len(agri_emb_list_idx)}")
            
        
        # Define optimized bootstrap iteration function
        def bootstrap_iteration(n, report_interval=100):
            if n % report_interval == 0:
                logger.info(f"Bootstrapping iteration: {n+1}/{n_iterations}")
                
            results = {}
            
            # Use pre-computed stratification info for resampling
            bootstrap_indices = np.concatenate([
                np.random.choice(class_indices[c], 
                            size=len(class_indices[c]), 
                            replace=True) 
                for c in unique_classes
            ])
            
            # Get bootstrap samples
            y_true_boot = y_true_values[bootstrap_indices]
            y_pred_boot = y_pred_values[bootstrap_indices]
            y_pred_proba_boot = y_pred_proba_values[bootstrap_indices]
            
            # Calculate metrics
            results['mcc'] = matthews_corrcoef(y_true_boot, y_pred_boot)
            
            precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_proba_boot)
            results['pr_auc'] = auc(recall, precision)
            
            results['f1'] = f1_score(y_true_boot, y_pred_boot)
            
            if SHAP:
                # Calculate SHAP values for bootstrap sample
                explanation = explainer(x.iloc[bootstrap_indices])

                if len(clim_emb_list_idx) > 0:
                    results['clim_shap'] = np.abs(explanation[:, clim_emb_list_idx].sum(axis=1)).mean()
                if len(dem_emb_list_idx) > 0:
                    results['dem_shap'] = np.abs(explanation[:, dem_emb_list_idx].sum(axis=1)).mean()
                if len(agri_emb_list_idx) > 0:
                    results['agri_shap'] = np.abs(explanation[:, agri_emb_list_idx].sum(axis=1)).mean()
            return results

        np.random.seed(42)
        
        # Execute bootstrap iterations in parallel
        bootstrap_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(bootstrap_iteration)(n, report_interval=max(1, n_iterations//10)) 
            for n in range(n_iterations)
        )
        
        # Process results
        metrics_names = ['mcc', 'pr_auc', 'f1']
        if SHAP:
            metrics_names.extend(['dem_shap', 'clim_shap', 'agri_shap'])
        
        # Pre-allocate storage space
        metrics_boot = {metric: np.zeros(n_iterations) for metric in metrics_names}
        
        # Fill results
        for i, result in enumerate(bootstrap_results):
            for metric in metrics_names:
                if metric in result:
                    metrics_boot[metric][i] = result[metric]
        
        # Calculate confidence intervals
        alpha = (1 - confidence_level) / 2
        results = {}
        
        for metric in metrics_names:
            values = metrics_boot[metric]
            results[metric] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, alpha * 100),
                'ci_upper': np.percentile(values, (1 - alpha) * 100)
            }
            
        return results

    def calculate_metrics_with_bootstrapping(
        self, y_true, y_pred, y_pred_proba, x, model, 
        n_iterations=1000, confidence_level=0.95, SHAP=False
    ):
        """
        Calculate MCC, PR-AUC and F1 scores with confidence intervals using stratified bootstrapping.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            x: Feature data
            model: Trained model
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level
            SHAP: Whether to calculate SHAP values
            
        Returns:
            Dict: Dictionary containing metrics and their confidence intervals
        """
        n_samples = len(y_true)
        metrics_boot = {
            'mcc': [],
            'pr_auc': [],
            'f1': []
        }
        if SHAP:
            metrics_boot['dem_shap'] = []
            metrics_boot['clim_shap'] = []
            metrics_boot['agri_shap'] = []
        
        def reset_if_dataframe(data):
            if isinstance(data, pd.DataFrame):
                return data.reset_index(drop=True)
            return data

        # Apply to all variables
        x = reset_if_dataframe(x)
        y_true = reset_if_dataframe(y_true)
        y_pred = reset_if_dataframe(y_pred)
        y_pred_proba = reset_if_dataframe(y_pred_proba)

        # Get all sample indices
        indices = np.arange(n_samples)
        
        explainer = shap.TreeExplainer(model)
        # Execute stratified bootstrapping
        for n in range(n_iterations):
            logger.info(f"Bootstrapping iteration: {n+1}")
            # Use stratified resampling
            bootstrap_indices = resample(indices, 
                                    n_samples=n_samples,
                                    stratify=y_true)
                
            # Calculate MCC
            metrics_boot['mcc'].append(
                matthews_corrcoef(y_true[bootstrap_indices], y_pred[bootstrap_indices])
            )
            
            # Calculate PR-AUC
            precision, recall, _ = precision_recall_curve(
                y_true[bootstrap_indices], 
                y_pred_proba[bootstrap_indices]
            )
            metrics_boot['pr_auc'].append(auc(recall, precision))
            
            # Calculate F1 score
            metrics_boot['f1'].append(
                f1_score(y_true[bootstrap_indices], y_pred[bootstrap_indices])
            )

            if SHAP:
                # Calculate SHAP values
                explanation = explainer(x.iloc[bootstrap_indices])

                clim_emb_list_idx = [i for i, col in enumerate(x.columns) if col.startswith('C')]
                explanation_clim = explanation[:, clim_emb_list_idx]
                clim_shap_sum = explanation_clim.values.sum(1)

                dem_emb_list_idx = [i for i, col in enumerate(x.columns) if col.startswith('D')]
                explanation_dem = explanation[:, dem_emb_list_idx]
                dem_shap_sum = explanation_dem.values.sum(1)

                agri_emb_list_idx = [i for i, col in enumerate(x.columns) if col in ['ramankutti', 'pre', 'post']]
                explanation_agri = explanation[:, agri_emb_list_idx]
                agri_shap_sum = explanation_agri.values.sum(1)
                
                metrics_boot['dem_shap'].append(np.abs(dem_shap_sum).mean())
                metrics_boot['clim_shap'].append(np.abs(clim_shap_sum).mean())
                metrics_boot['agri_shap'].append(np.abs(agri_shap_sum).mean())
        
        # Calculate confidence intervals
        alpha = (1 - confidence_level) / 2
        results = {}
        for metric in metrics_boot:
            values = np.array(metrics_boot[metric])
            mean_val = np.mean(values)
            ci_lower = np.percentile(values, alpha * 100)
            ci_upper = np.percentile(values, (1 - alpha) * 100)
            
            results[metric] = {
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        return results

    def _evaluate_model(
        self,
        model: XGBClassifier,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray = None,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """Evaluate single model."""
        if y_pred is None:
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            y_pred = model.predict(x_test)
        
        assert y_pred is not None and y_pred_proba is not None, "y_pred and y_pred_proba cannot be None"
        
        # ROC curve metrics
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # PR curve metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'roc_info': (fpr, tpr, roc_auc),
            'pr_info': (recall, precision, pr_auc)
        }

    def _evaluate_model_bootstrapping(
        self,
        model: XGBClassifier,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray = None,
        y_pred_proba: np.ndarray = None,
        SHAP=False,
        bootstrap_iterations: int = 1000
    ) -> Dict:
        """Evaluate single model with bootstrapping."""
        if y_pred is None:
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            y_pred = model.predict(x_test)
        
        assert y_pred is not None and y_pred_proba is not None, "y_pred and y_pred_proba cannot be None"
        
        metrics = self.calculate_metrics_with_bootstrapping_multiprocessing(
            y_test, y_pred, y_pred_proba, x_test, model, SHAP=SHAP, n_iterations=bootstrap_iterations
        )

        return metrics
    
    def _log_metrics(self, model_name: str, metrics: Dict):
        """Log evaluation metrics."""
        logger.info(f"\nModel: {model_name}")
        if 'accuracy' in metrics:
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'confusion_matrix' in metrics:
            logger.info(f"Confusion matrix:\n{metrics['confusion_matrix']}")
        if 'classification_report' in metrics:
            logger.info(f"Classification report:\n{metrics['classification_report']}")
        if 'f1_score' in metrics:
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        if 'mcc' in metrics:
            logger.info(f"MCC: {metrics['mcc']:.4f}")
        if 'pr_info' in metrics:
            logger.info(f"PR-AUC: {metrics['pr_info'][2]:.4f}")

    def save_results(self, base_path: str = "/path/to/results"):
        """Save model evaluation results."""
        logger_instance = ModelMetricsLogger(base_path)
        logger_instance.save_metrics(self.metrics)

    def plot_curves(self, save_path: Optional[str] = None):
        """Plot ROC and PR curves."""
        plt.figure(figsize=(12, 5))
        
        # Plot ROC curves
        plt.subplot(1, 2, 1)
        for model_name, metrics in self.metrics.items():
            fpr, tpr, roc_auc = metrics['roc_info']
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        # Plot PR curves
        plt.subplot(1, 2, 2)
        for model_name, metrics in self.metrics.items():
            recall, precision, pr_auc = metrics['pr_info']
            plt.plot(recall, precision, lw=2, label=f'{model_name} (PRAUC = {pr_auc:.3f})')
        
        plt.ylim(0, 1.1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_models(
        self, 
        base_path: str = "/path/to/saved_models",
        feature_info={}, 
        timestamp: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save all trained models and related configurations.
        
        Args:
            base_path (str): Base path for saving models
            feature_info: Feature information dictionary
            timestamp (Optional[str]): Timestamp for creating unique save directory
            
        Returns:
            Dict[str, str]: Dictionary containing save paths for each model
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create save directory
        save_dir = Path(base_path) / timestamp / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config_path = save_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save each model
        model_paths = {}
        for model_name, model in self.models.items():
            # Create model-specific directory
            model_dir = save_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model file
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
            
            # Save feature information (if available)
            if hasattr(model, 'feature_names_'):
                feature_path = model_dir / "feature_names.json"
                with open(feature_path, 'w') as f:
                    json.dump(list(model.feature_names_), f, indent=2)
                    
            model_paths[model_name] = str(model_path)
            
            logger.info(f"Model {model_name} saved to: {model_path}")
                
            if len(feature_info) != 0:
                cols_to_save = model.feature_names_in_
                _feature_info = feature_info.loc[feature_info['feature_names'].isin(cols_to_save), :]
                _feature_info.to_csv(model_dir / "feature_info.csv", index=False)
            
        # Save model path mapping
        path_map = save_dir / "model_paths.json"
        with open(path_map, 'w') as f:
            json.dump(model_paths, f, indent=2)

        return model_paths
    
    @classmethod
    def load_model(cls, model_path: str) -> XGBClassifier:
        """
        Load single saved model.
        
        Args:
            model_path (str): Model file path
            
        Returns:
            XGBClassifier: Loaded model
        """
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    @classmethod
    def load_models(cls, base_dir: str) -> Dict[str, XGBClassifier]:
        """
        Load all saved models from specified directory.
        
        Args:
            base_dir (str): Base directory containing all models
            
        Returns:
            Dict[str, XGBClassifier]: Mapping from model names to model objects
        """
        base_path = Path(base_dir)
        
        # Load model path mapping
        path_map_file = base_path / "models" / "model_paths.json"
        if not path_map_file.exists():
            raise FileNotFoundError(f"Model path mapping file not found: {path_map_file}")
            
        with open(path_map_file, 'r') as f:
            model_paths = json.load(f)
            
        # Load all models
        models = {}
        for model_name, model_path in model_paths.items():
            models[model_name] = cls.load_model(model_path)
            
        return models

