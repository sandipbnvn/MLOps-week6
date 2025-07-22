"""
Model evaluation utilities for the Iris classification task.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, Optional
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk issues
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Model evaluator for the Iris classification task."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Calculating evaluation metrics")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Multi-class metrics (macro average)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Store per-class metrics
        classes = y_true.unique()
        for i, class_name in enumerate(classes):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # ROC AUC (if probabilities are available)
        if y_proba is not None and y_proba.shape[1] == len(classes):
            try:
                # For multi-class, calculate ROC AUC for each class vs rest
                roc_auc_scores = []
                for i in range(len(classes)):
                    roc_auc = roc_auc_score((y_true == classes[i]).astype(int), y_proba[:, i])
                    roc_auc_scores.append(roc_auc)
                    metrics[f'roc_auc_{classes[i]}'] = roc_auc
                
                metrics['roc_auc_macro'] = np.mean(roc_auc_scores)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Round metrics for readability
        for key in metrics:
            if isinstance(metrics[key], float):
                metrics[key] = round(metrics[key], 4)
        
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
    
    def generate_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        logger.info("Generating classification report")
        
        report = classification_report(y_true, y_pred, output_dict=False)
        return report
    
    def generate_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        logger.info("Generating confusion matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                      y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Generate classification report
        classification_report_str = self.generate_classification_report(y_true, y_pred)
        
        # Generate confusion matrix
        confusion_matrix_array = self.generate_confusion_matrix(y_true, y_pred)
        
        # Check if model meets threshold requirements
        threshold_accuracy = self.config.get('threshold_accuracy', 0.95)
        meets_threshold = metrics['accuracy'] >= threshold_accuracy
        
        # Prepare evaluation results
        evaluation_results = {
            'metrics': metrics,
            'classification_report': classification_report_str,
            'confusion_matrix': confusion_matrix_array.tolist(),
            'meets_threshold': meets_threshold,
            'threshold_accuracy': threshold_accuracy,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'sample_count': len(y_true)
        }
        
        self.evaluation_results = evaluation_results
        
        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Meets threshold ({threshold_accuracy}): {meets_threshold}")
        
        return evaluation_results
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               output_path: str = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results (uses config if not provided)
            
        Returns:
            Path where results were saved
        """
        output_path = output_path or self.config.get('metrics_path', 'artifacts/metrics.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        return output_path
    
    def load_evaluation_results(self, results_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from file.
        
        Args:
            results_path: Path to the results file
            
        Returns:
            Loaded evaluation results
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        logger.info(f"Loading evaluation results from {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        logger.info("Evaluation results loaded successfully")
        return results
    
    def create_evaluation_plots(self, y_true: pd.Series, y_pred: np.ndarray,
                               y_proba: np.ndarray = None, 
                               output_dir: str = 'artifacts') -> Dict[str, str]:
        """
        Create evaluation plots and save them.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with plot file paths
        """
        logger.info("Creating evaluation plots")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = {}
        
        # Set style for better-looking plots
        plt.style.use('default')
        
        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=y_true.unique(), yticklabels=y_true.unique())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['confusion_matrix'] = confusion_matrix_path
        
        # 2. ROC Curves (if probabilities are available)
        if y_proba is not None:
            classes = y_true.unique()
            plt.figure(figsize=(10, 8))
            
            for i, class_name in enumerate(classes):
                fpr, tpr, _ = roc_curve((y_true == class_name).astype(int), y_proba[:, i])
                roc_auc = roc_auc_score((y_true == class_name).astype(int), y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend(loc="lower right")
            
            roc_curves_path = os.path.join(output_dir, 'roc_curves.png')
            plt.savefig(roc_curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['roc_curves'] = roc_curves_path
        
        logger.info(f"Evaluation plots created: {list(plot_paths.keys())}")
        return plot_paths
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of evaluation results.
        
        Returns:
            Dictionary with evaluation summary
        """
        if not self.evaluation_results:
            return {}
        
        metrics = self.evaluation_results.get('metrics', {})
        
        summary = {
            'accuracy': metrics.get('accuracy', 0),
            'precision_macro': metrics.get('precision_macro', 0),
            'recall_macro': metrics.get('recall_macro', 0),
            'f1_macro': metrics.get('f1_macro', 0),
            'meets_threshold': self.evaluation_results.get('meets_threshold', False),
            'sample_count': self.evaluation_results.get('sample_count', 0),
            'evaluation_timestamp': self.evaluation_results.get('evaluation_timestamp', '')
        }
        
        return summary 