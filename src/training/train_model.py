"""
Model training utilities for the Iris classification task.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple, Optional
import joblib
import os
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Model trainer for the Iris classification task."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_info = {}
        
    def create_model(self, model_type: str = 'DecisionTreeClassifier') -> Any:
        """
        Create a model instance based on configuration.
        
        Args:
            model_type: Type of model to create (overrides config)
            
        Returns:
            Model instance
        """
        model_type = model_type or self.config.get('type', 'DecisionTreeClassifier')
        parameters = self.config.get('parameters', {})
        
        logger.info(f"Creating {model_type} model with parameters: {parameters}")
        
        if model_type == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(**parameters)
        elif model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(**parameters)
        elif model_type == 'LogisticRegression':
            self.model = LogisticRegression(**parameters)
        elif model_type == 'SVC':
            self.model = SVC(**parameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_info = {
            'type': model_type,
            'parameters': parameters,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Model created successfully: {model_type}")
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        logger.info(f"Target distribution: {y_train.value_counts().to_dict()}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Store training info
        self.model_info.update({
            'training_samples': X_train.shape[0],
            'training_features': X_train.shape[1],
            'target_distribution': y_train.value_counts().to_dict(),
            'trained_at': pd.Timestamp.now().isoformat()
        })
        
        logger.info("Model training completed successfully")
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        if self.model is None:
            raise ValueError("No model available. Train the model first.")
        
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'cv_folds': cv
        }
        
        logger.info(f"Cross-validation results: {cv_results}")
        return cv_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model available. Train the model first.")
        
        logger.info(f"Making predictions on {X.shape[0]} samples")
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model available. Train the model first.")
        
        if hasattr(self.model, 'predict_proba'):
            logger.info(f"Getting prediction probabilities for {X.shape[0]} samples")
            probabilities = self.model.predict_proba(X)
            return probabilities
        else:
            logger.warning("Model does not support probability predictions")
            return None
    
    def save_model(self, model_path: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model (uses config if not provided)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model available. Train the model first.")
        
        model_path = model_path or self.config.get('artifact_path', 'artifacts/model.joblib')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_info': self.model_info,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.model_info = model_data.get('model_info', {})
            self.config = model_data.get('config', self.config)
        else:
            # Backward compatibility for models saved without metadata
            self.model = model_data
        
        logger.info("Model loaded successfully")
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = self.model_info.copy()
        
        if self.model is not None:
            info['model_type'] = type(self.model).__name__
            info['has_proba'] = hasattr(self.model, 'predict_proba')
        
        return info
    
    def train_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training pipeline")
        
        # Create and train model
        self.create_model()
        self.train_model(X_train, y_train)
        
        # Perform cross-validation
        cv_results = self.cross_validate(X_train, y_train)
        
        # Save model
        model_path = self.save_model()
        
        # Prepare results
        results = {
            'model_info': self.get_model_info(),
            'cv_results': cv_results,
            'model_path': model_path,
            'training_samples': X_train.shape[0],
            'training_features': X_train.shape[1]
        }
        
        # Add validation results if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            from sklearn.metrics import accuracy_score
            val_accuracy = accuracy_score(y_val, val_predictions)
            results['validation_accuracy'] = val_accuracy
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        logger.info("Model training pipeline completed successfully")
        return results 