"""
Data processing utilities for the Iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Data processor for the Iris dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor with configuration.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate the loaded data.
        
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating data...")
        
        if self.data is None:
            logger.error("No data loaded")
            return False
        
        # Check for required columns
        required_columns = self.config.get('feature_columns', []) + [self.config.get('target_column', 'species')]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_values = self.data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values found: {missing_values.to_dict()}")
        
        # Check data types
        numeric_columns = self.config.get('feature_columns', [])
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.error(f"Column {col} is not numeric")
                return False
        
        logger.info("Data validation passed")
        return True
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        feature_columns = self.config.get('feature_columns', [])
        target_column = self.config.get('target_column', 'species')
        test_size = self.config.get('test_size', 0.4)
        random_state = self.config.get('random_state', 42)
        stratify = self.config.get('stratify', True)
        
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        if stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        logger.info(f"Data split completed:")
        logger.info(f"  X_train shape: {self.X_train.shape}")
        logger.info(f"  X_test shape: {self.X_test.shape}")
        logger.info(f"  y_train shape: {self.y_train.shape}")
        logger.info(f"  y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data.
        
        Returns:
            Dictionary containing data summary
        """
        if self.data is None:
            return {}
        
        summary = {
            'total_samples': len(self.data),
            'features': len(self.config.get('feature_columns', [])),
            'target_distribution': self.data[self.config.get('target_column', 'species')].value_counts().to_dict(),
            'feature_stats': self.data[self.config.get('feature_columns', [])].describe().to_dict()
        }
        
        return summary
    
    def process_pipeline(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data processing pipeline")
        
        # Load data
        self.load_data(data_path)
        
        # Validate data
        if not self.validate_data():
            raise ValueError("Data validation failed")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Log summary
        summary = self.get_data_summary()
        logger.info(f"Data processing completed. Summary: {summary}")
        
        return X_train, X_test, y_train, y_test 