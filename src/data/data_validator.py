"""
Data validation utilities for the Iris dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Data validator for the Iris dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator with configuration.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.validation_results = {}
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """
        Validate data schema (columns and data types).
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if schema is valid, False otherwise
        """
        logger.info("Validating data schema...")
        
        # Check required columns
        required_columns = self.config.get('feature_columns', []) + [self.config.get('target_column', 'species')]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            self.validation_results['schema'] = {
                'valid': False,
                'missing_columns': missing_columns
            }
            return False
        
        # Check data types
        feature_columns = self.config.get('feature_columns', [])
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.error(f"Feature column {col} is not numeric")
                self.validation_results['schema'] = {
                    'valid': False,
                    'invalid_types': {col: str(data[col].dtype)}
                }
                return False
        
        logger.info("Schema validation passed")
        self.validation_results['schema'] = {'valid': True}
        return True
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality (missing values, etc.).
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data quality is acceptable, False otherwise
        """
        logger.info("Validating data quality...")
        
        quality_issues = []
        
        # Check for missing values
        feature_columns = self.config.get('feature_columns', [])
        missing_values = data[feature_columns].isnull().sum()
        
        if missing_values.sum() > 0:
            quality_issues.append(f"Missing values found: {missing_values.to_dict()}")
        
        # Check target distribution
        target_column = self.config.get('target_column', 'species')
        target_distribution = data[target_column].value_counts()
        min_samples_per_class = target_distribution.min()
        
        if min_samples_per_class < 10:  # Minimum 10 samples per class
            quality_issues.append(f"Class imbalance detected. Minimum samples per class: {min_samples_per_class}")
        
        if quality_issues:
            logger.warning(f"Data quality issues found: {quality_issues}")
            self.validation_results['quality'] = {
                'valid': False,
                'issues': quality_issues
            }
            return False
        
        logger.info("Data quality validation passed")
        self.validation_results['quality'] = {'valid': True}
        return True
    
    def validate_data_range(self, data: pd.DataFrame) -> bool:
        """
        Validate data ranges for Iris dataset features.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if ranges are valid, False otherwise
        """
        logger.info("Validating data ranges...")
        
        # Expected ranges for Iris dataset features
        expected_ranges = {
            'sepal_length': (4.0, 8.0),
            'sepal_width': (2.0, 5.0),
            'petal_length': (1.0, 7.0),
            'petal_width': (0.1, 2.5)
        }
        
        feature_columns = self.config.get('feature_columns', [])
        range_issues = []
        
        for col in feature_columns:
            if col in expected_ranges:
                min_val, max_val = expected_ranges[col]
                actual_min = data[col].min()
                actual_max = data[col].max()
                
                if actual_min < min_val or actual_max > max_val:
                    range_issues.append(
                        f"Column {col}: expected range [{min_val}, {max_val}], "
                        f"actual range [{actual_min:.2f}, {actual_max:.2f}]"
                    )
        
        if range_issues:
            logger.warning(f"Data range issues found: {range_issues}")
            self.validation_results['range'] = {
                'valid': False,
                'issues': range_issues
            }
            return False
        
        logger.info("Data range validation passed")
        self.validation_results['range'] = {'valid': True}
        return True
    
    def validate_target_values(self, data: pd.DataFrame) -> bool:
        """
        Validate target variable values.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if target values are valid, False otherwise
        """
        logger.info("Validating target values...")
        
        target_column = self.config.get('target_column', 'species')
        expected_targets = ['setosa', 'versicolor', 'virginica']
        
        actual_targets = data[target_column].unique()
        unexpected_targets = [t for t in actual_targets if t not in expected_targets]
        
        if unexpected_targets:
            logger.error(f"Unexpected target values: {unexpected_targets}")
            self.validation_results['target'] = {
                'valid': False,
                'unexpected_values': unexpected_targets
            }
            return False
        
        logger.info("Target validation passed")
        self.validation_results['target'] = {'valid': True}
        return True
    
    def run_all_validations(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all validation checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (all_valid, validation_results)
        """
        logger.info("Running all data validations...")
        
        validations = [
            self.validate_schema,
            self.validate_data_quality,
            self.validate_data_range,
            self.validate_target_values
        ]
        
        all_valid = True
        for validation in validations:
            if not validation(data):
                all_valid = False
        
        if all_valid:
            logger.info("All data validations passed")
        else:
            logger.error("Some data validations failed")
        
        return all_valid, self.validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'all_valid': all(result.get('valid', False) for result in self.validation_results.values()),
            'validation_results': self.validation_results,
            'total_checks': len(self.validation_results),
            'passed_checks': sum(1 for result in self.validation_results.values() if result.get('valid', False))
        }
        
        return summary 