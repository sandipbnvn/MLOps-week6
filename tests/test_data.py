"""
Unit tests for data processing and validation components.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_processor import DataProcessor
from src.data.data_validator import DataValidator


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'input_path': 'data/iris.csv',
            'test_size': 0.4,
            'random_state': 42,
            'target_column': 'species',
            'feature_columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'stratify': True
        }
        
        # Create sample data with more samples and better distribution
        self.sample_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                            5.9, 6.0, 6.1, 5.8, 5.7, 5.9, 6.2, 6.0, 5.8, 6.1,
                            6.6, 6.7, 6.8, 6.5, 6.4, 6.7, 6.9, 6.6, 6.5, 6.8],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                           2.8, 2.9, 3.0, 2.7, 2.8, 2.9, 3.0, 2.8, 2.7, 2.9,
                           3.0, 3.1, 3.2, 2.9, 3.0, 3.1, 3.2, 3.0, 2.9, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                            4.3, 4.4, 4.5, 4.2, 4.3, 4.4, 4.5, 4.3, 4.2, 4.4,
                            5.6, 5.7, 5.8, 5.5, 5.6, 5.7, 5.8, 5.6, 5.5, 5.7],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                           1.3, 1.4, 1.5, 1.2, 1.3, 1.4, 1.5, 1.3, 1.2, 1.4,
                           2.0, 2.1, 2.2, 1.9, 2.0, 2.1, 2.2, 2.0, 1.9, 2.1],
            'species': ['setosa'] * 10 + ['versicolor'] * 10 + ['virginica'] * 10
        })
        
        self.data_processor = DataProcessor(self.config)
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with valid data path
        data_path = 'data/iris.csv'
        if os.path.exists(data_path):
            data = self.data_processor.load_data(data_path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertListEqual(list(data.columns), 
                               ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    
    def test_validate_data(self):
        """Test data validation functionality."""
        # Test with valid data
        self.data_processor.data = self.sample_data
        is_valid = self.data_processor.validate_data()
        self.assertTrue(is_valid)
        
        # Test with missing columns
        invalid_data = self.sample_data.drop('species', axis=1)
        self.data_processor.data = invalid_data
        is_valid = self.data_processor.validate_data()
        self.assertFalse(is_valid)
        
        # Test with non-numeric feature columns
        invalid_data = self.sample_data.copy()
        invalid_data['sepal_length'] = ['a'] * 30  # 30 values to match DataFrame length
        self.data_processor.data = invalid_data
        is_valid = self.data_processor.validate_data()
        self.assertFalse(is_valid)
    
    def test_split_data(self):
        """Test data splitting functionality."""
        self.data_processor.data = self.sample_data
        X_train, X_test, y_train, y_test = self.data_processor.split_data()
        
        # Check shapes
        self.assertEqual(X_train.shape[1], 4)  # 4 features
        self.assertEqual(X_test.shape[1], 4)   # 4 features
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(len(y_test), X_test.shape[0])
        
        # Check feature columns
        expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.assertListEqual(list(X_train.columns), expected_features)
        self.assertListEqual(list(X_test.columns), expected_features)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        self.data_processor.data = self.sample_data
        summary = self.data_processor.get_data_summary()
        
        self.assertIn('total_samples', summary)
        self.assertIn('features', summary)
        self.assertIn('target_distribution', summary)
        self.assertIn('feature_stats', summary)
        
        self.assertEqual(summary['total_samples'], 30)
        self.assertEqual(summary['features'], 4)
        self.assertIn('setosa', summary['target_distribution'])
        self.assertIn('versicolor', summary['target_distribution'])
        self.assertIn('virginica', summary['target_distribution'])


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'target_column': 'species',
            'feature_columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        }
        
        # Create sample data with more samples and better distribution
        self.sample_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                            5.9, 6.0, 6.1, 5.8, 5.7, 5.9, 6.2, 6.0, 5.8, 6.1,
                            6.6, 6.7, 6.8, 6.5, 6.4, 6.7, 6.9, 6.6, 6.5, 6.8],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                           2.8, 2.9, 3.0, 2.7, 2.8, 2.9, 3.0, 2.8, 2.7, 2.9,
                           3.0, 3.1, 3.2, 2.9, 3.0, 3.1, 3.2, 3.0, 2.9, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                            4.3, 4.4, 4.5, 4.2, 4.3, 4.4, 4.5, 4.3, 4.2, 4.4,
                            5.6, 5.7, 5.8, 5.5, 5.6, 5.7, 5.8, 5.6, 5.5, 5.7],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                           1.3, 1.4, 1.5, 1.2, 1.3, 1.4, 1.5, 1.3, 1.2, 1.4,
                           2.0, 2.1, 2.2, 1.9, 2.0, 2.1, 2.2, 2.0, 1.9, 2.1],
            'species': ['setosa'] * 10 + ['versicolor'] * 10 + ['virginica'] * 10
        })
        
        self.validator = DataValidator(self.config)
    
    def test_validate_schema(self):
        """Test schema validation."""
        # Test with valid schema
        is_valid = self.validator.validate_schema(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test with missing columns
        invalid_data = self.sample_data.drop('species', axis=1)
        is_valid = self.validator.validate_schema(invalid_data)
        self.assertFalse(is_valid)
        
        # Test with non-numeric feature columns
        invalid_data = self.sample_data.copy()
        invalid_data['sepal_length'] = ['a'] * 30  # 30 values to match DataFrame length
        is_valid = self.validator.validate_schema(invalid_data)
        self.assertFalse(is_valid)
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Test with valid data
        is_valid = self.validator.validate_data_quality(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test with missing values
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'sepal_length'] = np.nan
        is_valid = self.validator.validate_data_quality(invalid_data)
        self.assertFalse(is_valid)
    
    def test_validate_data_range(self):
        """Test data range validation."""
        # Test with valid ranges
        is_valid = self.validator.validate_data_range(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test with out-of-range values
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'sepal_length'] = 10.0  # Out of expected range
        is_valid = self.validator.validate_data_range(invalid_data)
        self.assertFalse(is_valid)
    
    def test_validate_target_values(self):
        """Test target values validation."""
        # Test with valid target values
        is_valid = self.validator.validate_target_values(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test with invalid target values
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'species'] = 'invalid_species'
        is_valid = self.validator.validate_target_values(invalid_data)
        self.assertFalse(is_valid)
    
    def test_run_all_validations(self):
        """Test running all validations."""
        # Test with valid data
        all_valid, results = self.validator.run_all_validations(self.sample_data)
        self.assertTrue(all_valid)
        self.assertIn('schema', results)
        self.assertIn('quality', results)
        self.assertIn('range', results)
        self.assertIn('target', results)
        
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'species'] = 'invalid_species'
        all_valid, results = self.validator.run_all_validations(invalid_data)
        self.assertFalse(all_valid)
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        self.validator.run_all_validations(self.sample_data)
        summary = self.validator.get_validation_summary()
        
        self.assertIn('all_valid', summary)
        self.assertIn('validation_results', summary)
        self.assertIn('total_checks', summary)
        self.assertIn('passed_checks', summary)
        
        self.assertTrue(summary['all_valid'])
        self.assertEqual(summary['total_checks'], 4)
        self.assertEqual(summary['passed_checks'], 4)


if __name__ == '__main__':
    unittest.main() 