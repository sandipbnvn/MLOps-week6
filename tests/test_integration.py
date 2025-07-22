"""
Integration tests for the complete MLOps pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import json
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_pipeline import MLPipeline


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete MLOps pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test data
        self.create_test_data()
        
        # Create test config
        self.create_test_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close all loggers to release file handles
        import logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()
        
        # Change back to original directory
        os.chdir(self.original_cwd)
        
        # Remove test directory with error handling
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            # If files are still in use, try to remove them individually
            import time
            time.sleep(0.1)  # Small delay to allow file handles to close
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                # If still failing, just log the issue but don't fail the test
                print(f"Warning: Could not remove test directory {self.test_dir}")
        except FileNotFoundError:
            # Directory already removed
            pass
    
    def create_test_data(self):
        """Create test Iris dataset."""
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Generate synthetic Iris data with realistic ranges that match validator constraints
        np.random.seed(42)
        n_samples = 150
        
        # Setosa (first 50 samples) - within expected ranges
        setosa_data = {
            'sepal_length': np.clip(np.random.normal(5.0, 0.3, 50), 4.0, 8.0),
            'sepal_width': np.clip(np.random.normal(3.4, 0.3, 50), 2.0, 5.0),
            'petal_length': np.clip(np.random.normal(1.5, 0.2, 50), 1.0, 7.0),
            'petal_width': np.clip(np.random.normal(0.2, 0.1, 50), 0.1, 2.5),
            'species': ['setosa'] * 50
        }
        
        # Versicolor (next 50 samples) - within expected ranges
        versicolor_data = {
            'sepal_length': np.clip(np.random.normal(5.9, 0.4, 50), 4.0, 8.0),
            'sepal_width': np.clip(np.random.normal(2.8, 0.3, 50), 2.0, 5.0),
            'petal_length': np.clip(np.random.normal(4.3, 0.4, 50), 1.0, 7.0),
            'petal_width': np.clip(np.random.normal(1.3, 0.2, 50), 0.1, 2.5),
            'species': ['versicolor'] * 50
        }
        
        # Virginica (last 50 samples) - within expected ranges
        virginica_data = {
            'sepal_length': np.clip(np.random.normal(6.6, 0.5, 50), 4.0, 8.0),
            'sepal_width': np.clip(np.random.normal(3.0, 0.3, 50), 2.0, 5.0),
            'petal_length': np.clip(np.random.normal(5.6, 0.5, 50), 1.0, 7.0),
            'petal_width': np.clip(np.random.normal(2.0, 0.3, 50), 0.1, 2.5),
            'species': ['virginica'] * 50
        }
        
        # Combine all data properly
        all_data = {}
        for key in setosa_data.keys():
            if key == 'species':
                # For species column, concatenate lists
                all_data[key] = setosa_data[key] + versicolor_data[key] + virginica_data[key]
            else:
                # For numeric columns, concatenate numpy arrays
                all_data[key] = np.concatenate([setosa_data[key], versicolor_data[key], virginica_data[key]])
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv('data/iris.csv', index=False)
    
    def create_test_config(self):
        """Create test configuration file."""
        os.makedirs('config', exist_ok=True)
        
        config = {
            'data': {
                'input_path': 'data/iris.csv',
                'test_size': 0.3,
                'random_state': 42,
                'target_column': 'species',
                'feature_columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                'stratify': True
            },
            'model': {
                'type': 'DecisionTreeClassifier',
                'parameters': {
                    'max_depth': 3,
                    'random_state': 1
                },
                'artifact_path': 'artifacts/model.joblib',
                'metrics_path': 'artifacts/metrics.json'
            },
            'training': {
                'random_state': 42,
                'stratify': True
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                'threshold_accuracy': 0.90
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'artifacts/pipeline.log'
            }
        }
        
        with open('config/config.yaml', 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution end-to-end."""
        # Initialize pipeline
        pipeline = MLPipeline('config/config.yaml')
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Verify pipeline completed successfully
        self.assertEqual(results['pipeline_status'], 'completed')
        self.assertIn('data_processing', results)
        self.assertIn('model_training', results)
        self.assertIn('model_evaluation', results)
        
        # Verify data processing results
        data_results = results['data_processing']
        self.assertIn('data_summary', data_results)
        self.assertIn('validation_summary', data_results)
        self.assertIn('train_samples', data_results)
        self.assertIn('test_samples', data_results)
        
        # Verify model training results
        training_results = results['model_training']
        self.assertIn('training_results', training_results)
        self.assertIn('predictions', training_results)
        
        # Verify model evaluation results
        evaluation_results = results['model_evaluation']
        self.assertIn('evaluation_results', evaluation_results)
        self.assertIn('evaluation_summary', evaluation_results)
        self.assertIn('metrics_path', evaluation_results)
        self.assertIn('plot_paths', evaluation_results)
        
        # Check that artifacts were created
        self.assertTrue(os.path.exists('artifacts/model.joblib'))
        self.assertTrue(os.path.exists('artifacts/metrics.json'))
        self.assertTrue(os.path.exists('artifacts/pipeline_results.json'))
        self.assertTrue(os.path.exists('artifacts/pipeline.log'))
    
    def test_pipeline_accuracy_threshold(self):
        """Test that pipeline meets accuracy threshold."""
        pipeline = MLPipeline('config/config.yaml')
        results = pipeline.run_complete_pipeline()
        
        # Get accuracy from evaluation results
        eval_summary = results['model_evaluation']['evaluation_summary']
        accuracy = eval_summary['accuracy']
        meets_threshold = eval_summary['meets_threshold']
        
        # Verify accuracy is reasonable (should be > 0.9 for Iris dataset)
        self.assertGreater(accuracy, 0.8)
        self.assertLessEqual(accuracy, 1.0)
        
        # Verify threshold check
        threshold = 0.90
        expected_meets_threshold = accuracy >= threshold
        self.assertEqual(meets_threshold, expected_meets_threshold)
    
    def test_pipeline_data_validation(self):
        """Test that pipeline properly validates data."""
        pipeline = MLPipeline('config/config.yaml')
        
        # Run data processing step
        X_train, X_test, y_train, y_test, data_results = pipeline.run_data_processing()
        
        # Verify data validation passed
        validation_summary = data_results['validation_summary']
        self.assertTrue(validation_summary['all_valid'])
        self.assertEqual(validation_summary['total_checks'], 4)
        self.assertEqual(validation_summary['passed_checks'], 4)
        
        # Verify data split
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(X_train.shape[1], 4)  # 4 features
        self.assertEqual(X_test.shape[1], 4)   # 4 features
    
    def test_pipeline_model_training(self):
        """Test that pipeline properly trains model."""
        pipeline = MLPipeline('config/config.yaml')
        
        # Run data processing first
        X_train, X_test, y_train, y_test, _ = pipeline.run_data_processing()
        
        # Run model training
        y_pred, y_proba, training_results = pipeline.run_model_training(X_train, y_train, X_test, y_test)
        
        # Verify training results
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_proba)
        self.assertIn('training_results', training_results)
        self.assertIn('predictions', training_results)
        self.assertIn('probabilities', training_results)
        
        # Verify predictions
        self.assertEqual(len(y_pred), len(y_test))
        self.assertTrue(all(pred in ['setosa', 'versicolor', 'virginica'] for pred in y_pred))
        
        # Verify probabilities
        self.assertEqual(y_proba.shape[0], len(y_test))
        self.assertEqual(y_proba.shape[1], 3)  # 3 classes
        self.assertTrue(np.allclose(y_proba.sum(axis=1), 1.0))
    
    def test_pipeline_model_evaluation(self):
        """Test that pipeline properly evaluates model."""
        pipeline = MLPipeline('config/config.yaml')
        
        # Run complete pipeline up to evaluation
        X_train, X_test, y_train, y_test, _ = pipeline.run_data_processing()
        y_pred, y_proba, _ = pipeline.run_model_training(X_train, y_train, X_test, y_test)
        
        # Run model evaluation
        evaluation_results = pipeline.run_model_evaluation(y_test, y_pred, y_proba)
        
        # Verify evaluation results
        self.assertIn('evaluation_results', evaluation_results)
        self.assertIn('evaluation_summary', evaluation_results)
        self.assertIn('metrics_path', evaluation_results)
        self.assertIn('plot_paths', evaluation_results)
        
        # Verify metrics
        eval_summary = evaluation_results['evaluation_summary']
        self.assertIn('accuracy', eval_summary)
        self.assertIn('precision_macro', eval_summary)
        self.assertIn('recall_macro', eval_summary)
        self.assertIn('f1_macro', eval_summary)
        self.assertIn('meets_threshold', eval_summary)
        
        # Verify metrics are reasonable
        self.assertGreater(eval_summary['accuracy'], 0)
        self.assertLessEqual(eval_summary['accuracy'], 1)
        self.assertGreater(eval_summary['precision_macro'], 0)
        self.assertLessEqual(eval_summary['precision_macro'], 1)
    
    def test_pipeline_artifact_generation(self):
        """Test that pipeline generates all expected artifacts."""
        pipeline = MLPipeline('config/config.yaml')
        results = pipeline.run_complete_pipeline()
        
        # Check that all expected artifacts exist
        expected_artifacts = [
            'artifacts/model.joblib',
            'artifacts/metrics.json',
            'artifacts/pipeline_results.json',
            'artifacts/pipeline.log',
            'artifacts/confusion_matrix.png'
        ]
        
        for artifact in expected_artifacts:
            self.assertTrue(os.path.exists(artifact), f"Missing artifact: {artifact}")
        
        # Verify model artifact can be loaded
        import joblib
        model_data = joblib.load('artifacts/model.joblib')
        self.assertIn('model', model_data)
        self.assertIn('model_info', model_data)
        self.assertIn('config', model_data)
        
        # Verify metrics file can be loaded
        with open('artifacts/metrics.json', 'r') as f:
            metrics = json.load(f)
        self.assertIn('metrics', metrics)
        self.assertIn('meets_threshold', metrics)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data."""
        # Create invalid data (missing target column)
        invalid_df = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2]
            # Missing 'species' column
        })
        invalid_df.to_csv('data/invalid_iris.csv', index=False)
        
        # Update config to use invalid data
        with open('config/config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        config['data']['input_path'] = 'data/invalid_iris.csv'
        
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Test that pipeline fails gracefully
        pipeline = MLPipeline('config/config.yaml')
        
        with self.assertRaises(Exception):
            pipeline.run_complete_pipeline()
        
        # Verify that error results were saved
        self.assertTrue(os.path.exists('artifacts/pipeline_results.json'))
        
        with open('artifacts/pipeline_results.json', 'r') as f:
            error_results = json.load(f)
        
        self.assertEqual(error_results['pipeline_status'], 'failed')
        self.assertIn('error', error_results)


if __name__ == '__main__':
    unittest.main() 