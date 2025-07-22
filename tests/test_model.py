"""
Unit tests for model training and evaluation components.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.training.train_model import ModelTrainer
from src.training.model_evaluator import ModelEvaluator


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'type': 'DecisionTreeClassifier',
            'parameters': {
                'max_depth': 3,
                'random_state': 1
            },
            'artifact_path': 'artifacts/test_model.joblib'
        }
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        self.X_train = pd.DataFrame({
            'sepal_length': np.random.uniform(4.0, 8.0, n_samples),
            'sepal_width': np.random.uniform(2.0, 5.0, n_samples),
            'petal_length': np.random.uniform(1.0, 7.0, n_samples),
            'petal_width': np.random.uniform(0.1, 2.5, n_samples)
        })
        
        # Create target with some pattern
        self.y_train = pd.Series(['setosa'] * 33 + ['versicolor'] * 33 + ['virginica'] * 34)
        
        self.trainer = ModelTrainer(self.config)
    
    def test_create_model(self):
        """Test model creation."""
        model = self.trainer.create_model()
        self.assertIsNotNone(model)
        self.assertEqual(self.trainer.model_info['type'], 'DecisionTreeClassifier')
        self.assertEqual(self.trainer.model_info['parameters']['max_depth'], 3)
    
    def test_train_model(self):
        """Test model training."""
        self.trainer.create_model()
        model = self.trainer.train_model(self.X_train, self.y_train)
        
        self.assertIsNotNone(model)
        self.assertIn('training_samples', self.trainer.model_info)
        self.assertIn('training_features', self.trainer.model_info)
        self.assertIn('target_distribution', self.trainer.model_info)
        
        self.assertEqual(self.trainer.model_info['training_samples'], 100)
        self.assertEqual(self.trainer.model_info['training_features'], 4)
    
    def test_cross_validate(self):
        """Test cross-validation."""
        self.trainer.create_model()
        self.trainer.train_model(self.X_train, self.y_train)
        
        cv_results = self.trainer.cross_validate(self.X_train, self.y_train, cv=3)
        
        self.assertIn('mean_accuracy', cv_results)
        self.assertIn('std_accuracy', cv_results)
        self.assertIn('cv_scores', cv_results)
        self.assertIn('cv_folds', cv_results)
        
        self.assertGreater(cv_results['mean_accuracy'], 0)
        self.assertLessEqual(cv_results['mean_accuracy'], 1)
        self.assertEqual(cv_results['cv_folds'], 3)
        self.assertEqual(len(cv_results['cv_scores']), 3)
    
    def test_predict(self):
        """Test model prediction."""
        self.trainer.create_model()
        self.trainer.train_model(self.X_train, self.y_train)
        
        # Create test data
        X_test = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.3],
            'sepal_width': [3.5, 3.0, 3.3],
            'petal_length': [1.4, 1.4, 6.0],
            'petal_width': [0.2, 0.2, 2.5]
        })
        
        predictions = self.trainer.predict(X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(pred in ['setosa', 'versicolor', 'virginica'] for pred in predictions))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        self.trainer.create_model()
        self.trainer.train_model(self.X_train, self.y_train)
        
        X_test = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'sepal_width': [3.5, 3.0],
            'petal_length': [1.4, 1.4],
            'petal_width': [0.2, 0.2]
        })
        
        probabilities = self.trainer.predict_proba(X_test)
        
        if probabilities is not None:
            self.assertIsInstance(probabilities, np.ndarray)
            self.assertEqual(probabilities.shape[0], 2)
            self.assertEqual(probabilities.shape[1], 3)  # 3 classes
            self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        self.trainer.create_model()
        self.trainer.train_model(self.X_train, self.y_train)
        
        # Save model
        model_path = self.trainer.save_model()
        self.assertTrue(os.path.exists(model_path))
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(self.config)
        loaded_model = new_trainer.load_model(model_path)
        
        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(new_trainer.model)
        
        # Test that loaded model makes same predictions
        X_test = pd.DataFrame({
            'sepal_length': [5.1],
            'sepal_width': [3.5],
            'petal_length': [1.4],
            'petal_width': [0.2]
        })
        
        original_pred = self.trainer.predict(X_test)
        loaded_pred = new_trainer.predict(X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        self.trainer.create_model()
        self.trainer.train_model(self.X_train, self.y_train)
        
        info = self.trainer.get_model_info()
        
        self.assertIn('type', info)
        self.assertIn('parameters', info)
        self.assertIn('model_type', info)
        self.assertIn('has_proba', info)
        self.assertIn('training_samples', info)
        self.assertIn('training_features', info)
        
        self.assertEqual(info['type'], 'DecisionTreeClassifier')
        self.assertEqual(info['model_type'], 'DecisionTreeClassifier')
        self.assertTrue(info['has_proba'])
    
    def test_train_pipeline(self):
        """Test complete training pipeline."""
        X_test = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.3],
            'sepal_width': [3.5, 3.0, 3.3],
            'petal_length': [1.4, 1.4, 6.0],
            'petal_width': [0.2, 0.2, 2.5]
        })
        y_test = pd.Series(['setosa', 'setosa', 'virginica'])
        
        results = self.trainer.train_pipeline(self.X_train, self.y_train, X_test, y_test)
        
        self.assertIn('model_info', results)
        self.assertIn('cv_results', results)
        self.assertIn('model_path', results)
        self.assertIn('validation_accuracy', results)
        self.assertIn('training_samples', results)
        self.assertIn('training_features', results)
        
        self.assertGreater(results['validation_accuracy'], 0)
        self.assertLessEqual(results['validation_accuracy'], 1)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'metrics': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            'threshold_accuracy': 0.95,
            'metrics_path': 'artifacts/test_metrics.json'
        }
        
        # Create sample predictions
        self.y_true = pd.Series(['setosa', 'versicolor', 'virginica', 'setosa', 'versicolor'])
        self.y_pred = np.array(['setosa', 'versicolor', 'virginica', 'setosa', 'versicolor'])
        self.y_proba = np.array([
            [0.9, 0.05, 0.05],  # setosa
            [0.1, 0.8, 0.1],    # versicolor
            [0.05, 0.1, 0.85],  # virginica
            [0.85, 0.1, 0.05],  # setosa
            [0.1, 0.75, 0.15]   # versicolor
        ])
        
        self.evaluator = ModelEvaluator(self.config)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred, self.y_proba)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('precision_setosa', metrics)
        self.assertIn('recall_setosa', metrics)
        self.assertIn('f1_setosa', metrics)
        
        # Perfect predictions should give accuracy = 1.0
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision_macro'], 1.0)
        self.assertEqual(metrics['recall_macro'], 1.0)
        self.assertEqual(metrics['f1_macro'], 1.0)
    
    def test_generate_classification_report(self):
        """Test classification report generation."""
        report = self.evaluator.generate_classification_report(self.y_true, self.y_pred)
        
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)
        self.assertIn('setosa', report)
        self.assertIn('versicolor', report)
        self.assertIn('virginica', report)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        cm = self.evaluator.generate_confusion_matrix(self.y_true, self.y_pred)
        
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (3, 3))  # 3x3 for 3 classes
        
        # Perfect predictions should give diagonal matrix
        expected_cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])  # 2 setosa, 2 versicolor, 1 virginica
        np.testing.assert_array_equal(cm, expected_cm)
    
    def test_evaluate_model(self):
        """Test complete model evaluation."""
        results = self.evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba)
        
        self.assertIn('metrics', results)
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('meets_threshold', results)
        self.assertIn('threshold_accuracy', results)
        self.assertIn('evaluation_timestamp', results)
        self.assertIn('sample_count', results)
        
        self.assertTrue(results['meets_threshold'])  # accuracy = 1.0 > 0.95
        self.assertEqual(results['threshold_accuracy'], 0.95)
        self.assertEqual(results['sample_count'], 5)
    
    def test_save_and_load_evaluation_results(self):
        """Test saving and loading evaluation results."""
        results = self.evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba)
        
        # Save results
        output_path = self.evaluator.save_evaluation_results(results)
        self.assertTrue(os.path.exists(output_path))
        
        # Load results
        loaded_results = self.evaluator.load_evaluation_results(output_path)
        
        self.assertIn('metrics', loaded_results)
        self.assertIn('classification_report', loaded_results)
        self.assertIn('confusion_matrix', loaded_results)
        self.assertIn('meets_threshold', loaded_results)
        
        # Check that loaded results match original
        self.assertEqual(loaded_results['metrics']['accuracy'], results['metrics']['accuracy'])
        self.assertEqual(loaded_results['meets_threshold'], results['meets_threshold'])
    
    def test_get_evaluation_summary(self):
        """Test evaluation summary generation."""
        self.evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba)
        summary = self.evaluator.get_evaluation_summary()
        
        self.assertIn('accuracy', summary)
        self.assertIn('precision_macro', summary)
        self.assertIn('recall_macro', summary)
        self.assertIn('f1_macro', summary)
        self.assertIn('meets_threshold', summary)
        self.assertIn('sample_count', summary)
        self.assertIn('evaluation_timestamp', summary)
        
        self.assertEqual(summary['accuracy'], 1.0)
        self.assertTrue(summary['meets_threshold'])
        self.assertEqual(summary['sample_count'], 5)
    
    def test_evaluate_model_with_imperfect_predictions(self):
        """Test evaluation with imperfect predictions."""
        # Create imperfect predictions
        imperfect_y_pred = np.array(['setosa', 'versicolor', 'virginica', 'versicolor', 'setosa'])
        
        results = self.evaluator.evaluate_model(self.y_true, imperfect_y_pred, self.y_proba)
        
        self.assertLess(results['metrics']['accuracy'], 1.0)
        self.assertFalse(results['meets_threshold'])  # accuracy < 0.95


if __name__ == '__main__':
    unittest.main() 