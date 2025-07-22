#!/usr/bin/env python3
"""
Main pipeline runner for the Iris classification MLOps pipeline.
"""

import sys
import os
import argparse
from typing import Dict, Any
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.data_processor import DataProcessor
from src.data.data_validator import DataValidator
from src.training.train_model import ModelTrainer
from src.training.model_evaluator import ModelEvaluator


class MLPipeline:
    """Main MLOps pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        
        # Setup logging
        logging_config = self.config.get_logging_config()
        self.logger = setup_logger(
            name="ml_pipeline",
            level=logging_config.get('level', 'INFO'),
            log_file=logging_config.get('file', 'artifacts/pipeline.log')
        )
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.get_data_config())
        self.data_validator = DataValidator(self.config.get_data_config())
        self.model_trainer = ModelTrainer(self.config.get_model_config())
        self.model_evaluator = ModelEvaluator(self.config.get_evaluation_config())
        
        self.pipeline_results = {}
    
    def run_data_processing(self) -> Dict[str, Any]:
        """
        Run the data processing pipeline.
        
        Returns:
            Dictionary with data processing results
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING DATA PROCESSING PIPELINE")
        self.logger.info("=" * 50)
        
        try:
            # Get data path from config
            data_path = self.config.get('data.input_path', 'data/iris.csv')
            
            # Process data
            X_train, X_test, y_train, y_test = self.data_processor.process_pipeline(data_path)
            
            # Validate data
            all_valid, validation_results = self.data_validator.run_all_validations(self.data_processor.data)
            
            if not all_valid:
                self.logger.error("Data validation failed")
                raise ValueError("Data validation failed")
            
            # Get data summary
            data_summary = self.data_processor.get_data_summary()
            validation_summary = self.data_validator.get_validation_summary()
            
            results = {
                'data_summary': data_summary,
                'validation_summary': validation_summary,
                'validation_results': validation_results,
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            }
            
            self.pipeline_results['data_processing'] = results
            self.logger.info("Data processing pipeline completed successfully")
            
            return X_train, X_test, y_train, y_test, results
            
        except Exception as e:
            self.logger.error(f"Data processing pipeline failed: {e}")
            raise
    
    def run_model_training(self, X_train: Any, y_train: Any, 
                          X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Run the model training pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING MODEL TRAINING PIPELINE")
        self.logger.info("=" * 50)
        
        try:
            # Train model
            training_results = self.model_trainer.train_pipeline(X_train, y_train, X_test, y_test)
            
            # Make predictions on test set
            y_pred = self.model_trainer.predict(X_test)
            y_proba = self.model_trainer.predict_proba(X_test)
            
            results = {
                'training_results': training_results,
                'predictions': y_pred.tolist() if y_pred is not None else None,
                'probabilities': y_proba.tolist() if y_proba is not None else None
            }
            
            self.pipeline_results['model_training'] = results
            self.logger.info("Model training pipeline completed successfully")
            
            return y_pred, y_proba, results
            
        except Exception as e:
            self.logger.error(f"Model training pipeline failed: {e}")
            raise
    
    def run_model_evaluation(self, y_test: Any, y_pred: Any, 
                           y_proba: Any = None) -> Dict[str, Any]:
        """
        Run the model evaluation pipeline.
        
        Args:
            y_test: True test labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING MODEL EVALUATION PIPELINE")
        self.logger.info("=" * 50)
        
        try:
            # Evaluate model
            evaluation_results = self.model_evaluator.evaluate_model(y_test, y_pred, y_proba)
            
            # Save evaluation results
            metrics_path = self.model_evaluator.save_evaluation_results(evaluation_results)
            
            # Create evaluation plots
            plot_paths = self.model_evaluator.create_evaluation_plots(y_test, y_pred, y_proba)
            
            # Get evaluation summary
            evaluation_summary = self.model_evaluator.get_evaluation_summary()
            
            results = {
                'evaluation_results': evaluation_results,
                'evaluation_summary': evaluation_summary,
                'metrics_path': metrics_path,
                'plot_paths': plot_paths
            }
            
            self.pipeline_results['model_evaluation'] = results
            self.logger.info("Model evaluation pipeline completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation pipeline failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline.
        
        Returns:
            Dictionary with all pipeline results
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING COMPLETE ML PIPELINE")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Data Processing
            X_train, X_test, y_train, y_test, data_results = self.run_data_processing()
            
            # Step 2: Model Training
            y_pred, y_proba, training_results = self.run_model_training(X_train, y_train, X_test, y_test)
            
            # Step 3: Model Evaluation
            evaluation_results = self.run_model_evaluation(y_test, y_pred, y_proba)
            
            # Prepare final results
            final_results = {
                'pipeline_status': 'completed',
                'pipeline_timestamp': self.config.get('timestamp', ''),
                'data_processing': data_results,
                'model_training': training_results,
                'model_evaluation': evaluation_results
            }
            
            # Save complete pipeline results
            self.save_pipeline_results(final_results)
            
            self.logger.info("=" * 50)
            self.logger.info("COMPLETE ML PIPELINE FINISHED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Complete pipeline failed: {e}")
            final_results = {
                'pipeline_status': 'failed',
                'error': str(e),
                'pipeline_timestamp': self.config.get('timestamp', '')
            }
            self.save_pipeline_results(final_results)
            raise
    
    def save_pipeline_results(self, results: Dict[str, Any]) -> str:
        """
        Save complete pipeline results to file.
        
        Args:
            results: Pipeline results dictionary
            
        Returns:
            Path where results were saved
        """
        output_path = 'artifacts/pipeline_results.json'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline results saved to {output_path}")
        return output_path
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline results.
        
        Returns:
            Dictionary with pipeline summary
        """
        if not self.pipeline_results:
            return {'status': 'No pipeline results available'}
        
        summary = {
            'status': 'completed',
            'data_processing': self.pipeline_results.get('data_processing', {}),
            'model_training': self.pipeline_results.get('model_training', {}),
            'model_evaluation': self.pipeline_results.get('model_evaluation', {})
        }
        
        # Extract key metrics
        if 'model_evaluation' in summary and 'evaluation_summary' in summary['model_evaluation']:
            eval_summary = summary['model_evaluation']['evaluation_summary']
            summary['accuracy'] = eval_summary.get('accuracy', 0)
            summary['meets_threshold'] = eval_summary.get('meets_threshold', False)
        
        return summary


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='MLOps Pipeline for Iris Classification')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-only', action='store_true',
                       help='Run only data processing pipeline')
    parser.add_argument('--training-only', action='store_true',
                       help='Run only model training pipeline')
    parser.add_argument('--evaluation-only', action='store_true',
                       help='Run only model evaluation pipeline')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = MLPipeline(args.config)
        
        if args.data_only:
            # Run only data processing
            X_train, X_test, y_train, y_test, results = pipeline.run_data_processing()
            print("Data processing completed successfully")
            print(f"Results: {results}")
            
        elif args.training_only:
            # Run only model training (requires existing data)
            print("Training-only mode requires pre-processed data")
            print("Please run the complete pipeline first")
            
        elif args.evaluation_only:
            # Run only model evaluation (requires existing model and data)
            print("Evaluation-only mode requires trained model and test data")
            print("Please run the complete pipeline first")
            
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline()
            summary = pipeline.get_pipeline_summary()
            
            print("\n" + "=" * 50)
            print("PIPELINE SUMMARY")
            print("=" * 50)
            print(f"Status: {summary['status']}")
            if 'accuracy' in summary:
                print(f"Model Accuracy: {summary['accuracy']:.4f}")
                print(f"Meets Threshold: {summary['meets_threshold']}")
            print("=" * 50)
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 