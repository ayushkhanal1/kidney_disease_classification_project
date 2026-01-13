import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.evaluation import Evaluation
from src.logger import logging
import dagshub
import mlflow

dagshub.init(repo_owner='ayukhanalsh100', repo_name='kidney_disease_classification_project', mlflow=True)
with mlflow.start_run():
   mlflow.log_param('parameter name', 'value')
   mlflow.log_metric('metric name', 1)


STAGE_NAME="Evaluation"

class EvaluationTrainingPipeline:
    """
    Orchestrates the model evaluation pipeline stage.
    """
    def __init__(self):
        pass


    def main(self):
        """
        Executes the evaluation process:
        1. Initialize ConfigurationManager and fetch EvaluationConfig.
        2. Instantiate the Evaluation component.
        3. Run evaluation scoring and log into MLflow.
        """
        # Step 1: Manage and fetch evaluation configuration
        config = ConfigurationManager()
        eval_config=config.get_evaluation_config()
        
        # Step 2: Initialize the Evaluation component
        evaluation=Evaluation(eval_config)
        
        # Step 3: Start evaluation process
        evaluation.evaluation()      # Calculate and save metrics (e.g., scores.json)
        #evaluation.log_into_mlflow()  # Record experience metrics and parameters in MLflow



if __name__ == '__main__':
    try:
        logging.info("*******************")
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        
        # Initialize the pipeline for model training
        model_evaluation_pipeline = EvaluationTrainingPipeline()
        
        # Execute the main function: load model, setup generators, and train
        model_evaluation_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logging.exception(e)
        raise e
