from src.Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.Classifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.Classifier.pipeline.stage_04_evaluation import EvaluationTrainingPipeline
from src.logger import logging

"""
MAIN ENTRY POINT
----------------
This script orchestrates the entire end-to-end Machine Learning pipeline.
It imports and executes each stage sequentially.

Pipeline Flow:
1. Data Ingestion: Download and extract dataset.
2. Prepare Base Model: Initialize VGG16 and add custom classification head.
3. Training: Train the model on the kidney scan data.
4. Evaluation: Validate performance and log results to MLflow.
"""

# ----------------- STAGE 1: Data Ingestion -----------------
# This stage handles downloading the raw data and extracting it for further processing.
STAGE_NAME = "Data Ingestion"
try:
        logging.info(f"\n\n ---------- {STAGE_NAME} started ------------------- \n\n")
        # The 'TrainingPipeline' classes encapsulate the logic for each stage.
        # This keeps main.py clean and readable.
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        logging.info("Data ingestion pipeline initialized")
        
        # main() triggers the internal methods (download, extract) defined in the component.
        data_ingestion_pipeline.main()
        logging.info("Data ingestion process completed")
        logging.info(f"\n\n ---------- {STAGE_NAME} completed ------------------- \n\n")

except Exception as e:
        # Catching and logging exceptions ensures we know exactly which stage failed.
        logging.exception(e)
        raise e


# ----------------- STAGE 2: Prepare Base Model -----------------
# This stage involves initializing a pre-trained model (VGG16) and customizing it.
STAGE_NAME = "Prepare base model"
try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # We customized the VGG16 model here (freezing layers, adding Dense layers).
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        
        # This creates the 'base_model_updated.h5' which is ready for training.
        prepare_base_model_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logging.exception(e)
        raise e


# ----------------- STAGE 3: Training -----------------
# This stage handles loading the prepared model and training it on the dataset.
STAGE_NAME = "Training"
try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Training logic including augmentation and the optimizer fix is contained here.
        model_training_pipeline = ModelTrainingPipeline()
        
        # This produces 'model.h5' in the artifacts/training directory.
        model_training_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logging.exception(e)
        raise e


# ----------------- STAGE 4: Evaluation -----------------
# Finally, we evaluate the trained model's accuracy and log metrics to MLflow.
STAGE_NAME="Evaluation"
try:
        logging.info("*******************")
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        
        # Evaluation uses the validation data subset and logs results to scores.json and MLflow.
        model_evaluation_pipeline = EvaluationTrainingPipeline()
        
        # If tracking setup (like DagsHub) is active, results will appear in your dashboard.
        model_evaluation_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

except Exception as e:
        logging.exception(e)
        raise e



