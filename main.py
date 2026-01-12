from src.Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.Classifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.logger import logging

# ----------------- STAGE 1: Data Ingestion -----------------
# This stage handles downloading the raw data and extracting it for further processing.
STAGE_NAME = "Data Ingestion"
try:
        logging.info(f"\n\n ----------{STAGE_NAME} started ------------------- \n\n")
        # Initialize the pipeline for data ingestion
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        logging.info("Data ingestion pipeline initialized")
        
        # Execute the main function of the pipeline (downloading and unzipping)
        data_ingestion_pipeline.main()
        logging.info("Data ingestion process completed")
        logging.info(f"\n\n ----------{STAGE_NAME} completed ------------------- \n\n")

except Exception as e:
        # Log any error that occurs during this stage and re-raise the exception
        logging.exception(e)
        raise e


# ----------------- STAGE 2: Prepare Base Model -----------------
# This stage involves initializing a pre-trained model and customizing it for our classification task.
STAGE_NAME = "Prepare base model"
try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Initialize the pipeline for base model preparation
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        
        # Execute the main function: load base model and add custom output layers
        prepare_base_model_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        # Log any error that occurs during this stage
        logging.exception(e)
        raise e


# ----------------- STAGE 3: Training -----------------
# This stage handles loading the prepared model and training it on the dataset.
STAGE_NAME = "Training"
try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Initialize the pipeline for model training
        model_training_pipeline = ModelTrainingPipeline()
        
        # Execute the main function: load model, setup generators, and train
        model_training_pipeline.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        # Log any error that occurs during this stage
        logging.exception(e)
        raise e

