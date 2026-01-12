from src.Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.logger import logging

# Entry point for the training pipeline
STAGE_NAME = "Data Ingestion"
try:
        logging.info(f"\n\n ----------{STAGE_NAME} started ------------------- \n\n")
        # Initialize the pipeline for data ingestion
        data_ingestion_pipeline=DataIngestionTrainingPipeline()
        logging.info("Data ingestion pipeline started")
        # Execute the main function of the pipeline (download and unzip)
        data_ingestion_pipeline.main()
        logging.info("Data ingestion completed")
        logging.info(f"\n\n ----------{STAGE_NAME} completed ------------------- \n\n")

except Exception as e:
        # Log the exception and re-raise it
        logging.exception(e)
        raise e




STAGE_NAME = "Prepare base model"
try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logging.exception(e)
        raise e