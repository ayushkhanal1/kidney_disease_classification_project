from Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.logger import logging

STAGE_NAME = "Data Ingestion"
try:
        logging.info(f"\n\n ----------{STAGE_NAME} started ------------------- \n\n")
        data_ingestion_pipeline=DataIngestionTrainingPipeline()
        logging.info("Data ingestion pipeline started")
        data_ingestion_pipeline.main()
        logging.info("Data ingestion completed")
        logging.info(f"\n\n ----------{STAGE_NAME} completed ------------------- \n\n")

except Exception as e:
        logging.exception(e)
        raise e