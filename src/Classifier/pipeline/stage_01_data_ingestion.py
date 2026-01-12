from src.logger import logging
from Classifier.config.configuration import ConfigurationManager
from Classifier.components.data_ingestion import dataingestion

STAGE_NAME = "Data Ingestion" 

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = dataingestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
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