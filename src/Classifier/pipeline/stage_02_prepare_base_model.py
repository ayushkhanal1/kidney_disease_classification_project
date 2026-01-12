from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.prepare_base_model import PrepareBaseModel
from src.logger import logging


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    """
    Orchestrates the base model preparation.
    This includes loading the pre-trained architecture and updating it for our specific task.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the base model preparation steps:
        1. Initialize ConfigurationManager.
        2. Retrieve the model preparation configuration.
        3. Use the PrepareBaseModel component to fetch and update the model.
        """
        # Step 1: Manage and fetch the configuration
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        
        # Step 2: Prepare the base model using the component
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Step 3: Fetch the original pre-trained model (e.g., VGG16)
        prepare_base_model.get_base_model()
        
        # Step 4: Add custom layers and compile the model
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Instantiate and run the pipeline
        pipeline_obj = PrepareBaseModelTrainingPipeline()
        pipeline_obj.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log error details if the pipeline fails
        logging.exception(e)
        raise e
