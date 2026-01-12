from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.training import Training
from src.logger import logging


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    """
    Orchestrates the model training pipeline stage.
    Connects the configuration management with the Training component.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the training process:
        1. Initialize ConfigurationManager and fetch TrainingConfig.
        2. Instantiate the Training component.
        3. Load base model, setup generators, and start training.
        """
        # Step 1: Manage and fetch training configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()
        
        # Step 2: Initialize the Training component with config
        training = Training(config=training_config)
        
        # Step 3: Execute training steps
        training.get_base_model()         # Load the prepared base model
        training.train_valid_generator()  # Prepare data generators (train/val)
        training.train()                  # Perform training with re-compilation fix


if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Instantiate and run the pipeline stage
        obj = ModelTrainingPipeline()
        obj.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any errors encountered during the training pipeline
        logging.exception(e)
        raise e
