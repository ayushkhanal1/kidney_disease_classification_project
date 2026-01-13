import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.Classifier.config.configuration import ConfigurationManager
from src.Classifier.components.training import Training
from src.logger import logging


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    """
    Orchestrates the model training pipeline stage.
    Connects the configuration management with the Training component.
    
    This class acts as the 'glue' between your settings and the actual training logic.
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
        # Step 1: Manage and fetch training configuration (paths, hyperparameters)
        config = ConfigurationManager()
        training_config = config.get_training_config()
        
        # Step 2: Initialize the Training component with the fetched config
        training = Training(config=training_config)
        
        # Step 3: Run the training workflow:
        training.get_base_model()         # 1. Load the customized VGG16 model from artifacts
        training.train_valid_generator()  # 2. Prepare the images (normalization & augmentation)
        training.train()                  # 3. Start training and save the final result


if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Instantiate and run the pipeline stage
        modeltraining = ModelTrainingPipeline()
        modeltraining.main()
        
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any errors encountered during the training pipeline
        logging.exception(e)
        raise e
