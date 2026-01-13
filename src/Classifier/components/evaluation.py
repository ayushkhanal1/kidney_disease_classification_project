import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.Classifier.utils.common import save_json
from src.Classifier.entity.config_entity import EvaluationConfig


class Evaluation:
    """
    Component for evaluating the performance of the trained model.
    Handles data generation for validation, model loading, scoring, and MLflow logging.
    """
    def __init__(self, config: EvaluationConfig):
        """
        Initializes the evaluation component with configuration.
        """
        self.config = config

    
    def _valid_generator(self):
        """
        Setup the validation data generator.
        (Internal method called during the evaluation process)
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,          # Scale pixel values [0-255] to [0-1]
            validation_split=0.30      # Use 30% of data for validation if split isn't pre-defined
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # Match model's input resolution
            batch_size=self.config.params_batch_size,       # Number of images to process at once
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,             # Keep images in order for evaluation reproducibility
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Static helper method to load a Keras model from the disk.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """
        Executes the evaluation process:
        1. Loads the model.
        2. Prepares the validation generator.
        3. Calculates loss and accuracy scores.
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Saves the resulting evaluation metrics to a local JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        Logs the evaluation results and parameters into MLflow for experiment tracking.
        Handles both local and remote (like DagsHub) tracking URIs.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Log hyperparameters used for this run
            mlflow.log_params(self.config.all_params)
            
            # Log the final evaluation metrics
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Logic for registering the model in MLflow's Model Registry
            # Model registry requires a remote database connection; it doesn't work with simple 'file' storage.
            if tracking_url_type_store != "file":
                # Register the model with a name
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                # Just log the model artifact locally
                mlflow.keras.log_model(self.model, "model")
