import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Component for downloading and customizing a pre-trained model.
    It uses VGG16 as an example and adds custom top layers for the kidney disease classification.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the component with relevant configuration.
        """
        self.config = config

    
    def get_base_model(self):
        """
        Loads the pre-trained VGG16 model with the specified image size and weights.
        The top layers (fully connected part) can be excluded via configuration.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Saves the raw base model for future reference or reuse
        self.save_model(path=self.config.base_model_path, model=self.model)

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        A helper function to customize the architecture:
        - Freezes specified layers.
        - Flattens the output of the base model.
        - Adds a Dense layer for prediction.
        - Compiles the final model.
        """
        # Freeze base layers to prevent retraining if needed
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Build custom classification layers on top of the base model output
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Create the full model integrating the base and our custom classification head
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model with SGD optimizer and categorical crossentropy loss
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        """
        Triggers the creation of the full model using updated parameters (e.g., classes, LR).
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Saves the newly updated model ready for training
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves a Keras model to the specified path.
        """
        model.save(path)
