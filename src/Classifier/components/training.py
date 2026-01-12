import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.Classifier.entity.config_entity import TrainingConfig
from pathlib import Path


class Training:
    """
    Component for training the deep learning model.
    Handles loading the base model, setting up data generators, and executing the training process.
    """
    def __init__(self, config: TrainingConfig):
        """
        Initializes the training component with configuration.
        """
        self.config = config

    
    def get_base_model(self):
        """
        Loads the pre-trained updated base model from the specified path.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Creates training and validation data generators with optional data augmentation.
        Uses Keras ImageDataGenerator for flow_from_directory.
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained model to the specified path.
        """
        model.save(path)


    def train(self):
        """
        Performs the model training.
        Includes a re-compilation step to ensure fresh optimizer state and avoid "Unknown variable" errors.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Re-compiling the model to avoid common "Unknown variable"/optimizer state errors after loading
        # This is critical for compatibility when resuming training from a saved file
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
