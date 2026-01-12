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
        This model includes the VGG16 base and your custom classification layers.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Creates training and validation data generators with optional data augmentation.
        Uses Keras ImageDataGenerator for flow_from_directory.
        """
        # Arguments common to both training and validation generators
        datagenerator_kwargs = dict(
            rescale = 1./255,          # Normalize pixel values from [0, 255] to [0, 1]
            validation_split=0.20      # Reserve 20% of the images for verification (validation)
        )

        # Arguments controlling the flow of images during training
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # Resize images to match model input (e.g., 224x224)
            batch_size=self.config.params_batch_size,       # Process this many images at once
            interpolation="bilinear"                        # Method used for resizing images
        )

        # 1. Setup the validation generator (no augmentation, only resizing/scaling)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # 2. Setup the training generator (optionally includes random transformations)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,       # Randomly rotate images up to 40 degrees
                horizontal_flip=True,    # Mirror images horizontally
                width_shift_range=0.2,   # Shift image horizontally
                height_shift_range=0.2,  # Shift image vertically
                shear_range=0.2,         # Apply shear transformation
                zoom_range=0.2,          # Randomly zoom in/out
                **datagenerator_kwargs
            )
        else:
            # If no augmentation, use the simple scaling generator
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,                # Shuffle images to help the model generalize
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
        # Calculate how many steps (batches) are needed to see the whole data in one epoch
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # CRITICAL FIX FOR NOTBOOK ERROR: 
        # When loading a saved model for training, TensorFlow sometimes misses the optimizer variables.
        # Re-compiling the model here with a fresh SGD optimizer instance solves the "Unknown variable" issue.
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
        )

        # Start the training process
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the finalized model after training is complete
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
