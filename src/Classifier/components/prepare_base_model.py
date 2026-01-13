import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Component for downloading and customizing a pre-trained model.
    
    This component implements 'Transfer Learning'. We take a model that already knows 
    how to see (VGG16 trained on ImageNet) and adapt it to our kidney scan data.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the component with relevant configuration.
        """
        self.config = config

    
    def get_base_model(self):
        """
        Loads the pre-trained VGG16 model.
        
        VGG16 is a famous deep learning architecture. 'imagenet' weights mean it 
        already has 'knowledge' from millions of generic images.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top # Usually False, so we can add our own classification head
        )

        # Saves the raw base model for future reference or reuse
        self.save_model(path=self.config.base_model_path, model=self.model)

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Technical Core: Adapting the model for our specific task.
        
        Why do we freeze layers? 
        The early layers of VGG16 detect basic shapes (edges, circles). We 'freeze' 
        them so we don't destroy this knowledge during training.
        """
        # Freeze base layers to prevent retraining if needed
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # 2. Flattening: Converts the 2D visual features into a 1D vector.
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # 3. Dense Layer: The actual 'brain' that decides the class.
        # 'softmax' activation converts the output into probabilities for each class.
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
        Orchestrates the creation of the final trainable model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,              # Freeze VGG16 layers (Transfer Learning)
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
