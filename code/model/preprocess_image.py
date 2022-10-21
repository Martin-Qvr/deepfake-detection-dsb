import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Solve GPU/CPU keras warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def images_to_keras_dataset(train_path: str,
                            data_dir: str,
                            validation_split: str=0.2,
                            batch_size: int=32,
                            target_size: tuple=(150, 150)
                            ) -> keras.preprocessing.image.DataFrameIterator:
    """
    This function transforms images folder into keras train and validation set.

    Input : validation split, batch size and target size.
    Output : train and validation sets.
    """
    
    train_df = pd.read_csv(train_path, dtype=str, index_col=0)
    train_df.loc[:, "image_id"] = train_df.loc[:, "image_id"] + ".jpg"

    datagen = ImageDataGenerator(rescale=1./255,
                                 validation_split=validation_split,
    )

    train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="image_id",
    y_col="label",
    subset='training',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=target_size)

    validation_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="image_id",
    y_col="label",
    subset='validation',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=target_size)

    return train_generator, validation_generator
