from model.preprocess_image import images_to_keras_dataset
from model.deepfake_model import DeepfakeModel
import numpy as np
from PIL import Image
import os
import pandas as pd


test_folder = 'test'
mock_images_folder = 'mock_data_tmp'
mock_path = 'mock.csv'


def generate_random_image(color:str = 'RGB'):
    imarray = np.random.rand(100,100,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert(color)
    return im


def dataset_creation():
    dataset_dict = {'image_id': [], 'label': []}
    # randomly generate colored images
    for i in range(10):
        im = generate_random_image()
        im.save(os.path.join(test_folder, mock_images_folder, f"deep_fake_{i}.jpg"))
        dataset_dict['image_id'].append(f"deep_fake_{i}")
        dataset_dict['label'].append(1)

    # randomly generate grayscale images
    for i in range(10):
        im = generate_random_image('L')
        im.save(os.path.join(test_folder, mock_images_folder, f"real_{i}.jpg"))
        dataset_dict['image_id'].append(f"real_{i}")
        dataset_dict['label'].append(0)

    train_df = pd.DataFrame(dataset_dict)
    train_df.to_csv(os.path.join(test_folder, mock_path))
    train_generator, validation_generator = images_to_keras_dataset(train_path=os.path.join(test_folder, mock_path), 
                                                                    data_dir=os.path.join(test_folder, mock_images_folder), batch_size=1)
    
    return train_generator, validation_generator


def test_dataset_creation():
    train_generator, validation_generator = dataset_creation()
    assert len(train_generator.labels + validation_generator.labels) == 20
    assert len(set(train_generator.labels)) == 2
    
    
def test_model_creation():
    df_mod = DeepfakeModel(batch_size=1)
    model = df_mod.create_model()
    # assert that the loss is the one we want
    assert model.loss == 'binary_crossentropy'
    
    
def test_model_training():
    train_generator, validation_generator = dataset_creation()
    df_mod = DeepfakeModel(batch_size=1)
    model = df_mod.create_model()
    initial_weights = model.get_weights()
    _ = df_mod.train_model(model, train_generator, validation_generator)
    final_weights = model.get_weights()
    # assert that each layer gets trained (at least one weight per layer needs to have changed)
    for i in range(len(initial_weights)):
        assert (final_weights[i] - initial_weights[i]).any()
        
def test_model_saving():
    df_mod = DeepfakeModel(batch_size=1)
    model = df_mod.create_model()
    df_mod.save_model(model, 'test_model')
    assert 'test_model' in os.listdir("saved_models")