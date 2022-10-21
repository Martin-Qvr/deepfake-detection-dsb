from preprocess_image import images_to_keras_dataset
from deepfake_model import DeepfakeModel

if __name__ == "__main__":
    # Images to keras object
    train_set, validation_set = images_to_keras_dataset(train_path="/home/jovyan/hfactory_magic_folders/tooling_for_the_data_scientist/deepfakes_detection/train.csv",
                                                        data_dir="/home/jovyan/hfactory_magic_folders/tooling_for_the_data_scientist/deepfakes_detection/images/")
    # Deepfake model class
    df_mod = DeepfakeModel()
    # Create the model
    model = df_mod.create_model()
    # Train the model
    _ = df_mod.train_model(model, train_set, validation_set)
    # Save the model
    df_mod.save_model(model, "model")
    
    
