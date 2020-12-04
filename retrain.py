import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from data_size import load_data_train_size
from evaluation import evaluate
from train import train_model, model_type, project_name, train_data, valid_data, NUM_TRAIN, NUM_VALID, batch_size
from load_model_weight import load_reconstructed_model

if __name__ == "__main__":

    # It can be used to reconstruct the model identically.

    reconstructed_model = load_reconstructed_model()

    print('Model compiled!')
    print(reconstructed_model.summary())

    # copy code training from train.py to here !!!

