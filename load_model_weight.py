import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from data_size import load_data_train_size
from evaluation import evaluate
from train import train_model, model_type, project_name, train_data, valid_data, NUM_TRAIN, NUM_VALID, batch_size

epochs = 20


def load_reconstructed_model(model_type='VGG16', method=0):
    if method == 0:
        loaded_model = keras.models.load_model("model_save/%s_%s_modelArchitecture.h5" % (model_type, project_name))
        return loaded_model
    else:
        json_file = open('model_save/%s_%s_model_json.json' % (model_type, project_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_save/%s_%s_modelWeights.h5" % (model_type, project_name))
        return loaded_model