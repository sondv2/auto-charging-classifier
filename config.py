from fine_tune_xception import fine_tune
import os
# model_type = 'ResNet50'
# model_type = 'EfficientNetB3'
model_type = 'Xception'
# model_type = 'define'
project_name = 'auto-charging-classifier'
train_data = 'data/train/'
valid_data = 'data/valid/'
model_save = 'model_save/'
n_class = 5
epochs = 50
batch_size = 32
image_width, image_height = 224, 224
inputshape = (image_width, image_height, 3)
inputsize = (image_width, image_height)
gpus = 8
multi_gpu_flag = True
lr = 1e-3
train_split = 0.75

BASE_DIR = './'

MODEL_DIR = os.path.join(BASE_DIR, 'model_save')
MODEL_FILE = os.path.join(MODEL_DIR, 'model_fold_{}.h5')

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
TEST_FILE = os.path.join(DATA_DIR, 'test.json')

RESULT_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'outputs')
RESULT_FILE = os.path.join(OUTPUT_DIR, 'submission_{}.csv')
