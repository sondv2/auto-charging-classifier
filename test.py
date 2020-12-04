import os

import cv2 as cv
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pandas as pd

from load_model_weight import load_reconstructed_model
from train import model_type

predictions_model = []
model = load_reconstructed_model(model_type, 0)
filenames = os.listdir('data/test/')

for filename in tqdm(filenames):
    bgr_img = cv.imread(os.path.join('data/test/', filename))
    rgb_img = cv.resize(cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB) / 255, (224, 224))
    rgb_img = np.expand_dims(rgb_img, 0)
    predicts = model.predict(rgb_img)
    class_id = np.argmax(predicts)
    predictions_model.append({'filename': filename, 'class_id': class_id})

result = pd.DataFrame(predictions_model)
result.to_csv("data/Plant_Seedlings_Classification_Submission.csv", index=False)
