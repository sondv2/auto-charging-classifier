from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from clr_callback import CyclicLR

def callbacks_save_model_weight():

    BEST_WEIGHTS = 'model_save/weight_fold_%s.hdf5' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    save_checkpoint = ModelCheckpoint(BEST_WEIGHTS, monitor='val_acc', verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True, mode='max')
    callbacks = [early_stop, save_checkpoint]
    return callbacks

def callbacks_save_model():

    BEST_MODELS = 'model_save/model_%s.hdf5' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    save_checkpoint = ModelCheckpoint(BEST_MODELS, monitor='val_acc', verbose=1,
                                      save_best_only=True, mode='max')
    callbacks = [early_stop, save_checkpoint]
    return callbacks

def callbacks_clr(train_size, batch_size):

    train_steps = np.ceil(float(train_size) / float(batch_size))
    BEST_WEIGHTS = 'model_save/weight_fold_%s.hdf5' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    TRAINING_LOG = 'logs/trainlog_fold_%s.csv' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    clr = CyclicLR(base_lr=1e-7, max_lr=2e-4, step_size=4 * train_steps, mode='exp_range', gamma=0.99994)
    early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    save_checkpoint = ModelCheckpoint(BEST_WEIGHTS, monitor='val_acc', verbose=1, save_weights_only=True,
                                      save_best_only=True, mode='max')
    csv_logger = CSVLogger(TRAINING_LOG, append=False)
    callbacks = [early_stop, save_checkpoint, csv_logger, clr]
    return callbacks