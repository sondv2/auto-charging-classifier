from keras import optimizers
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model

def fine_tune(base_model, n_class, multi_gpu_flag=False, gpus=1, method=0):

    if method == 0:
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(units=n_class, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        print('efficientnet fine tune, success!')
        model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
        if multi_gpu_flag:
            model = multi_gpu_model(model, gpus=gpus)
        return model
    else:
        return base_model
