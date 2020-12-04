from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Activation
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import regularizers
from keras.optimizers import Adam

def fine_tune(base_model, n_class, multi_gpu_flag=False, gpus=1, method=0):

    if method == 0:
        x = base_model.output
        x = Flatten()(x)

        x = Dense(1024, kernel_regularizer=regularizers.l2(0.05))(x)
        x = Activation('relu')(x)
        x = Dropout(0.28)(x)

        x = Dense(1024, kernel_regularizer=regularizers.l2(0.05))(x)
        x = Activation('relu')(x)
        x = Dropout(0.28)(x)
        predictions = Dense(n_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model compiled!')
        if multi_gpu_flag:
            model = multi_gpu_model(model, gpus=gpus)
        return model
    else:
        return base_model
