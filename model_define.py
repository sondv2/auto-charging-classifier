import efficientnet.keras as efn
from keras import applications
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Dropout, \
    GlobalAveragePooling2D, MaxPool2D, Flatten, concatenate, Input
from keras.models import Sequential, Model
from keras.optimizers import nadam, Adam

def freeze_layers(model, pos=10):
    # for layer in model.layers[:]:
    #     layer.trainable = False
    model.trainable = False
def customize_mode():
    # Building the model
    image_model = Sequential()

    # CNN 1
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    image_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    image_model.add(Dropout(0.2))

    # CNN 2
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(0.2))

    # CNN 3
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(0.2))

    # CNN 4
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    image_model.add(Flatten())

    # Image input encoding
    image_input = Input(shape=(75, 75, 3))
    encoded_image = image_model(image_input)

    # Inc angle input
    inc_angle_input = Input(shape=(1,))

    # Combine image and inc angle
    combined = concatenate([encoded_image, inc_angle_input])

    dense_model = Sequential()

    # Dense 1
    dense_model.add(Dense(512, activation='relu', input_shape=(257,)))
    dense_model.add(Dropout(0.2))

    # Dense 2
    dense_model.add(Dense(256, activation='relu'))
    dense_model.add(Dropout(0.2))

    # Output
    dense_model.add(Dense(1, activation="sigmoid"))

    output = dense_model(combined)

    # Final model
    combined_model = Model(inputs=[image_input, inc_angle_input], outputs=output)

    optimizer = Adam(lr=0.001, decay=0.0)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return combined_model

def model_define(modeltype, inputshape):

    if modeltype == 'define':
        model = customize_mode()
        print('Model: define !')
    elif modeltype == 'EfficientNetB3':
        model = efn.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        print('Model: EfficientNetB3, weights loaded!')
    elif modeltype == 'ResNet50':
        model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=inputshape, pooling='avg')
        freeze_layers(model)
        print('Model: ResNet50, weights loaded!')
    elif modeltype == 'Xception':
        model = applications.Xception(include_top=False, weights='imagenet', input_shape=inputshape)
        freeze_layers(model)
        print('Model: Xception, weights loaded!')
    else:
        pass

    return model
