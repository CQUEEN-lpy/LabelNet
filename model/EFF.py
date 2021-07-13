import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

"""
return the a CLS net using the EfficientNetB4 as the backbone network 
Parameters:
1. img_height,img_width: the input images shape
2. class_nums: the number of classes of the CLS task
3. binary: whether is a binary CLS task
"""

def get_EFFmodel(img_height, img_width, class_nums, binary=False):
    # load the EFF_model and use the pretraining weights(imagenet)
    EFF_model = tf.keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(img_height, img_width, 3), pooling=None,
        classifier_activation=None,
    )

    #data_augmentation
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.3),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )

    # classifier
    if not binary:
        classifier = keras.Sequential([
            layers.Dropout(0.2),
            layers.Dense(class_nums, activation='softmax')]
        )
    else:
        classifier = keras.Sequential([
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')]
        )

    # normalization
    norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

    # create my own model and compile
    inputs = keras.Input(shape=(img_width, img_height, 3))

    x = data_augmentation(inputs)
    x = norm_layer(x)
    x = EFF_model(x, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = classifier(x)
    model = keras.Model(inputs, outputs)
    model.trainable = True
    return model
