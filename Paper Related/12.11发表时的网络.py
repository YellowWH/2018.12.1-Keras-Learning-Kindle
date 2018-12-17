import keras

from keras import layers
from keras import Input
from keras import Sequential, Model
from keras.models import Model
from keras import applications

searchblock_image_shape = Input(shape=(15, 15, 3))
branch_one_pix = layers.Conv2D(128, 3, activation='relu', padding='same')(searchblock_image_shape)
branch_one_pix = layers.Conv2D(128, 3, activation='relu', padding='same')(branch_one_pix)
branch_one_pix = layers.Flatten()(branch_one_pix)

right_image_shape = Input(shape=(3000, 2000, 3))
branch_whole_image = layers.Conv2D(128, 3, activation='relu', padding='same')(right_image_shape)
branch_whole_image = layers.Conv2D(128, 3, activation='relu', padding='same')(branch_whole_image)
branch_whole_image = layers.Conv2D(128, 3, activation='relu', padding='same')(branch_whole_image)
branch_whole_image = layers.Flatten()(branch_whole_image)

concatenate1 = layers.concatenate([branch_one_pix, branch_whole_image], axis=-1)
disparityL = layers.Dense(4000, activation='relu')(concatenate1)
disparityL = layers.Dense(3000, activation='relu')(disparityL)
disparityL = layers.Dense(2000, activation='relu')(disparityL)
disparityL = layers.Dense(2000, activation='softmax')(disparityL)

model = Model([searchblock_image_shape, right_image_shape], disparityL)
model.summary()

image_shape_L = Input(shape=(3000, 2000, 3))
image_shape_R = Input(shape=(3000, 2000, 3))

branch_L = layers.Conv2D(128, 5, activation='relu', padding='same')(image_shape_L)
branch_L = layers.MaxPooling2D(4, strides=4, padding='same')(branch_L)
branch_L = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_L)
branch_L = layers.MaxPooling2D(4, strides=4, padding='same')(branch_L)
branch_L = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_L)
branch_L = layers.MaxPooling2D(4, strides=4, padding='same')(branch_L)
branch_L = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_L)
branch_L = layers.MaxPooling2D(4, strides=4, padding='same')(branch_L)

branch_R = layers.Conv2D(128, 5, activation='relu', padding='same')(image_shape_R)
branch_R = layers.MaxPooling2D(4, strides=4, padding='same')(branch_R)
branch_R = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_R)
branch_R = layers.MaxPooling2D(4, strides=4, padding='same')(branch_R)
branch_R = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_R)
branch_R = layers.MaxPooling2D(4, strides=4, padding='same')(branch_R)
branch_R = layers.Conv2D(128, 5, activation='relu', padding='same')(branch_R)
branch_R = layers.MaxPooling2D(4, strides=4, padding='same')(branch_R)

branch_L = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_L)
branch_L = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_L)
branch_L = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_L)

branch_R = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_R)
branch_R = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_R)
branch_R = layers.Conv2D(64, 3, activation='relu', padding='same')(branch_R)

concatenate = layers.concatenate([branch_L, branch_R], axis=-1)
disparityL2 = layers.Flatten()(concatenate)
disparityL2 = layers.Dense(1500, activation='relu')(disparityL2)
disparityL2 = layers.Dense(1500, activation='softmax')(disparityL2)

model2 = Model([image_shape_L, image_shape_R], disparityL2)

model2.summary()
