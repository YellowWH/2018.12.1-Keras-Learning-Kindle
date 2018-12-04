import keras
from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
preditions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], preditions)
model.fit([left_data, right_data], targets)