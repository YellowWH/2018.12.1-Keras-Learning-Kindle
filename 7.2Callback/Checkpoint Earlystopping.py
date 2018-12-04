import keras
from keras import layers
from keras.models import Model

callback_list = [keras.callbacks.EarlyStopping(monitor='acc',
                                               patience=1,
                                               ),
                 keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                 monitor='val_loss',
                                                 save_best_only=Ture,
                                                 )]

x = layers.Conv2D(128, 3, activation='relu')

model = Model()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callback_list,
          validation_data=(x_val, y_val))