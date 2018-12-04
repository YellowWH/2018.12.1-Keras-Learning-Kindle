import keras
from keras import layers
from keras.models import Model

callback_list = [keras.callbacks.EarlyStopping(monitor='acc', #监视模型的精度
                                               patience=1, #如果精度在多与1轮的时间(即两轮)里不再改善,则终止
                                               ),
                 keras.callbacks.ModelCheckpoint(filepath='my_model.h5', #模型权重保存路径
                                                 monitor='val_loss', #如果val_loss没有改善就不用保存,文件里就一直保持最优模型权重
                                                 save_best_only=Ture,
                                                 ),
                 keras.callbacks.ReduceLROnPlateau(
                                                   monitor='val_loss', #监控模型的验证损失
                                                   factor=0.1,  #触发时将学习率除以10
                                                   patience=10, #如果验证损失在10轮内都没有改善,那么就触发这个回调函数
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
