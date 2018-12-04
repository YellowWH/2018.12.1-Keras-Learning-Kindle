import keras
from keras import layers

#残差连接可以稍微解决梯度消失和标识瓶颈问题
#假设有一个4维的张量x
x=...

#以下是恒等残差连接 Identity Residual Connection
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x])

#以下是线性残差连接 Linear Residual Connection
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.MaxPooling2D(2, strides=2)(y)

residual =layers.Conv2D(128, 1, strides=2, padding='same')(x)

y = layers.add([y, residual])


