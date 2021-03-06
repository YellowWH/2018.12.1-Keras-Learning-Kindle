##预测发帖人的年龄性别收入水平
import keras
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None, ), dtype='int32', name='post')
embedded_post = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_post)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_predition = layers.Dense(1, name='age')(x)
income_predition = layers.Dense(num_income_groups,
                                activation='softmax',
                                name='income')(x)
gender_predition = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_predition, income_predition, gender_predition])

model.summary()

#用不同函数逼近和设置不同权重
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

#当层有名字的时候可以用以下代码
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse',
#                     'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'},
#               loss_weights={'age': 0.25,
#                             'income': 1.,
#                             'gender': 10.}) 

#将数据输入到刚建立的 多输出模型 中
model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10, batch_size=64)

# model.fit(posts, {'age': age_targets,
#                   'income': income_targets,
#                   'gender': gender_targets},
#           epochs=10, batch_size=64)

