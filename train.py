import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from utils.data_generator import DataGenerator
from model import model

path_arg = {
    "train": './t_class5/',
    "validation": './v_class5/',
    "test": './v_class5/',
    "result": './resutls_c5_t0100_m2/'
}

EPOCH = 200
batch_size = 32
time_interval = 16
deep = 3

train_generator = DataGenerator(path_arg, batch_size=batch_size, shuffle=True)
x_len, y_len = train_generator.get_data_len()
train_steps_per_epoch = x_len // batch_size


policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
model = model.get_model()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=EPOCH)

