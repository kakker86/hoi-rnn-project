import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from config import GetConfig
from utils.data_generator import DataGenerator
from model import model


config = GetConfig()

BATCH_SIZE = 32
EPOCH = 200
LEARNING_RATE = 0.0001

train_generator = DataGenerator(config.get_dir_path(), batch_size=BATCH_SIZE, shuffle=True)
x_len, y_len = train_generator.get_data_len()
train_steps_per_epoch = x_len // BATCH_SIZE


policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
model = model.get_model()
model.summary()

# metric method
category_acc = tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None)
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

# callback method
polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=LEARNING_RATE,
                                                          decay_steps=EPOCH,
                                                          end_learning_rate=LEARNING_RATE * 0.1, power=0.5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.get_dir_path()['tensorboard'], write_graph=True, write_images=True)

callback = [tensorboard, lr_scheduler]



model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=[category_acc, precision, recall])


model.fit(train_generator,
          steps_per_epoch=train_steps_per_epoch,
          epochs=EPOCH,
          callbacks=callback)

