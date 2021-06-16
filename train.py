import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from model.loss import ce_loss
from config import GetConfig
from utils.data_generator import DataGenerator
from model import model


config = GetConfig()
params = config.get_hyperParams()

tf.keras.backend.clear_session()


train_generator = DataGenerator(config.get_dir_path(), batch_size=params['batch_size'], shuffle=True, mode='train')
valid_generator = DataGenerator(config.get_dir_path(), batch_size=params['batch_size'], shuffle=True, mode='validation')
train_len, _ = train_generator.get_data_len()
valid_len, _ = valid_generator.get_data_len()
train_steps_per_epoch = train_len // params['batch_size']
valid_steps_per_epoch = valid_len // params['batch_size']


policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
model = model.get_model()
model.summary()

# metric method
category_acc = tf.keras.metrics.CategoricalAccuracy()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

# callback method
polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=params['lr'],
                                                          decay_steps=params['epoch'],
                                                          end_learning_rate=params['lr'] * 0.1, power=0.5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.get_dir_path()['tensorboard'], write_graph=True, write_images=True)

callback = [tensorboard, lr_scheduler]



model.compile(loss=ce_loss,
              optimizer=optimizer,
              metrics=[category_acc, precision, recall])


with tf.device('/device:GPU:0'):
    model.fit(train_generator,
              validation_data=valid_generator,
              validation_steps=valid_steps_per_epoch,
              validation_batch_size=params['batch_size'],
              steps_per_epoch=train_steps_per_epoch,
              epochs=params['epoch'],
              callbacks=callback,
              batch_size=params['batch_size']
              )

