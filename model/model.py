import tensorflow as tf

def get_model(timesteps=100, n_class=5):
    x_input = tf.keras.layers.Input(shape=(timesteps, 1))
    x = x_input

    y = tf.keras.layers.Dense(1024)(x)
    # for i in range(deep - 1):
    #     x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = x + y
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    x_return = tf.keras.layers.Dense(n_class, activation='softmax', dtype='float32')(x)

    return tf.keras.models.Model(x_input, x_return)
