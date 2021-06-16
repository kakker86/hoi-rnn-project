import sys
import tensorflow as tf

def ce_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
    # softmax_ce_loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    # tf.print(tf.math.reduce_mean(ce_loss) ,'\n', output_stream=sys.stdout, summarize=-1)
    return tf.math.reduce_mean(ce_loss)