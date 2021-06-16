import tensorflow as tf
import numpy as np
import os
import pandas as pd
import random
import tqdm
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.mixed_precision import experimental as mixed_precision

epoch = 200
batch_size = 32
timesteps = 100
step_per_batch = 200
time_interval = 16
deep = 3
train_able = True
# 클래스 숫자
n_class = 5

# True : train, test : False

def check_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


train_path = 'datasets/t_class5/'
val_path = 'datasets/v_class5/'
test_path = 'datasets/v_class5/'
result = './resutls_c5_t0100_m2/'
save_weight = result + 'save_weights/'
save_backup = result + 'back_weights/'
test_result = result + 'test/'
check_file(result)
check_file(save_weight)
check_file(save_backup)
check_file(test_result)
train_data = os.listdir(train_path)
# val_path = os.listdir(val_path)
test_data = os.listdir(test_path)
data_list = []
loss_list = []
accuracy_list = []
def train_data_processing(data):
    t, f = data.shape
    start = random.randint(0, t - timesteps - 1)
    input_data = np.reshape(data[start: start + timesteps, 0], (1, timesteps, 1))
    targets = np.reshape(data[start: start + timesteps, 1], (-1)).astype(np.int) # target (100, ) -> 1 1 1 1 1 0 1 1 1
    result_data = np.eye(n_class)[targets] # (100, 5) : [0,1,0,0,0] [1,0,0,0,0]
    result_data = np.reshape(result_data, [1, timesteps, n_class]) # (100, 5)
    for i in range(batch_size - 1):
        start = random.randint(0, t - timesteps)
        data_ = np.reshape(data[start: start + timesteps, 0], (1, timesteps, 1))
        input_data = np.append(input_data, data_, axis=0)

        targets = np.reshape(data[start: start + timesteps, 1], (-1)).astype(np.int)
        result_data_ = np.reshape(np.eye(n_class)[targets], [1, timesteps, n_class])
        result_data = np.append(result_data, result_data_, axis=0)
    return input_data, result_data.astype(np.float)


def test_data_processing(data):
    t, f = data.shape
    input_data = np.reshape(data[0:timesteps, 0], (1, timesteps, 1))
    targets = np.reshape(data[0:timesteps, 1], (-1)).astype(np.int)
    result_data = np.eye(n_class)[targets]
    result_data = np.reshape(result_data, [1, timesteps, n_class])
    for i in range(8, t - timesteps, time_interval):
        input_data = np.append(input_data, np.reshape(data[i: i + timesteps, 0], (1, timesteps, 1)), axis=0)

        targets = np.reshape(data[i: i + timesteps, 1], (-1)).astype(np.int)

        result_data_ = np.reshape(np.eye(n_class)[targets], [1, timesteps, n_class])
        result_data = np.append(result_data, result_data_, axis=0)
    return input_data, result_data.astype(np.float)


def activation(x):
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return np.array(x).astype(np.int64)

def act2(x):
    re_list = []
    for c0, c1, c2, c3, c4 in x:
        y=np.array([c0, c1, c2, c3, c4])
        if np.argmax(y) == 0:
            re_list.append(0)
        elif np.argmax(y) == 1:
            re_list.append(1)
        elif np.argmax(y) == 2:
            re_list.append(2)
        elif np.argmax(y) == 3:
            re_list.append(3)
        elif np.argmax(y) == 4:
            re_list.append(4)
    return np.array(re_list).astype(np.int64)

def act3(x):
    max_x = np.max(x)
    x[x < 1] = 0
    x[x >= 1] = max_x
    return np.array(x).astype(np.int64)

def checking(data):

    if collections.Counter(data)[0] > 44:
        i = 0
    else:
        i = np.max(data)
    return str(i).encode()

def accuracy(x, y):
    count = 0
    x = np.argmax(x, axis = 2).reshape(-1, 1)
    y = np.argmax(y, axis = 2).reshape(-1, 1)
    for i, j in enumerate(x):
        if j == y[i]:
            count += 1
    return count


def get_model():
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

# def get_model():
#     x_input = tf.keras.layers.Input(shape=(timesteps, 1))
#     x = x_input
#     for i in range(deep - 1):
#         x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
#     x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
#     x_return = tf.keras.layers.Dense(n_class, activation='softmax')(x)
#
#     return tf.keras.models.Model(x_input, x_return)

def train():
    if train_able:
        for i, j in enumerate(train_data):
            data_list.append(pd.read_csv(train_path + j).to_numpy())
        policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
        mixed_precision.set_policy(policy)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        model = get_model()
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        for i in range(epoch):
            pbar = tqdm.tqdm(range(step_per_batch))
            total_accuracy = 0
            total_loss = 0
            for j in pbar:
                x, y = train_data_processing(data_list[random.randint(0, len(train_data) - 1)])
                total_loss += model.train_on_batch(x, y)
                data = model.predict(x)
                total_accuracy += accuracy(data, y) / (timesteps * batch_size)
                pbar.set_description("epoch %d : %d, batch %d : %d, Loss : %0.4f, accuracy : %0.4f%%" % (
                epoch, i + 1, step_per_batch, j + 1, total_loss / (j + 1), total_accuracy / (j + 1) * 100))
                loss_list.append(total_loss / (j + 1))
                accuracy_list.append(total_accuracy / (j + 1) * 100)
            if (i + 1) % 10 == 0:
                model.save_weights(save_weight + 'epoch' + str(i + 1) + '.h5')
            plt.plot(loss_list)
            plt.ylabel('loss')
            plt.xlabel('batch')
            plt.savefig(result + 'loss.png')
            plt.clf()
            plt.plot(accuracy_list)
            plt.ylabel('accuracy')
            # plt.ylim([95, 100])
            plt.xlabel('batch')
            plt.savefig(result + 'accuracy.png')
            plt.clf()

        model.save_weights(save_backup)
        return 0
    else:
        t_met = 0
        for i, j in enumerate(test_data):
            data_list.append(pd.read_csv(test_path + j).to_numpy())
        model = get_model()
        model.summary()
        model.load_weights(save_weight + 'epoch200.h5')
        for i, j in enumerate(data_list):
            b, _ = j.shape
            total_result = np.zeros(b)
            f = open(test_result + test_data[i][:-4] + ".txt", 'w')
            g = open(test_result + test_data[i][:-4] + "total.txt", 'w')
            x, y = test_data_processing(j)
            data = model.predict(x)
            met = accuracy(data, y) / (timesteps * len(x)) * 100
            met = format(met, '.4f')
            print(test_data[i] + ' : ' + str(met) + '%')
            t_met += float(met)

            # plt.figure(figsize=(6.4, 9.6))
            for k, l in enumerate(data):
                # plt.clf()
                predict = act3(act2(l))
                # pre = checking(predict)
                # print(pre)
                # total_result[k * time_interval:k * time_interval + timesteps] += predict
                f.write('predict :\t\t' + str(predict) + '\n')
                f.write('groundtruth :\t' + str(y[k]) + '\n')
                f.write('\n')
                # plt.subplot(311, xlim=(0, timesteps), ylim=(-1500, 1500), title='Raw Signal', xlabel='Timestep',
                #             ylabel='Amplitude (uV)').plot(data_list[0][:, 0][k * time_interval:(k * time_interval) + timesteps], lw=1, color='red')
                # plt.subplot(312, xlim=(0, timesteps), ylim=(-100, 100), title='Raw Signal', xlabel='Timestep',
                #             ylabel='Amplitude (uV)').plot(data_list[0][:, 0][k * time_interval:(k * time_interval) + timesteps], lw=1, color='red')
                # plt.subplot(313, xlim=(0, timesteps), ylim=(-1, 7), title='Predict Signal (0, 1)', xlabel='Timestep', ylabel='Predict Value').plot(predict[:len(predict)], lw=1, color='blue')
                # plt.tight_layout()
                # plt.draw()
                # plt.pause(0.1)
                # plt.cla()
            # g.writelines('predict :\t\t' + str(total_result / int(64 / time_interval)) + '\n')
            # g.writelines('groundtruth :\t' + str(j[:, -1] / 1) + '\n')
            f.close()
            # g.close()
        print('-----------------------------------------------')
        avg_met = format(t_met/len(data_list), '.4f')
        print('Total Average Accuracy : ' + str(avg_met) + '%')

with tf.device('/device:GPU:0'):
    train()