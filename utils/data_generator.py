import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
import pandas as pd
import random
import numpy as np

train_path = './t_class5/'
val_path = './v_class5/'
test_path = './v_class5/'
result = './resutls_c5_t0100_m2/'
save_weight = result + 'save_weights/'
save_backup = result + 'back_weights/'
test_result = result + 'test/'

# os.makedir

# os.listdir

path_arg = {
    "train": '../t_class5/',
    "validation": './v_class5/',
    "test": './v_class5/',
    "result": '../resutls_c5_t0100_m2/'
}


class DataGenerator(Sequence):
    def __init__(self,
                 path_args,
                 batch_size: int,
                 shuffle: bool = True):

        self.path_args = path_args
        self.save_weight = self.path_args['result'] + 'save_weights/'
        self.save_backup = self.path_args['result'] + 'back_weights/'
        self.test_result = self.path_args['result'] + 'test/'

        self.train_data = os.listdir(self.path_args['train'])

        os.makedirs(self.save_weight, exist_ok=True)
        os.makedirs(self.save_backup, exist_ok=True)
        os.makedirs(self.test_result, exist_ok=True)


        self.timeSteps = 100
        self.num_classes = 5
        self.x_list = []
        self.y_list = []
        self.load_dataset()

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def load_dataset(self):
        for i, j in enumerate(self.train_data):
            data = pd.read_csv(self.path_args['train'] + j).to_numpy()
            # data.shape = (549, 2)
            t, f = data.shape  # t: 549, f: 2

            start = random.randint(0, t - self.timeSteps - 1)
            input_data = np.reshape(data[start: start + self.timeSteps, 0], (1, self.timeSteps, 1))
            targets = np.reshape(data[start: start + self.timeSteps, 1], (-1)).astype(np.int)
            result_data = np.eye(self.num_classes)[targets]
            result_data = np.reshape(result_data, [1, self.timeSteps, self.num_classes])

            # self.data_list.append([input_data, result_data.astype(np.float)])
            self.x_list.append(input_data)
            self.y_list.append(result_data.astype(np.float))

    def get_data_len(self):
        return len(self.x_list), len(self.y_list)


    def __len__(self):
        return int(np.floor(len(self.x_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x = [self.x_list[k] for k in indexes]
        y = [self.y_list[k] for k in indexes]

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

