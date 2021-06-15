#import tensorflow as tf
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

class DatasetGenerator():
    def __init__(self, path_args):
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
        self.x = []
        self.y = []
        self.load_dataset()

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
            self.x.append(input_data)
            self.y.append(result_data.astype(np.float))

test = DatasetGenerator(path_arg)


from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self,
                 x_data: list,
                 y_data: list,
                 batch_size: int,
                 shuffle: bool = False):

        self.x_list = x_data # <<<<
        self.y_list = y_data # <<<<
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.files[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        x_list_temp = [self.x_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(x_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


