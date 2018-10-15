import numpy as np
import keras

"""

"""


class DataGenerator(keras.utils.Sequence):
    """

    """

    def __init__(self, input_indices: np.ndarray, input_shape: np.ndarray, samples_path: list, labels_path: list,
                 batch_size: int = 32, n_classes=2, shuffle=True, signals_in: int = 1, op_mode: str = 'point',
                 label: str = 'binary', signal_type: int = 0):

        """

        :param input_indices:
        :param input_shape:
        :param samples_path:
        :param labels_path:
        :param batch_size:
        :param n_classes:
        :param shuffle:
        :param signals_in:
        :param op_mode:
        :param label:
        :param signal_type:
        """

        # Save direct paths for the signals
        self.signals_in = signals_in

        if self.signals_in == 1:
            self.ecg_path = samples_path[0]

        elif self.signals_in == 2:
            self.ecg_path = samples_path[0]
            self.r_peaks_path = samples_path[1]

        elif self.signals_in == 3:
            self.ecg_path = samples_path[0]
            self.r_peaks_path = samples_path[1]
            self.entropy_path = samples_path[2]

        # Save direct paths for the lables
        self.dont_care_path = labels_path[0]
        self.nsr_beats_path = labels_path[1]
        self.af_predict_path = labels_path[2]
        self.af_beats_path = labels_path[3]

        self.signal_type = signal_type
        self.label = label
        self.samples_path = samples_path
        self.labels_path = labels_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.input_indices = input_indices
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = 0
        self.op_mode = op_mode
        self.on_epoch_end()

    def __len__(self):
        """

        :return:
        """

        return int(np.floor(len(self.input_indices) / self.batch_size))

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        # Generate indexes of the batch
        indexes = self.indexes[(index * self.batch_size):((index + 1) * self.batch_size)]

        # Find list of indices
        list_indices_temp = [self.input_indices[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_indices_temp)

        return x, y

    def on_epoch_end(self):
        """

        :return:
        """

        self.indexes = np.arange(len(self.input_indices))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indices_temp):
        """

        :param list_indices_temp:
        :return:
        """

        # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.input_shape))

        if self.op_mode == 'segment':
            if self.label == 'binary':
                y = np.empty((self.batch_size, 1, 1), dtype=int)
            else:
                y = np.empty((self.batch_size, 1, 4), dtype=int)

        else:
            if self.label == 'binary':
                y = np.empty((self.batch_size, *self.input_shape), dtype=int)

            else:
                y = np.empty((self.batch_size, self.input_shape[0], 4), dtype=int)

        # Generate data
        max_ind = np.max(list_indices_temp) + self.input_shape[0]
        min_ind = np.min(list_indices_temp)

        ecg = np.load(self.ecg_path)[min_ind:max_ind]

        if self.signals_in == 2:
            r_peaks = np.load(self.r_peaks_path)[min_ind:max_ind]

        elif self.signals_in == 3:
            r_peaks = np.load(self.r_peaks_path)[min_ind:max_ind]
            entropy = np.load(self.entropy_path)[min_ind:max_ind]

        for i, ind in enumerate(list_indices_temp):
            # Store sample
            # if self.signals_in == 1:
            #     x[i, :] = np.load(self.ecg_path)[ind:(ind + self.input_shape[0])]
            #
            # elif self.signals_in == 2:
            #     x[i, :] = np.array([ecg[ind:(ind + self.input_shape[0])], r_peaks[ind:(ind + self.input_shape[0])]]).T
            #
            # elif self.signals_in == 3:
            #     x[i, :] = np.array([ecg[ind:(ind + self.input_shape[0])], r_peaks[ind:(ind + self.input_shape[0])],
            #                         entropy[ind:(ind + self.input_shape[0])]]).T

            # Normalize index
            ind -= min_ind
            if self.signal_type == 0:
                x[i, :] = ecg[ind:(ind + self.input_shape[0])]

            elif self.signal_type == 1:
                x[i, :] = r_peaks[ind:(ind + self.input_shape[0])]

            else:
                x[i, :] = entropy[ind:(ind + self.input_shape[0])]

        del ecg

        if self.label == 'catagorical':
            dont_care = np.load(self.dont_care_path)[min_ind:max_ind]
            norm_sin = np.load(self.nsr_beats_path)[min_ind:max_ind]
            af_predict = np.load(self.af_predict_path)[min_ind:max_ind]
            af_beat = np.load(self.af_beats_path)[min_ind:max_ind]

        else:
            af_predict = np.load(self.af_predict_path)[min_ind:max_ind]

        for i, ind in enumerate(list_indices_temp):
            # Load labels
            # Normalize index
            ind -= min_ind
            if self.label == 'catagorical':
                dc = dont_care[ind:(ind + self.input_shape[0])]
                nsr = norm_sin[ind:(ind + self.input_shape[0])]
                afp = af_predict[ind:(ind + self.input_shape[0])]
                afb = af_beat[ind:(ind + self.input_shape[0])]

                if self.op_mode == 'segment':
                    tmp_y = np.array([dc, nsr, afp, afb]).T

                    labels_sum = np.sum(tmp_y, axis=0)
                    label = np.argmax(labels_sum)

                    if label.size > 1:
                        label = label[0]

                    y_place_holder = np.zeros([1, 4])
                    y_place_holder[0, label] = 1
                    y[i, :] = y_place_holder

                else:
                    y[i, :] = np.array([dc, nsr, afp, afb]).T

            else:
                if self.op_mode == 'segment':
                    tmp_y = af_predict[ind:(ind + self.input_shape[0])]

                    labels_sum = np.mean(tmp_y, axis=0)

                    if labels_sum > 0.5:
                        label = 1

                    else:
                        label = 0

                    y[i, 0, 0] = label

                else:
                    y[i, :] = af_predict[ind:(ind + self.input_shape[0]), :]

        return x, y
