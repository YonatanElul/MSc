from scipy.signal import correlate2d

import numpy as np


def auto_corr2d(in_arr: np.ndarray, kernel: (int, int) = (3, 3), stride: (int, int) = (2, 2), mode: str = 'valid'):
    M = in_arr.shape[0]
    N = in_arr.shape[1]

    for i in range(0, M - kernel[0], stride[0]):
        for j in range(0, N - kernel[1], stride[1]):
            local_arr = in_arr[i:(i + kernel[0]), j:(j + kernel[1])]

            if i == 0 and j == 0:
                ac = np.expand_dims(correlate2d(local_arr, in_arr, mode=mode), axis=2)

            else:
                ac = np.concatenate((ac, np.expand_dims(correlate2d(local_arr, in_arr, mode=mode), axis=2)), axis=2)

    return ac

