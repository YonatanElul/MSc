import numpy as np


class DataRecord(object):
    """
    TODO: Add Documentation
    """

    def __init__(self, name: str, frequency: int, ecg_signal: np.ndarray, time: np.ndarray, qrs_time: np.ndarray,
                 annotations: dict = None, comments: dict = None, additional_signals: dict = None):
        """

        :param name:
        :param frequency:
        :param ecg_signal:
        :param time:
        :param qrs_time:
        :param annotations:
        :param comments:
        :param additional_signals:
        """

        self.record_name = name
        self.Fs = frequency
        self.ecg = ecg_signal
        self.annotations = annotations
        self.comments = comments
        self.aux_signals = additional_signals
        self.time = time
        self.qrs_time = qrs_time

    def __len__(self):
        """

        :return:
        """

        return self.ecg.shape[0]

