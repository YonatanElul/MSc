import numpy as np
import wfdb
from ML.Utilities.DataRecord import DataRecord
import os


class DBHandler(object):
    """
    TODO: Add Documentation
    """

    def __init__(self, db_dir_path: str, db_name: str):
        """

        :param db_dir_path:
        :param db_name:
        """

        self.db_name = db_name
        self.db_dir = db_dir_path
        self.records_names = []
        self.db = []
        self._create_db()

    def _create_db(self):
        """

        :return:
        """

        db = []
        files = os.listdir(self.db_dir)

        for file in files:

            if file[-3:] != 'dat':
                continue

            file_name = file[:-4]
            if file_name not in self.records_names:
                self.records_names.extend([file_name])

        for rec in self.records_names:
            # Extract data
            record = wfdb.rdsamp((self.db_dir + '\\' + rec))

            signal = record[0]
            frequency = record[1]['fs']
            step = 1 / frequency

            time = np.linspace(start=0, stop=(step * int(signal.shape[0])), num=int(signal.shape[0]))

            comments = record[1]['comments']

            # Deal with each database appropriately
            if self.db_name == 'Fantasia':
                anno = wfdb.rdann((self.db_dir + '\\' + rec), 'ecg')

                qrs_annotations = anno.sample
                beat_annotations = anno.symbol
                AF_beats = []
                other_beats = []
                N_labels = np.ones_like(signal)

                ind = 0
                for beat in beat_annotations:
                    if beat_annotations == 'N':
                        ind += 1
                        continue

                    elif beat == '(AFIB' or beat == 'A' or beat == 'a':
                        AF_beats.extend([ind])
                        ind += 1

                    else:
                        other_beats.extend([ind])
                        ind += 1

                AF_beats = np.array(AF_beats)
                other_beats = np.array(other_beats)

                annotations = {'R_peaks': qrs_annotations, 'beat_type': beat_annotations,
                               'AF_beats': np.array(AF_beats), 'N_labels': N_labels,
                               'Other_beats': np.array(other_beats)}

                qrs_time = time[qrs_annotations]
                ecg_ind = record[1]['sig_name'].index['ECG']
                ecg = signal[:, ecg_ind]
                aux_signals = {}

                for sig in record[1]['sig_name']:
                    if sig == 'ECG':
                        continue

                    else:
                        sig_ind = record[1]['sig_name'].index(sig)
                        aux_signals[sig] = signal[:, sig_ind]

            elif self.db_name == 'BIH-Arrhythmia' or self.db_name == 'Noise-Stress':
                anno = wfdb.rdann((self.db_dir + '\\' + rec), 'atr')

                qrs_annotations = anno.sample
                beat_annotations = anno.symbol

                AF_beats = []
                other_beats = []

                ind = 0
                for beat in beat_annotations:
                    if beat_annotations == 'N':
                        ind += 1
                        continue

                    elif beat == '(AFIB' or beat == 'A' or beat == 'a':
                        AF_beats.extend([ind])
                        ind += 1

                    else:
                        other_beats.extend([ind])
                        ind += 1

                annotations = {'R_peaks': qrs_annotations, 'beat_type': beat_annotations,
                               'AF_beats': np.array(AF_beats),
                               'Other_beats': np.array(other_beats)}

                qrs_time = time[qrs_annotations]
                ecg = signal[:, 0]
                aux_signals = {}

                for sig in record[1]['sig_name']:
                    if sig == 'V5':
                        continue

                    else:
                        sig_ind = record[1]['sig_name'].index(sig)
                        aux_signals[sig] = signal[:, sig_ind]

            elif self.db_name == 'BIH-Fibrillation':
                qrs = wfdb.rdann((self.db_dir + '\\' + rec), 'qrs')
                anno = wfdb.rdann((self.db_dir + '\\' + rec), 'atr')

                qrs_annotations = qrs.sample
                beat_annotations = anno.aux_note
                AF_beats = []
                AF_labels = np.zeros_like(signal[:, 1]).astype(np.int8)

                af_labels = []
                n_labels = []
                i = 0
                for beat in beat_annotations:
                    if beat == '(AFIB':
                        AF_beats.append([anno.sample[i]])
                        af_labels.append([anno.sample[i]])
                        i += 1

                    else:
                        n_labels.append([anno.sample[i]])
                        i += 1
                        continue

                if beat_annotations[0] == '(AFIB':
                    corr = 0
                else:
                    corr = 1

                if beat_annotations[-1] != '(AFIB':
                    for i in range(af_labels.__len__()):
                        AF_labels[af_labels[i][0]:n_labels[i + corr][0]] = 1
                        i += 1
                else:
                    for i in range(af_labels.__len__()):
                        if i == af_labels.__len__() - 1:
                            AF_labels[af_labels[i][0]:] = 1
                            continue

                        else:
                            AF_labels[af_labels[i][0]:n_labels[i + corr][0]] = 1
                            i += 1

                annotations = {'R_peaks': qrs_annotations, 'beat_type': beat_annotations,
                               'AF_beats': np.array(AF_beats), 'AF_labels': AF_labels,
                               'N_beats': np.array(n_labels)}

                qrs_time = time[qrs_annotations]
                ecg_ind = record[1]['sig_name'].index('ECG1')
                ecg = signal[:, ecg_ind]
                aux_signals = {}

                for sig in record[1]['sig_name']:
                    if sig == 'ECG1':
                        continue

                    else:
                        sig_ind = record[1]['sig_name'].index(sig)
                        aux_signals[sig] = signal[:, sig_ind]

            elif self.db_name == 'Long-Term-AF':
                anno = wfdb.rdann((self.db_dir + '\\' + rec), 'atr')

                qrs_annotations = anno.sample
                beat_annotations = anno.symbol

                AF_beats = []
                other_beats = []

                ind = 0
                for beat in beat_annotations:
                    if beat_annotations == 'N':
                        ind += 1
                        continue

                    elif beat == '(AFIB' or beat == 'A' or beat == 'a':
                        AF_beats.extend([ind])
                        ind += 1

                    else:
                        other_beats.extend([ind])
                        ind += 1

                annotations = {'R_peaks': qrs_annotations, 'beat_type': beat_annotations,
                               'AF_beats': np.array(AF_beats),
                               'Other_beats': np.array(other_beats)}
                try:
                    qrs_time = time[qrs_annotations]
                    ecg = signal[:, 0]
                    aux_signals = {'ECG2': signal[:, 1]}

                except:
                    pass

            # Create the DataRecord object
            db.extend([DataRecord(name=rec, frequency=frequency, ecg_signal=ecg, time=time,
                                  comments=comments, annotations=annotations,
                                  additional_signals=aux_signals, qrs_time=qrs_time)])

        self.db = db

    def update_record(self, sig, idx):
        self.db[idx].ecg = sig

    def __getitem__(self, n: int):
        return self.db[n]

    def __setitem__(self, n: int, record: DataRecord):
        self.db[n] = record

    def __delitem__(self, n: int):
        self.db.remove(self.db[n])

    def __len__(self):
        return len(self.db)
