from DeterministicAnalytics.EntropyMethods.Utilities.CalcEnt import EntropyManipulator

import pickle
import os
import numpy as np


def db2npy(db_path: str, af_predict_time: int = 3, dont_care_beats: int = 1, segment_length: int = 120,
           db_type: str = 'train', healthy: bool = False):

    """

    :param db_path:
    :param af_predict_time:
    :param dont_care_beats:
    :param segment_length:
    :param db_type:
    :param healthy:
    :return:
    """

    db = pickle.load(open(db_path, 'rb')).db
    entropy_calculator = EntropyManipulator()

    i = 0
    k = 0
    j = 0
    for patient in db:

        print('Managing patient #' + str(k + 1))

        if patient.record_name[0] == 'f':
            # Record from the Fantasia database
            healthy = True

        else:
            # Record from the Atrial Fibrillation Database
            healthy = False

        predict_interval = af_predict_time * patient.Fs * 60

        # Get signals
        signal = patient.ecg
        nan_inds = np.isnan(signal)

        # Announce in case there are 'nan' elements in one of the signals
        nans = np.where(nan_inds is True)
        try:
            if nans[0][0]:
                print('************************************************************')
                print('Record: ' + patient.record_name + ' contain "NaN" elements')
                print('The "NaN" elements are: ' + str(nans[0]))
                print('************************************************************')

        except:
            continue

        signal[nan_inds] = 0
        signal = (signal - np.mean(signal)) / (np.std(signal))
        signal = np.reshape(signal, [signal.size, 1])
        r_peaks = np.reshape(patient.annotations['R_peaks'], [patient.annotations['R_peaks'].size, 1])
        entropy = entropy_calculator.calc_avg_shannon_ent_rr(signal)
        entropy = np.reshape(entropy, [entropy.size, 1])
        # p_waves = eng.detect_interest_points_in_ECG(signal)

        # Get labels
        # Initiate labels' vectors
        af_beats = np.zeros(signal.size,)
        af_preds = np.zeros(signal.size,)
        n_beats = np.zeros(signal.size,)
        dont_care = np.zeros(signal.size,)

        if healthy:
            n_beats = np.ones(signal.size,)

            label = np.array([dont_care, n_beats, af_preds, af_beats]).T

            if j == 0 and i == 0:
                ECG = signal
                labels = label

            else:
                ECG = np.vstack([ECG, signal])
                labels = np.vstack([labels, label])

            j += 1
            k += 1
            continue

        else:

            ending_inds = np.zeros(4, )
            for m, af_beat in enumerate(patient.annotations['AF_beats']):
                # End Cases Checks
                if not patient.annotations['N_beats'].size:
                    # Meaning that the entire signal is an AF recording
                    af_starting_r_peak = np.where(patient.annotations['R_peaks'] >=
                                                  patient.annotations['AF_beats'][m][0])[0][0]

                    corrected_af_starting_r_peak = af_starting_r_peak + dont_care_beats
                    af_starting_ind = patient.annotations['R_peaks'][corrected_af_starting_r_peak]

                    # Find & label the 'Don't Care' interval - Prior to the AF event
                    dont_care_starting_r_peak = af_starting_r_peak - dont_care_beats
                    dont_care_ind = patient.annotations['R_peaks'][dont_care_starting_r_peak]

                    if dont_care_ind > 0:
                        dont_care[dont_care_ind:af_starting_ind] = 1
                        af_preds_ind = dont_care_ind - predict_interval

                        if af_preds_ind < 0:
                            af_preds_ind = 0

                        else:
                            dont_care[0:af_preds_ind] = 1

                        af_preds[af_preds_ind:dont_care_ind] = 1

                    else:
                        dont_care[0:af_starting_ind] = 1

                    af_beats[af_starting_ind:] = 1

                    label = np.array([dont_care, n_beats, af_preds, af_beats]).T

                    if i == 0:
                        ECG = signal
                        # R_Peaks = r_peaks
                        # Optimizer = entropy
                        labels = label
                        # r_peaks_inds = patient.annotations['R_peaks']

                    else:
                        ECG = np.vstack([ECG, signal])
                        # R_Peaks = np.vstack([R_Peaks, r_peaks])
                        # Optimizer = np.vstack([Optimizer, entropy])
                        labels = np.vstack([labels, label])
                        # r_peaks_inds = np.vstack([r_peaks_inds, patient.annotations['R_peaks']])

                    # Move to next patient
                    break

                # Get index of starting AF - R-Peak
                af_starting_r_peak = np.where(patient.annotations['R_peaks'] >= patient.annotations['AF_beats'][m][0])[0][0]
                corrected_af_starting_r_peak = af_starting_r_peak + dont_care_beats
                af_starting_ind = patient.annotations['R_peaks'][corrected_af_starting_r_peak]

                # Find & label the 'Don't Care' interval - Prior to the AF event
                dont_care_starting_r_peak = af_starting_r_peak - dont_care_beats

                # Find the prediction interval
                af_predict_starting_ind = patient.annotations['R_peaks'][dont_care_starting_r_peak] - predict_interval

                if af_predict_starting_ind < ending_inds[0] or af_predict_starting_ind < ending_inds[1] \
                        or af_predict_starting_ind < ending_inds[2] or af_predict_starting_ind < ending_inds[3]:
                    af_predict_starting_ind = int(np.max(ending_inds))

                dont_care_starting_ind = patient.annotations['R_peaks'][dont_care_starting_r_peak]
                dont_care_ending_ind = patient.annotations['R_peaks'][corrected_af_starting_r_peak]

                if dont_care_starting_ind > af_predict_starting_ind:
                    dont_care[dont_care_starting_ind:dont_care_ending_ind] = 1

                # Label the AF prediction interval
                if m > 0:
                    if af_predict_starting_ind > af_starting_ind:
                        af_predict_starting_ind = af_starting_ind

                # Check if there was NSR previously in case this is the first AF event
                if m == 0:
                    # 'Look Backwards'
                    if patient.annotations['AF_beats'][0][0] < patient.annotations['N_beats'][0][0]:
                        # There is no previous NSR
                        r = m
                        if af_predict_starting_ind < 0:
                            af_predict_starting_ind = 0
                            af_predict_ending_ind = af_predict_starting_ind + predict_interval

                            if af_predict_ending_ind > dont_care_starting_ind:
                                af_predict_ending_ind = dont_care_starting_ind

                            af_preds[af_predict_starting_ind:af_predict_ending_ind] = 1

                    else:
                        # There is a previous NSR
                        r = m + 1
                        if af_predict_starting_ind <= 0:
                            # NSR is too short, relate to it as an AF event prediction interval
                            af_predict_starting_ind = 0
                            af_predict_ending_ind = af_predict_starting_ind + predict_interval

                            if af_predict_ending_ind > dont_care_starting_ind:
                                af_predict_ending_ind = dont_care_starting_ind

                            af_preds[af_predict_starting_ind:af_predict_ending_ind] = 1

                        else:
                            # There was a NSR interval, which is relatively long, prior to the AF event
                            af_predict_ending_ind = af_predict_starting_ind + predict_interval

                            if af_predict_ending_ind > dont_care_starting_ind:
                                af_predict_ending_ind = dont_care_starting_ind

                            af_preds[af_predict_starting_ind:af_predict_ending_ind] = 1
                            n_beats[0:af_predict_starting_ind] = 1

                else:
                    # Label the AF prediction interval
                    af_predict_ending_ind = af_predict_starting_ind + predict_interval

                    if af_predict_ending_ind > af_starting_ind or af_predict_ending_ind > dont_care_starting_ind:
                        # If there is a 'Don't Care' interval
                        if dont_care_starting_ind > af_predict_starting_ind:
                            af_predict_ending_ind = dont_care_starting_ind

                        # There isn't a 'Don't Care' interval
                        else:
                            af_predict_ending_ind = af_starting_ind

                    if af_predict_ending_ind < af_predict_starting_ind:
                        af_predict_ending_ind = af_predict_starting_ind

                    af_preds[af_predict_starting_ind:af_predict_ending_ind] = 1

                # 'Look Forward'
                # Find the end of the AF event
                if r < patient.annotations['N_beats'].size:
                    af_ending_r_peak = np.where(patient.annotations['R_peaks'] >=
                                                patient.annotations['N_beats'][r][0])[0][0]

                    corrected_af_ending_r_peak = af_ending_r_peak - dont_care_beats
                    af_ending_ind = patient.annotations['R_peaks'][corrected_af_ending_r_peak]

                else:
                    # Recording is ending with an AF event
                    af_ending_ind = af_beats.size - 1

                # Label the AF event
                af_beats[af_starting_ind:af_ending_ind] = 1

                # Label the 'Forward' NSR
                # Find the next AF event if there is one
                if m + 1 == patient.annotations['AF_beats'].size:
                    # No more AF events for the remainder of the recording
                    n_beats[af_ending_ind:] = 1

                else:
                    # There are more AF events upfront
                    next_af_r_peak = np.where(patient.annotations['R_peaks'] >=
                                              patient.annotations['AF_beats'][m + 1][0])[0][0]
                    next_af_ind = patient.annotations['R_peaks'][next_af_r_peak - dont_care_beats] - predict_interval

                    if next_af_ind < dont_care_ending_ind:
                        next_af_ind = dont_care_ending_ind
                        r += 1

                        ending_inds[0] = dont_care_ending_ind
                        ending_inds[1] = next_af_ind
                        ending_inds[2] = af_predict_ending_ind
                        ending_inds[3] = af_ending_ind

                        continue

                    n_beats[af_ending_ind:next_af_ind] = 1

                ending_inds[0] = dont_care_ending_ind
                ending_inds[1] = next_af_ind
                ending_inds[2] = af_predict_ending_ind
                ending_inds[3] = af_ending_ind

                r += 1

            label = np.array([dont_care, n_beats, af_preds, af_beats]).T

            if i == 0 and j == 0:
                ECG = signal
                # R_Peaks = r_peaks
                # Optimizer = entropy
                labels = label
                # r_peaks_inds = patient.annotations['R_peaks']

            else:
                ECG = np.vstack([ECG, signal])
                # R_Peaks = np.vstack([R_Peaks, r_peaks])
                # Optimizer = np.vstack([Optimizer, entropy])
                labels = np.vstack([labels, label])
                # r_peaks_inds = np.vstack([r_peaks_inds, patient.annotations['R_peaks']])

            i += 1
            k += 1

    if healthy:
        root_path = os.getcwd()
        np.save(os.path.join(root_path, 'ECG_' + db_type), ECG)

        np.save(os.path.join(root_path, 'labels_0_' + db_type), labels[:, 0])
        np.save(os.path.join(root_path, 'labels_1_' + db_type), labels[:, 1])
        np.save(os.path.join(root_path, 'labels_2_' + db_type), labels[:, 2])
        np.save(os.path.join(root_path, 'labels_3_' + db_type), labels[:, 3])

    else:
        # Calculate labels for the R Peaks
        # r_peaks_labels_0 = labels[:, 0][r_peaks_inds]
        # r_peaks_labels_1 = labels[:, 1][r_peaks_inds]
        # r_peaks_labels_2 = labels[:, 2][r_peaks_inds]
        # r_peaks_labels_3 = labels[:, 3][r_peaks_inds]

        # Prepare the labels for segment classifying
        # signal_length = int(patient.Fs * segment_length)
        # poss_inds = int(labels.shape[0] - signal_length - 1)
        # segments_labels = np.zeros([poss_inds, labels.shape[1]])

        # for k in range(poss_inds):
        #    # Find the appropriate label for the segment
        #    labels_sum = np.sum(labels[k:(k + signal_length), :], axis=0)
        #    label = np.argmax(labels_sum)
        #    segments_labels[k, label] = 1

        #  Save arrays
        root_path = os.getcwd()
        np.save(os.path.join(root_path, 'ECG_' + db_type), ECG)
        # np.save(os.path.join(root_path, 'R_Peaks_' + db_type), R_Peaks)
        # np.save(os.path.join(root_path, 'Entropy_' + db_type), Optimizer)

        np.save(os.path.join(root_path, 'labels_0_' + db_type), labels[:, 0])
        np.save(os.path.join(root_path, 'labels_1_' + db_type), labels[:, 1])
        np.save(os.path.join(root_path, 'labels_2_' + db_type), labels[:, 2])
        np.save(os.path.join(root_path, 'labels_3_' + db_type), labels[:, 3])

        # np.save(os.path.join(root_path, 'r_peaks_labels_0' + db_type), r_peaks_labels_0)
        # np.save(os.path.join(root_path, 'r_peaks_labels_1' + db_type), r_peaks_labels_1)
        # np.save(os.path.join(root_path, 'r_peaks_labels_2' + db_type), r_peaks_labels_2)
        # np.save(os.path.join(root_path, 'r_peaks_labels_3' + db_type), r_peaks_labels_3)


# db2npy(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Database\Pickles\Training_DB.pkl',
#        af_predict_time=3, segment_length=120, db_type='train')
#
# db2npy(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Database\Pickles\Validate_DB.pkl',
#        af_predict_time=3, segment_length=120, db_type='validation')
#
# db2npy(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Database\Pickles\Test_DB.pkl',
#        af_predict_time=3, segment_length=120, db_type='test')
#
# db2npy(r'B:\Studies\Semester 7\Project 1\Database\Normal_Sinus_Rythm_Mini_DB.pkl',
#        af_predict_time=3, segment_length=120, db_type='NSR_test', healthy=True)
