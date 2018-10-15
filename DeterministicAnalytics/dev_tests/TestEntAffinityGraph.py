from DeterministicAnalytics.EntropyMethods.Utilities.CalcEnt import EntropyManipulator
from DeterministicAnalytics.EntropyMethods.Utilities.CalcEntAffinityGraph import build_affinity_graph

import time
import pickle
import numpy as np

print('Starting testing script...' + '\n')
db = pickle.load(open(r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_1.pkl", 'rb'))
db = db.db

fs = db[0].Fs
start_ind = 10

print('Database loading time is:' + '\n', time.clock())

# Compute Optimizer
ent_calc = EntropyManipulator()
n = 1
patients_ent1 = []
patients_ent2 = []
patients_ent3 = []

time.clock()
print('Class initialization time is:' + '\n', time.clock())

end_ind_1 = 10 ** 9
end_ind_2 = 10 ** 9
end_ind_3 = 10 ** 9

for i, patient in enumerate(db):
    print('\n' + 'Caculating Entropy for patient ' + str(i + 1) + ' out of ' + str(len(db)) + ' patients.')

    print('\n' + 'Calculating first entropy measure')
    ent_1 = ent_calc.calc_multi_scale_entropy_moment(time=patient.qrs_time[start_ind:],
                                                     rr_peaks=(patient.annotations['R_peaks'][start_ind:] / fs),
                                                     tau=1, order=2)

    print('\n' + 'Calculating second entropy measure')
    ent_2 = ent_calc.calc_multi_scale_entropy_moment(time=patient.qrs_time[start_ind:],
                                                     rr_peaks=(patient.annotations['R_peaks'][start_ind:] / fs),
                                                     tau=10, order=2)

    print('\n' + 'Calculating third entropy measure')
    ent_3 = ent_calc.calc_multi_scale_entropy_moment(time=patient.qrs_time[start_ind:],
                                                     rr_peaks=(patient.annotations['R_peaks'][start_ind:] / fs),
                                                     tau=1, order=4)

    patients_ent1.append(ent_1)
    patients_ent2.append(ent_2)
    patients_ent3.append(ent_3)

    print('\n' + 'Current iteration time is:' + '\n', time.clock())

    if ent_1[0].size < end_ind_1:
        end_ind_1 = ent_1[0].size - 1

    if ent_2[0].size < end_ind_2:
        end_ind_2 = ent_2[0].size - 1

    if ent_3[0].size < end_ind_3:
        end_ind_3 = ent_3[0].size - 1

# Construct Affinity Matrix
knn_num = 4
patients_ent_array_1 = np.zeros((end_ind_1, len(db)))
patients_ent_array_2 = np.zeros((end_ind_2, len(db)))
patients_ent_array_3 = np.zeros((end_ind_3, len(db)))

for i, ent in enumerate(patients_ent1):
    patients_ent_array_1[:, i] = patients_ent1[i][0][0:end_ind_1]

for i, ent in enumerate(patients_ent2):
    patients_ent_array_2[:, i] = patients_ent2[i][0][0:end_ind_2]

for i, ent in enumerate(patients_ent3):
    patients_ent_array_3[:, i] = patients_ent3[i][0][0:end_ind_3]

patients_ent_array = [patients_ent_array_1, patients_ent_array_2, patients_ent_array_3]

afffinity_g = build_affinity_graph(patients_ent_array, knn_num=knn_num)





