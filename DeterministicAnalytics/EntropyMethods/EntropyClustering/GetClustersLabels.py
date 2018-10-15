from DeterministicAnalytics.EntropyMethods.EntropyClustering.ClusterDB import cluster_db

import numpy as np
import pickle


lt1_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_1.pkl', 'rb')).db
lt2_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_2.pkl', 'rb')).db

db = lt1_db + lt2_db
del lt1_db, lt2_db

lt3_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_3.pkl', 'rb')).db

db = db + lt3_db
del lt3_db

lt4_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_4.pkl', 'rb')).db

db = db + lt4_db
del lt4_db

lt5_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_5.pkl', 'rb')).db

db = db + lt5_db
del lt5_db

lt6_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_6.pkl', 'rb')).db

db = db + lt6_db
del lt6_db

lt7_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_7.pkl', 'rb')).db

db = db + lt7_db
del lt7_db

lt8_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Long-Term_DB_8.pkl', 'rb')).db

db = db + lt8_db
del lt8_db

nsr_db = pickle.load(open(r'B:\Studies\Semester 7\Project 1\Database\Normal_Sinus_Rythm_DB.pkl', 'rb')).db

db = db + nsr_db
del nsr_db

clusters = cluster_db(db, n_clusters=3, knn=4, affin_knn=5, n_init=30)
clusters_labels = clusters.labels_

cluster_zero = np.where(clusters_labels == 0)
cluster_one = np.where(clusters_labels == 1)
cluster_two = np.where(clusters_labels == 2)

cluster_zero = np.array(cluster_zero[0]).astype(np.int).tolist()
cluster_one = np.array(cluster_one[0]).astype(np.int).tolist()
cluster_two = np.array(cluster_two[0]).astype(np.int).tolist()

cluster_zero_names = []
for ind in cluster_zero:
    cluster_zero_names.append(db[ind].record_name)

cluster_one_names = []
for ind in cluster_one:
    cluster_one_names.append(db[ind].record_name)

cluster_two_names = []
for ind in cluster_two:
    cluster_two_names.append(db[ind].record_name)

np.save('B:\Studies\Semester 8\Project\AF Seperations\\cluster_0.npy', cluster_zero_names)
np.save('B:\Studies\Semester 8\Project\AF Seperations\\cluster_1.npy', cluster_one_names)
np.save('B:\Studies\Semester 8\Project\AF Seperations\\cluster_2.npy', cluster_two_names)
