import pickle
from scipy.io import savemat

db1 = pickle.load(open(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Long-Term_DB_1.pkl', 'rb'))
db1 = db1.db
db2 = pickle.load(open(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Long-Term_DB_2.pkl', 'rb'))
db2 = db2.db
db3 = pickle.load(open(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Long-Term_DB_3.pkl', 'rb'))
db3 = db3.db
db4 = pickle.load(open(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Long-Term_DB_4.pkl', 'rb'))
db4 = db4.db
db5 = pickle.load(open(r'B:\Studies\Semester 7\Project 1\AtrialFibrillation\Normal_Sinus_Rythm_DB.pkl', 'rb'))
db5 = db5.db

db = db1 + db2 + db3 + db4 + db5

del db1, db2, db3, db4, db5

names = []
for rec in db:
    names.append([int(rec.record_name)])

rec_names = {'Records': names}
savemat('RecordNames', rec_names)

# spectral_clusterer = spectral_ent_classifier(knn_affinity=8, knn_spectral=8, spectral_clusters=3,
#                                              spectral_re_init=20, spectral_n_jobs=-1, spectral_kernel='precomputed')
# spectral_clusterer = spectral_clusterer.cluster_patients(db)
# clusters3 = {'Clusters': spectral_clusterer.labels_}
# savemat('SpectralClusteringLabels_3', clusters3)
#
# del spectral_clusterer
#
# spectral_clusterer = spectral_ent_classifier(knn_affinity=8, knn_spectral=8, spectral_clusters=2,
#                                              spectral_re_init=20, spectral_n_jobs=-1, spectral_kernel='precomputed')
# spectral_clusterer = spectral_clusterer.cluster_patients(db)
# clusters2 = {'Clusters': spectral_clusterer.labels_}
# savemat('SpectralClusteringLabels_2', clusters2)

