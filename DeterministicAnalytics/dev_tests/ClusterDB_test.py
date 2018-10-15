from DeterministicAnalytics.EntropyMethods.EntropyClustering.ClusterDB import cluster_db

import pickle
import numpy as np


data = pickle.load(open(r"B:\Studies\Master's\CardiacDiagnostics\Database\Ent_LT_NSR_AF_DBS.pkl", 'rb'))
entropies = data['Entropy']

record_names = data['Records_Names']
record_names = record_names[0:124]
record_names.__delitem__(4)

print(record_names)

spect_cls = cluster_db(entropies_list=entropies, n_clusters=3, knn=6, affin_knn=8, merged_features=False)

file = open(r"B:\Studies\Master's\CardiacDiagnostics\Database\EntropyClusters.pkl", 'wb')
pickle.dump(obj=spect_cls, file=file)
file.close()

labels = spect_cls.labels_

zero_inds = np.where(labels == 0)
one_inds = np.where(labels == 1)
two_inds = np.where(labels == 2)

zero_inds = zero_inds[0].tolist()
one_inds = one_inds[0].tolist()
two_inds = two_inds[0].tolist()

zero_files = []
one_files = []
two_files = []

for ind in zero_inds:
    zero_files.append(record_names[ind])

for ind in one_inds:
    one_files.append(record_names[ind])

for ind in two_inds:
    two_files.append(record_names[ind])

clusters = [zero_files, one_files, two_files]

file = open(r"B:\Studies\Master's\CardiacDiagnostics\Database\Clusteres.pkl", 'wb')
pickle.dump(obj=clusters, file=file)
file.close()
