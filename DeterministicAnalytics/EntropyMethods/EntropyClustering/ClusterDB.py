import sklearn.cluster as cls
from DeterministicAnalytics.EntropyMethods.Utilities.CalcEntAffinityGraph import build_affinity_graph
import numpy as np


def create_features_matrices(entropies_list: list, remove_from_edges: int = 180, local_segment: int = None):
    """
    Description:

    :param entropies_list:
    :param remove_from_edges:
    :param local_segment:
    :return:
    """

    # Find shortest signal in each entropy measure
    shortest_signals = (10 ** 9) * np.ones(len(entropies_list)).astype(int)
    for i, l in enumerate(entropies_list):
        for time, ent in l:
            if time.size < shortest_signals[i]:
                shortest_signals[i] = time.size

    # Extract Entropies
    features = []
    times = []

    for k, entropy_measure in enumerate(entropies_list):
        for i, tup in enumerate(entropy_measure):
            time, ent = tup
            entropy = ent[remove_from_edges:(shortest_signals[k] - remove_from_edges)]
            entropy = np.expand_dims(entropy, axis=1)

            tm = time[remove_from_edges:(shortest_signals[k] - remove_from_edges)]
            tm = np.expand_dims(tm, axis=1)

            if i == 0:
                if local_segment is None:
                    feature_map = entropy
                    tm_mat = tm

                else:
                    # Generate a random index from which to extract a local segment
                    starting_ind = np.random.choice(a=np.arange(start=0, stop=(entropy.size - local_segment - 1)),
                                                    size=1)
                    feature_map = entropy[starting_ind:(starting_ind + local_segment)]
                    tm_mat = tm[starting_ind:(starting_ind + local_segment)]

            else:
                if local_segment is None:
                    feature_map = np.concatenate((feature_map, entropy), axis=1)
                    tm_mat = np.concatenate((tm_mat, entropy), axis=1)

                else:
                    # Generate a random index from which to extract a local segment
                    starting_ind = np.random.choice(a=np.arange(start=0, stop=(entropy.size - local_segment - 1)),
                                                    size=1)
                    entropy = entropy[starting_ind:(starting_ind + local_segment)]
                    tm = tm[starting_ind:(starting_ind + local_segment)]

                    feature_map = np.concatenate((feature_map, entropy), axis=1)
                    tm_mat = np.concatenate((tm_mat, entropy), axis=1)

        features.append(feature_map)
        times.append(tm_mat)

    return features, times


def cluster_db(entropies_list: list, n_clusters: int=2, n_init: int=20, knn: int=4, affin_knn: int = 5,
               n_jobs: int=-1, affinity_kernel: str='precomputed', remove_from_edges: int = 180,
               local_segment: int = None, merged_features: bool = True):

    """
    Description:
    This function performs the Spectral Clustering Algorithm over a given database - db, and as a default, via using
    the affinity graph as calculated by the 'CalcEntAffinityGrpah' function. The Spectral Clustering Algorithm itself,
    except for the affinity graph, is computed by the Spectral Clustering method, which is implemented in the scikit-
    learn library.

    :param entropies_list:
    :param n_clusters:
    :param n_init:
    :param knn:
    :param affin_knn:
    :param n_jobs:
    :param affinity_kernel:
    :param remove_from_edges:
    :param local_segment:
    :param merged_features:
    :return:
    """

    # Extract features
    features, times = create_features_matrices(entropies_list=entropies_list, remove_from_edges=remove_from_edges,
                                               local_segment=local_segment)

    # Construct Affinity Matrix
    nn_graph, affinity_graph = build_affinity_graph(feature_maps=features, knn_num=affin_knn,
                                                    merged_features=merged_features)

    # Define Hyper Parameters
    spect_cls = cls.SpectralClustering(n_clusters=n_clusters, n_init=n_init, affinity=affinity_kernel,
                                       n_neighbors=knn, n_jobs=n_jobs)

    # Cluster Patients
    spect_cls.fit(affinity_graph)

    return spect_cls

