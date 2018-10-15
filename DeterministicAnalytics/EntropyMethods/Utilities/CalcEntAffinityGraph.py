import numpy as np


def build_affinity_graph(feature_maps: list, knn_num: int = 4, merged_features: bool = False):
    """
    Description:
    This function builds the affinity graph between different 1D vectors, used in the Spectral Clustering algorithm.
    It uses an affinity measure of: A = exp(-sum{|v_1 - v_2|}), over each feature map, it then sums over all
    the resulting affinity matrices, in order to calculate a final affinity measure.
    We employs two magnitude regularization at two points:
    Prior to exponentiation, and after the exponentiation.

    :param feature_maps: list - A list of I matrices, each of size M_i X N, with M_i representing the length of each
                         feature vector in the current feature map, and N representing the number of different samples.
    :param knn_num: int - Optional - Indicates the number of nearest neighbors to use, while building the Laplacian
                    matrix, to be further used in the Spectral Clustering algorithm. Default value = 4.
    :param merged_features: bool - Optional -
    :return: nn_graph, affinity_graph - two NumPy.ndarray matrices of size N X N. - nn_graph is a sparse, {0, 1} graph,
             where:

             nn_graph[i, j] = 1, if j is the nearest neighbor of i, and nn_graph[i, j] = 0, otherwise.

             Notice that the nn_graph is not necessarily symmetric. affinity_graph - is a matrix holding at each
             element the affinity measure:

            affinity_graph[i, j] = exp(-sum{|v_i - v_j|}).
    """

    # Setup
    samples_num = feature_maps[0].shape[1]
    features_num = len(feature_maps)
    affinity_maps = np.zeros([features_num, samples_num, samples_num])
    nn_graph = np.zeros([samples_num, samples_num])

    if merged_features:
        for k, feature in enumerate(feature_maps):
            for i in range(samples_num):
                # Local setup
                current_sample = feature[:, i]

                # Create a matrix where each column is equal to 'current_sample' in order to easily compute distances.
                current_sample_clone_matrix = np.multiply(current_sample, np.ones([samples_num, 1]))
                current_sample_clone_matrix = current_sample_clone_matrix.transpose()

                # Calculate the L1 distance of the i-th sample from all other samples
                distances = np.abs(feature - current_sample_clone_matrix)
                distances = np.sum(distances, axis=0)

                # Calculate & Store Affinity
                affinity_maps[k, :, i] = distances

            # Apply normalization for the current affinity map
            affinity_maps[k, :, :] /= np.max(affinity_maps[k, :, :])

            # Calculate the final affinity measure and perform another regularization
            affinity_maps[k, :, :] = np.exp(-affinity_maps[k, :, :]) / np.max(affinity_maps[k, :, :])

        affinity_graph = np.sum(affinity_maps, axis=0)
        sorted_dist_graph = np.sort(affinity_graph, axis=0)

        for i in range(samples_num):
            for j in range(knn_num):
                ind = np.where(affinity_graph[:, i] == sorted_dist_graph[:, i][(samples_num - 2 - j)])
                nn_graph[ind, i] = 1

    else:
        # First concatenate all of the features vectors for each sample together & Normalize each feature map
        for i, feature_map in enumerate(feature_maps):
            if i == 0:
                concat_features = feature_map / np.max(feature_map)

            else:
                concat_features = np.concatenate((concat_features, (feature_map / np.max(feature_map))), axis=0)

        # From continue regularly
        affinity_maps = np.zeros([samples_num, samples_num])
        for i in range(samples_num):
            # Local setup
            current_sample = concat_features[:, i]

            # Create a matrix where each column is equal to 'current_sample' in order to easily compute distances.
            current_sample_clone_matrix = np.multiply(current_sample, np.ones([samples_num, 1]))
            current_sample_clone_matrix = current_sample_clone_matrix.transpose()

            # Calculate the L1 distance of the i-th sample from all other samples
            distances = np.abs(concat_features - current_sample_clone_matrix)
            distances = np.sum(distances, axis=0)

            # Calculate & Store Affinity
            affinity_maps[:, i] = distances

        affinity_graph = np.sum(affinity_maps, axis=0)
        sorted_dist_graph = np.sort(affinity_graph, axis=0)

        for i in range(samples_num):
            for j in range(knn_num):
                ind = np.where(affinity_graph[:, i] == sorted_dist_graph[:, i][(samples_num - 2 - j)])
                nn_graph[ind, i] = 1

    return nn_graph, affinity_graph
