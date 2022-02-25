import numpy as np
import math

import initialization_methods as init_methods
from scipy.spatial.distance import cdist


class FuzzyClustering:

    def __init__(self, n_clusters=3, tolerance=0.01, max_iter=100, runs=1, fuzziness=2,
                 init_method="forgy", method="Cmeans", debug=False):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter
        self.init_method = init_method
        self.debug = debug
        self.method = method
        self.fuzziness = fuzziness

        # There is no need to run the algorithm multiple times if the
        # initialization method is not a random process
        self.runs = runs if init_method == 'forgy' else 1

    def _init_mem(self, X):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        # initialize memberships
        rnd = np.random.RandomState(42)
        memberships = rnd.rand(n_clusters, n_samples)

        # update membership relative to classes
        summation = memberships.sum(axis=0).reshape(-1, 1)
        denominator = np.repeat(summation, n_clusters, axis=1).T
        memberships = memberships / denominator
        return memberships

    def fit(self, X):
        row_count, col_count = X.shape
        X_values = self.__get_values(X)
        X_labels = np.zeros(row_count)

        costs = np.zeros(self.runs)
        all_clusters = []
        membership = []
        counter = 0

        for i in range(self.runs):
            cluster_centers = self.__initialize_means(X_values, row_count)
            # print(row_count, self.n_clusters)
            # membership = np.zeros((self.n_clusters, row_count))
            membership = self._init_mem(X_values)

            for it in range(self.max_iter):
                counter += 1
                if self.debug:
                    print("iter number : ", counter)
                previous_membership = np.copy(membership)
                # calculate the new centers of the clusters
                cluster_centers = self.__compute_new_clusters(X_values, membership)
                # compute the distances of all samples from all clusters
                distances = self.__compute_distances(X, cluster_centers, previous_membership)
                # compute the membership between all samples and all clusters
                membership = self.__compute_membership(X_values, distances)
                # print("member", membership)
                if self.debug:
                    print("iteration number : ", it, "\n", cluster_centers)
                # print("cluster", cluster_means)

                # clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                membership_not_changed = np.abs(membership - previous_membership) < self.tolerance
                if np.all(membership_not_changed) != False:
                    break

            X_values_with_labels = np.append(X_values, X_labels[:, np.newaxis], axis=1)

            all_clusters.append((cluster_centers, X_values_with_labels))
            costs[i] = self.__compute_cost(X_values, X_labels, cluster_centers)

        best_clustering_index = costs.argmin()

        self.cost_ = costs[best_clustering_index]

        print("number of iteration = ", counter)
        # return all_clusterings[best_clustering_index]
        return (cluster_centers, membership.transpose())

    def _calculate_fuzzyCov(self, X, memberships, new_class_centers):
        # calculating covariance matrix in its fuzzy form
        fuzzy_mem = (memberships ** self.fuzziness).T
        n_clusters = self.n_clusters
        FcovInv_Class = []
        dim = X.shape[1]
        for i in range(n_clusters):
            diff = X - new_class_centers[i]
            left = np.dot((fuzzy_mem[:, i].reshape(-1, 1) * diff).T, diff) / np.sum(fuzzy_mem[:, i], axis=0)
            Fcov = ((np.linalg.det(left)) ** (-1 / dim)) * left
            FcovInv = np.linalg.inv(Fcov)
            FcovInv_Class.append(FcovInv)

        return FcovInv_Class

    def __compute_distances(self, X, cluster, memberships):
        if self.method == "Gustafson–Kessel":
            n_clusters = self.n_clusters
            FcovInv_Class = self._calculate_fuzzyCov(X, memberships, cluster)

            # calculating mahalanobis distance
            mahalanobis_Class = np.zeros((n_clusters, X.shape[0]))

            for i in range(n_clusters):
                for k in range(X.shape[0]):
                    diff = X[k] - cluster[i]
                    left = np.matmul(diff.T, FcovInv_Class[i])
                    mahalanobis = np.matmul(left, diff)
                    mahalanobis_Class[i][k] = mahalanobis
            distance = np.array(mahalanobis_Class).T
            return distance
        elif self.method == "Cmeans":
            # distances = np.zeros((row_count, self.n_clusters))
            # for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            #   distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis=1)
            distance = cdist(X, cluster, metric="euclidean")
            return distance

    def __compute_membership(self, X, distances):
        row_count = X.shape[0]

        membership = np.zeros((self.n_clusters, row_count))
        temp2 = np.zeros((row_count, self.n_clusters))
        with np.errstate(divide='ignore', invalid='ignore'):
            for cluster_index in range(self.n_clusters):
                for j in range(self.n_clusters):
                    temp2[:, j] = (distances[:, cluster_index] / distances[:, j]) ** (2 / (self.fuzziness - 1))
                membership[cluster_index, :] = 1 / np.sum(temp2, axis=1)
        where_are_NaNs = np.isnan(membership)
        if self.debug:
            print("where are nan ", where_are_NaNs[where_are_NaNs == True])
        membership[where_are_NaNs] = 1

        return membership

    def __compute_new_clusters(self, X, membership):
        fuzzy_mem = membership ** self.fuzziness
        #13.6
        # new_clusters = np.zeros((self.n_clusters, X.shape[1]))
        # for i in range(self.n_clusters):
        #     temp = np.reshape(fuzzy_mem[i], (X.shape[0], -1)) * X
        #     temp = np.sum(temp, axis=0)
        #     temp2 = np.sum(fuzzy_mem[i])
        #     new_clusters[i] = temp / temp2
        new_clusters = (np.dot(X.T, fuzzy_mem.T) / np.sum(fuzzy_mem, axis=1)).T  # 12.6
        return new_clusters

    # def __compute_distances(self, X, cluster_means, row_count):
    #     distances = np.zeros((row_count, self.n_clusters))
    #     for cluster_mean_index, cluster_mean in enumerate(cluster_means):
    #         distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis=1)
    #
    #     return distances

    def __initialize_means(self, X, row_count):
        if self.init_method == 'forgy':
            return init_methods.forgy(X, row_count, self.n_clusters)
        elif self.init_method == 'maximin':
            return init_methods.maximin(X, self.n_clusters)
        elif self.init_method == 'macqueen':
            return init_methods.macqueen(X, self.n_clusters)
        elif self.init_method == 'var_part':
            return init_methods.var_part(X, self.n_clusters)
        else:
            raise Exception('The initialization method {} does not exist or not implemented'.format(self.init_method))

    def __label_examples(self, distances):
        return distances.argmin(axis=1)

    def __compute_cost(self, X, labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X[labels == cluster_mean_index]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis=1).sum()

        return cost

    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)


