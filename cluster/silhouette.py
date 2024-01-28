import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        k = int(np.max(y))
        num_obs, num_feat = np.shape(X)
 
        silhouette_scores = {i: 0.0 for i in range(num_obs)}

        for i in range(k+1):
            points = np.asarray(y == i).nonzero()[0]
            print(len(points))
            #First get the mean dist from other points in a cluster for each point
            cluster_i = X[y[:,0] == float(i)]
            intra_dist = cdist(cluster_i,cluster_i)
            a = []
            num_in_cluster = np.shape(intra_dist)[0]
            for j in range(num_in_cluster):
                sum = 0
                for h in range(num_in_cluster):
                    sum += intra_dist[j][h]
                a_i = sum / (num_in_cluster)
                a.append(a_i)

            inter_dist_avg = {j: [] for j in range(num_in_cluster)}
            for m in range(k):
                if m != i:
                    cluster_m = X[y[:,0] == float(m)]
                    inter_dist = cdist(cluster_i,cluster_m)
                    num_in_cluster_m = np.shape(inter_dist)[1]
                    for n in range(num_in_cluster):
                        sum = 0
                        for p in range(num_in_cluster_m):
                            sum += inter_dist[n][p]
                        inter_dist_avg[n].append(sum / num_in_cluster_m)
            b = []
            for j in range(num_in_cluster):
                b.append(np.min(inter_dist_avg[j]))

            for i in range(len(points)):
                silhouette_scores[points[i]] = (b[i] - a[i])/np.max((b[i], a[i]))

        silhouette_scores_array = np.zeros((num_obs, 1))
        for key in silhouette_scores.keys():
            silhouette_scores_array[key] = silhouette_scores[key]
        
        return silhouette_scores_array





