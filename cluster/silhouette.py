import numpy as np
from scipy.spatial.distance import cdist
import warnings


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

        k = int(np.max(y))              #The number of cluster labels loaded in
        num_obs, num_feat = np.shape(X) #The dimensionality of the data matrix loaded in

        #Need at least two clusters to have an inter-cluster distance
        if k == 0:
            raise(ValueError('Need at least two clusters to calculate silhouette scores'))
        

        #Has a key for each observation, updated with scores as they are calculated
        silhouette_scores = {i: 0.0 for i in range(num_obs)}

        #Loop through every cluster (labeled 0 to k) and calculate silhouette 
        # socre for every point in the cluster
        for i in range(k+1):
            points = np.asarray(y == i).nonzero()[0] #The row numbers of every point in cluster i

            #Start by finding a_i for each point, the average distance other points 
            # belonging to the same cluster

            cluster_i = X[y[:,0] == float(i)]       #All points belonging to the cluster labeled i

            #Using cdist, we get a distance matrix holding the distance to every point in the 
            # cluster for every point in the cluster

            intra_dist = cdist(cluster_i,cluster_i) 
            a = [] #to hold all the a_i's
            num_in_cluster = np.shape(intra_dist)[0] #The number of observations in cluster i

            #Loop through the distance matrix, finding the average of the distance for every point
            for j in range(num_in_cluster):
                sum = 0
                for h in range(num_in_cluster):
                    sum += intra_dist[j][h]
                #calculate a_i for each point
                if num_in_cluster - 1 == 0:
                    warnings.warn('Cluster of size 1 encountered, a_i set to 0.')
                    a_i = 0.0
                else:
                    a_i = sum / (num_in_cluster - 1) #-1 as we subtract out the distance from a point to itself
                a.append(a_i)                        #add to "a" list
            
            #Fnding b_i for each point, the average distance to other points 
            # belonging to different clusters

            
            #Has a key going to an empty list for every point in cluster i, updated with 
            # average distance to every other cluster for each point as they are calculated
            inter_dist_avg = {j: [] for j in range(num_in_cluster)} 
            for m in range(k+1):
                #Avoid current cluster
                if m != i:
                    cluster_m = X[y[:,0] == float(m)] #all points belonging to another cluster m
                    
                    #Using cdist, get a distance matrix holding the distance to every point in 
                    # cluster m for every point in cluster i
                    inter_dist = cdist(cluster_i,cluster_m)

                    #Loop through the distance matrix, finding the average of the distance for every point
                    num_in_cluster_m = np.shape(inter_dist)[1]
                    for n in range(num_in_cluster):
                        sum = 0
                        for p in range(num_in_cluster_m):
                            sum += inter_dist[n][p]
                        #Calculate the average distance and add to dictionary
                        inter_dist_avg[n].append(sum / num_in_cluster_m)
            b = [] #to hold the b_i's

            #add the minumum average distance to b
            for j in range(num_in_cluster):
                b.append(np.min(inter_dist_avg[j]))

            #calculate the silhouette score for each point in cluster i
            for i in range(len(points)):
                silhouette_scores[points[i]] = (b[i] - a[i])/np.max((b[i], a[i]))

        #go from the disctionary to an array
        silhouette_scores_array = np.zeros((num_obs, 1))
        for key in silhouette_scores.keys():
            silhouette_scores_array[key] = silhouette_scores[key]
        
        return silhouette_scores_array





