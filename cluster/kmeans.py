import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        #ensure input arguments are the right type
        if type(k) != int:
            raise(TypeError('k is not of type int'))
        if k < 1:
            raise(ValueError('k must be at least 1'))
        if type(tol) != float:
            raise(TypeError('tol is not of type float'))
        if type(max_iter) != int:
            raise(TypeError('max_int is not of type int'))
        
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.was_fit_run = False

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if type(mat) != np.ndarray:
            raise(ValueError('Input data is not a numpy array.'))

        SSE = 0 #a dummy SSE(sum of squared distances from the points in the clsuters to their centers)
        num_obs, num_feat = np.shape(mat) #extract number of observations and features in mat

        for i in range(num_obs):
            for j in range(num_feat):
                if type(mat[i][j]) == np.int64 or type(mat[i][j]) == np.float64:
                    pass
                else:
                    raise(ValueError('Non-numeric present in data'))

        if self.k > num_obs:
            raise(ValueError('k must be less than or equal to the number of observations.'))

        #Pick random points in mat to as the intial cluster centers
        random_rows = np.random.choice(num_obs, self.k)
        cluster_centers = mat[random_rows, :]

        #Now to do Loyd's algorithm
        for a in range(0, self.max_iter):
            cluster_membership = np.zeros((num_obs, self.k)) #intialize r matrix, denoting which matrix each point belongs to

            dists = cdist(mat, cluster_centers) #calculate distance to each cluster center for every point


            #Assign each point to the cluster to which it is closest to the center of
            for j in range(num_obs):
                curr_min_dist = 0
                for i in range(1, self.k):
                    if dists[j][i] < dists[j][curr_min_dist]:
                        curr_min_dist = i
                cluster_membership[j][curr_min_dist] = 1
            
            new_SSE = 0 #To hold the new SSE

            new_clusters = np.zeros((self.k, num_feat)) #to hold the new cluster centers

            #Calculate the new center point of each cluster, and the SSE for the new clusters
            for i in range(self.k):
                cluster_i = mat[cluster_membership[:, i] == 1] #extracts every row that is a member of cluster i
                for j in range(np.shape(cluster_i)[0]):
                    for m in range(num_feat):
                        new_SSE += np.power(cluster_i[j][m] - cluster_centers[i][m], 2)
                new_clusters[i] = np.mean(cluster_i, axis = 0)
            
            #If the change in SSE is below tolerance, stop Loyd's
            if np.abs(new_SSE - SSE) < self.tol:
                break
            
            #update the cluster centers and the SSE and repeat
            cluster_centers = new_clusters
            SSE = new_SSE

        #Don't want to loop forever if change in SSE never goes below tolerance
        if a == self.max_iter:
            raise(RuntimeError('K-means fit failed to converge.'))
        
        #Save the fitted cluster centers to the object
        self.cluster_centers = cluster_centers

        #Save the final SSE
        self.SSE = SSE

        #Track that the fit method was run
        self.was_fit_run = True


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if not self.was_fit_run:
            raise(RuntimeError('predict called before fit. Please run fit first'))
        num_obs, num_feat = np.shape(mat) #extract number of observation and features
        cluster_membership = np.zeros((num_obs, self.k)) #intialize r matrix, denoting which matrix each point belongs to

        dists = cdist(mat, self.cluster_centers) #compute distance to cluster centers from fit method for each observation

        #Assign each point to the cluster to which it is closest to the center of
        for j in range(num_obs):
            curr_min_dist = 0
            for i in range(1, self.k):
                if dists[j][i] < dists[j][curr_min_dist]:
                    curr_min_dist = i
            cluster_membership[j][curr_min_dist] = 1

        #Create the cluster label matrix to be output
        cluster_labels = np.zeros((num_obs, 1))
        for i in range(num_obs):
            for j in range(self.k):
                if cluster_membership[i][j] == 1:
                    cluster_labels[i][0] = j
        
        return cluster_labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        if not self.was_fit_run:
            raise(RuntimeError('get_error called before fit. Please run fit first'))

        return self.SSE

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        if not self.was_fit_run:
            raise(RuntimeError('get_centroids called before fit. Please run fit first'))

        return self.cluster_centers
