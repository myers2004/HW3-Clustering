# Write your k-means unit tests here
import numpy as np
from cluster import (
        KMeans,
        make_clusters)

import pytest

def test_kmeans_():
    """
    TODO: Write your unit test for a k-means clustering here. 
    Create an instance of your kmeans class and check that edge 
    cases are handled correctly
    """
    
    #Testing invalid ks
    with pytest.raises(TypeError):
        km = KMeans(k=3.7) #not int
    with pytest.raises(ValueError):
        km = KMeans(k=0) #k must be >= 1
    random_data = np.random.rand(100, 2)
    km = KMeans(k=101) 
    with pytest.raises(ValueError):
        km.fit(random_data) #need to have less k than number of observations

    #testing invlaid tolerance
    with pytest.raises(TypeError):
        km = KMeans(k=3,tol = 4) #tol must be float

    #testing invalid max_int
    with pytest.raises(TypeError):
        km = KMeans(k=3,max_iter=4.3) #max_inter must be int

    #running the methods in the wrong order
    #fit must run first
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.predict(random_data)
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.get_centroids()
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.get_error()

    #Tet an error is raised if a non-numeric element is in the input array
    with pytest.raises(ValueError):
        test = np.array([[4,3, 'D', 'E']])
        km.fit(test)

    #Test an error is raised if something other than a numpy array is input into fit
    with pytest.raises(ValueError):
        km.fit(4)
