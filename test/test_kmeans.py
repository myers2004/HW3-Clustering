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
        km = KMeans(k=3.7)
    with pytest.raises(ValueError):
        km = KMeans(k=0)
    random_data = np.random.rand(10, 2)
    km = KMeans(k=11)
    with pytest.raises(ValueError):
        km.fit(random_data)

    #testing invlaid tolerance
    with pytest.raises(TypeError):
        km = KMeans(k=3,tol = 4)

    #testing invalid max_int
    with pytest.raises(TypeError):
        km = KMeans(k=3,max_iter=4.3)

    #running the methods in the wrong order
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.predict(random_data)
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.get_centroids()
    with pytest.raises(RuntimeError):
        km = KMeans(k=3)
        km.get_error()
