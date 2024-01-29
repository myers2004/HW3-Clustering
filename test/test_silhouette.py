# write your silhouette score unit tests here
import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters)

from sklearn.metrics import silhouette_score
import pytest

def test_silhouette_():
    """
    TODO: Write your unit test for a silhouette scoring here. 
    Create an instance of your kmeans class and check that edge 
    cases are handled correctly. Test against sklearn.
    """

    #Check the mean score against the mean score given by sklearn
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)

    mean_score = np.mean(scores,axis=0)

    mean_sklearn_score = silhouette_score(clusters, np.ravel(pred))

    assert mean_score - mean_sklearn_score < 0.0000000001

    #Check that an error is raised if there is inly one cluster in labels
    one_cluster_label = np.zeros((500,1))
    with pytest.raises(ValueError):
        scores = Silhouette().score(clusters, one_cluster_label)

