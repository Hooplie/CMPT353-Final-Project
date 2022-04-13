import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from matplotlib import cm

def best_k(data):
    max_clusters = 10
    silhouette_coef = np.ndarray(max_clusters-1)
    x_vals = list(range(2, max_clusters))
    data = data.fillna(data.mean())
    for k in x_vals:
        clusters = cluster.KMeans(n_clusters=k, init='k-means++')
        clusters.fit(data)
        sc = metrics.silhouette_score(data, clusters.labels_)
        silhouette_coef[k-2] = sc
    # plt.scatter(x_vals, silhouette_coef)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Coefficient')
    # plt.title('Silhouette Coefficients of Clusterings \nwith Different Numbers of Clusters')
    # plt.show()
    return np.argmax(silhouette_coef) + 2

def plot_clusters(data, n_clusters):
    data = data.fillna(data.mean())
    clusters = cluster.KMeans(n_clusters, init='k-means++')
    clusters.fit(data)
    print(metrics.silhouette_score(data, clusters.labels_))
    print(clusters.cluster_centers_)
    plt.scatter(data.iloc[:,0], data.iloc[:,1], c=clusters.labels_, cmap=cm.get_cmap('Set1'), alpha=0.5)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title('Visualization of Clustering')
    plt.show()