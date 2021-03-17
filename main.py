from mat2csv import mat2csv

import pandas as pd
import numpy as np

from termcolor import colored
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cof import COF

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

spectral_clustering_evaluation = {
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': [],
}

kmeans_evaluation = {
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': [],
}

dbscan_evaluation = {
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': [],
}


def prepare_data(filename):
    data = pd.read_csv(filename)
    print(f'File {filename} \nIlosc wyjątków \n {data["class"].value_counts()}', )
    data = StandardScaler().fit_transform(data)
    data = normalize(data)
    data = PCA(n_components=2).fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = ['P1', 'P2']
    return data


def silhouette(dataset, labels, dictionary):
    value = silhouette_score(dataset, labels)
    dictionary['silhouette'].append(value)
    print(f'silhouette value = {value}')
    return value


def davies_bouldin(dataset, labels, dictionary):
    value = davies_bouldin_score(dataset, labels)
    print(f'davies_bouldin value = {value}')
    dictionary['davies_bouldin'].append(value)
    return value


def calinski_harabasz(dataset, labels, dictionary):
    value = calinski_harabasz_score(dataset, labels)
    print(f'calinski_harabasz value = {value}')
    dictionary['calinski_harabasz'].append(value)
    return value


def dbscan(dataset, eps, min_samples, name):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    cluster_plot(dataset, clustering, name, eps, min_samples)
    print(colored(f'Clustering quality for {name}', 'red'))
    silhouette(dataset, clustering.labels_, dbscan_evaluation)
    davies_bouldin(dataset, clustering.labels_, dbscan_evaluation)
    calinski_harabasz(dataset, clustering.labels_, dbscan_evaluation)


def kmeans(dataset, nclusters, ninit, maxiter, name):
    clustering = KMeans(n_clusters=nclusters, n_init=ninit, max_iter=maxiter).fit(dataset)
    cluster_plot(dataset, clustering, name, ninit, maxiter)
    print(colored(f'Clustering quality for {name}', 'red'))
    silhouette(dataset, clustering.labels_, kmeans_evaluation)
    davies_bouldin(dataset, clustering.labels_, kmeans_evaluation)
    calinski_harabasz(dataset, clustering.labels_, kmeans_evaluation)


def spectral_clustering(dataset, nclusters, affinity, random_state, name):
    clustering = SpectralClustering(n_clusters=nclusters, affinity=affinity, random_state=random_state).fit(dataset)
    cluster_plot(dataset, clustering, name, affinity, random_state)
    print(colored(f'Clustering quality for {name}', 'red'))
    silhouette(dataset, clustering.labels_, spectral_clustering_evaluation)
    davies_bouldin(dataset, clustering.labels_, spectral_clustering_evaluation)
    calinski_harabasz(dataset, clustering.labels_, spectral_clustering_evaluation)


def cluster_plot(dataset, clustering, name, param1, param2):
    groups = clustering.labels_
    dataset['COLORS'] = groups
    n_clusters_ = len(set(groups)) - (1 if -1 in groups else 0)
    for group in np.unique(groups):
        label = f'Cluster {group}' if group != -1 else 'Noise points'
        filtered_group = dataset[dataset['COLORS'] == group]
        plt.scatter(filtered_group['P1'], filtered_group['P2'], label=label)
    if name == 'DBSCAN':
        plt.figtext(0.05, 0.84, f'Number of clusters: {n_clusters_}\nEps: {param1}\nMin_samples: {param2}\n',
                    fontsize=15)
    elif name == 'K-Means':
        plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c="#000000", s=100)
        plt.figtext(0.05, 0.84, f'Number of clusters: {n_clusters_}\nN_init: {param1}\nMax_iter: {param2}',
                    fontsize=15)
    else:
        plt.figtext(0.05, 0.84, f'Number of clusters: {n_clusters_}\nAffinity: {param1}\nRandom_state: {param2}',
                    fontsize=15)
    plt.title(name)
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.legend()
    plt.show()


def outlier_plot(dataset, outlier_index, contamination, n_neighbors, name):
    outlier_values = dataset.iloc[outlier_index]
    number_of_outlier = len(outlier_values)
    plt.title(name, loc='center', fontsize=20)
    plt.scatter(dataset["P1"], dataset["P2"], color="b", s=65)
    plt.scatter(outlier_values["P1"], outlier_values["P2"], color="r")
    plt.figtext(0.05, 0.86,
                f'contamination = {contamination}\nn_neighbors = {n_neighbors} \nnumber of outlier = {number_of_outlier}',
                fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.show()


def outlier_remove(outlier_index, filename):
    data = pd.read_csv('satellite.csv')
    x = []
    for item in outlier_index:
        x.extend(item)
    data.drop(x, axis=0, inplace=True)
    data.to_csv(filename)


def lof(n_neighbors, contamination, name):
    dataset = prepare_data(df_names[0])
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(dataset)
    outlier_index = np.where(clf == -1)
    outlier_plot(dataset, outlier_index, contamination, n_neighbors, name)
    outlier_remove(outlier_index, 'df_without_outliers_lof.csv')


def cof(n_neighbors, contamination, name):
    dataset = prepare_data(df_names[0])
    clf = COF(n_neighbors=n_neighbors, contamination=contamination).fit_predict(dataset)
    outlier_index = np.where(clf == 1)
    outlier_plot(dataset, outlier_index, contamination, n_neighbors, name)
    outlier_remove(outlier_index, 'df_without_outliers_cof.csv')


def special_outlier_remove(filename):
    data = pd.read_csv(df_names[0])
    col = data['class']
    x = []
    for i, j in enumerate(col):
        if j == 1:
            x.append(i)
    data.drop(x, axis=0, inplace=True)
    data.to_csv(filename)


file_name = 'satellite'
mat2csv(file_name + '.mat', file_name + '.csv')
df_names = [file_name + '.csv', 'df_without_outliers_by_hand.csv', 'df_without_outliers_lof.csv',
            'df_without_outliers_cof.csv']

lof(36, 0.32, 'LOF')
cof(36, 0.32, 'COF')

special_outlier_remove('df_without_outliers_by_hand.csv')

for df_name in df_names:
    df = prepare_data(df_name)
    spectral_clustering(df, 2, 'nearest_neighbors', 0, 'Spectral_Clustering')
    kmeans(df, 6, 10, 300, 'K-Means')
    dbscan(df, 0.5, 5, 'DBSCAN')


def evaluation_comparison(dictionary, name):
    for element in dictionary:
        plt.bar(['with', 'without', 'lof', 'cof'], dictionary[element])
        plt.title(f'{name}, evaluation method {element}')
        plt.show()


evaluation_comparison(kmeans_evaluation, 'K-means')
evaluation_comparison(dbscan_evaluation, 'DBSCAN')
evaluation_comparison(spectral_clustering_evaluation, 'Spectral_Clustering')
