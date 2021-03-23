from mat2csv import mat2csv
import json
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


def clustering_quality(method, name, dictionary):
    value = method
    dictionary[name].append(value)
    print(f'{name} value = {value}')
    return value


def methods(dataset, clustering, params, dictionary):
    cluster_plot(dataset, clustering, params)
    print(colored(f"Clustering quality for {params['name']}", 'red'))
    clustering_quality(silhouette_score(dataset, clustering.labels_), 'silhouette', dictionary)
    clustering_quality(davies_bouldin_score(dataset, clustering.labels_), 'davies_bouldin', dictionary)
    clustering_quality(calinski_harabasz_score(dataset, clustering.labels_), 'calinski_harabasz', dictionary)


def kmeans(dataset, params):
    clustering = KMeans(n_clusters=params['nclusters'], n_init=params['ninit'], max_iter=params['maxiter']).fit(
        dataset)
    methods(dataset, clustering, params, kmeans_evaluation)


def dbscan(dataset, params):
    clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(dataset)
    methods(dataset, clustering, params, dbscan_evaluation)


def spectral_clustering(dataset, params):
    clustering = SpectralClustering(n_clusters=params['nclusters'], affinity=params['affinity'],
                                    random_state=params['random_state']).fit(dataset)
    methods(dataset, clustering, params, spectral_clustering_evaluation)


def cluster_plot(dataset, clustering, params):
    groups = clustering.labels_
    dataset['COLORS'] = groups
    n_clusters_ = len(set(groups)) - (1 if -1 in groups else 0)
    for group in np.unique(groups):
        label = f'Cluster {group}' if group != -1 else 'Noise points'
        filtered_group = dataset[dataset['COLORS'] == group]
        plt.scatter(filtered_group['P1'], filtered_group['P2'], label=label)
    if params['name'] == 'DBSCAN':
        plt.figtext(0.05, 0.84,
                    f"Number of clusters: {n_clusters_}\nEps: {params['eps']}\nMin_samples: {params['min_samples']}\n",
                    fontsize=15)
    elif params['name'] == 'K-Means':
        plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c="#000000", s=100)
        plt.figtext(0.05, 0.84,
                    f"Number of clusters: {n_clusters_}\nN_init: {params['ninit']}\nMax_iter: {params['maxiter']}",
                    fontsize=15)
    else:
        plt.figtext(0.05, 0.84,
                    f"Number of clusters: {n_clusters_}\nAffinity: {params['affinity']}\nRandom_state: {params['random_state']}",
                    fontsize=15)
    plt.title(params['name'])
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
    data = pd.read_csv(f'{file_name}.csv')
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


def read_config(filename='config.json'):
    with open(filename, 'r') as f:
        return json.load(f)


params = read_config()
file_name = 'satellite'
mat2csv(f'{file_name}.mat', f'{file_name}.csv')
df_names = [f'{file_name}.csv', 'df_without_outliers_by_hand.csv', 'df_without_outliers_lof.csv',
            'df_without_outliers_cof.csv']

lof(36, 0.32, 'LOF')
cof(36, 0.32, 'COF')

special_outlier_remove('df_without_outliers_by_hand.csv')

for df_name in df_names:
    df = prepare_data(df_name)
    kmeans(df, params['K-Means'])
    dbscan(df, params['DBSCAN'])
    # spectral_clustering(df, params['Spectral_Clustering'])


def evaluation_comparison(dictionary, name):
    for element in dictionary:
        plt.bar(['with', 'without', 'lof', 'cof'], dictionary[element])
        plt.title(f'{name}, evaluation method {element}')
        plt.show()


evaluation_comparison(kmeans_evaluation, 'K-means')
evaluation_comparison(dbscan_evaluation, 'DBSCAN')
# evaluation_comparison(spectral_clustering_evaluation, 'Spectral_Clustering')
