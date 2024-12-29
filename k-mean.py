import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def kmeans_clustering(data, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels, kmeans

def em_clustering(data, n_components):
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    cluster_labels = gmm.fit_predict(data)
    return cluster_labels, gmm

def compare_clustering(data, n_clusters):
    
    # Apply K-Means
    kmeans_labels, kmeans_model = kmeans_clustering(data, n_clusters)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)

    # Apply EM (Gaussian Mixture Model)
    em_labels, gmm_model = em_clustering(data, n_components=n_clusters)
    em_silhouette = silhouette_score(data, em_labels)

    # Add cluster labels to data for visualization
    data['KMeans_Cluster'] = kmeans_labels
    data['EM_Cluster'] = em_labels

    # Visualize clustering results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='KMeans_Cluster', palette='viridis', ax=axes[0])
    axes[0].set_title(f"K-Means Clustering (Silhouette: {kmeans_silhouette:.2f})")
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='EM_Cluster', palette='coolwarm', ax=axes[1])
    axes[1].set_title(f"EM Clustering (Silhouette: {em_silhouette:.2f})")
    plt.show()

    # Print silhouette scores for comparison
    print(f"Silhouette Score - K-Means: {kmeans_silhouette:.2f}")
    print(f"Silhouette Score - EM: {em_silhouette:.2f}")

# Example usage
csv_file = 'data.csv'  # Replace with your CSV file path
n_clusters = 3         # Number of clusters

# Load data
data = pd.read_csv(csv_file)

# Ensure only numerical columns are used for clustering
numerical_data = data.select_dtypes(include=['number'])

# Compare clustering methods
compare_clustering(numerical_data, n_clusters)
