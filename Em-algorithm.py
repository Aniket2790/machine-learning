import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def em_clustering(csv_file, n_components=2):
   
    # Load data from CSV
    data = pd.read_csv(csv_file)
    
    # Convert to numerical data if necessary
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")
    
    # Check if there are numerical columns for clustering
    numerical_data = data.select_dtypes(include=['number'])
    if numerical_data.empty:
        raise ValueError("No numerical data found in the CSV file for clustering.")
    
    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(numerical_data)
    
    # Predict cluster labels
    cluster_labels = gmm.predict(numerical_data)
    data['Cluster'] = cluster_labels  # Add cluster labels to the original DataFrame
    
    return data, gmm

def plot_clusters(data, cluster_column='Cluster'):
 
    if 'Cluster' not in data.columns:
        raise ValueError(f"'{cluster_column}' column not found in the DataFrame.")
    
    sns.pairplot(data, hue=cluster_column, diag_kind='kde', palette='viridis')
    plt.show()

# Example usage
csv_file = 'data.csv'  # Replace with your CSV file path
n_components = 3       # Number of clusters

try:
    clustered_data, gmm_model = em_clustering(csv_file, n_components=n_components)
    print(clustered_data.head())  # Display the first few rows of the clustered data
    
    # Visualize the clusters
    plot_clusters(clustered_data)
except Exception as e:
    print("Error:", e)
