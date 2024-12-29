from sklearn.decomposition import PCA
import numpy as np

def apply_pca(data, n_components=2):
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca.explained_variance_ratio_

 
data = np.random.rand(100, 5)  # 100 samples, 5 features
transformed_data, variance_ratio = apply_pca(data, 2)
print("Transformed Data:", transformed_data)
print("Explained Variance Ratio:", variance_ratio)
