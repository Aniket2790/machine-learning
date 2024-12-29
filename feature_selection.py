from sklearn.ensemble import RandomForestClassifier
import numpy as np

def feature_importance_random_forest(data, labels):
   
    model = RandomForestClassifier()
    model.fit(data, labels)
    return model.feature_importances_

 
data = np.random.rand(100, 5)  # 100 samples, 5 features
labels = np.random.randint(0, 2, 100)  # Binary labels
importances = feature_importance_random_forest(data, labels)
print("Feature Importances:", importances)
