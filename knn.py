from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
def load_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target  # Features and labels
    return X, y, iris.target_names

# Split the dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train the k-NN classifier
def train_knn(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Evaluate the model and print correct and wrong predictions
def evaluate_knn(knn, X_test, y_test, target_names):
    predictions = knn.predict(X_test)
    
    correct = []
    wrong = []
    
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
        if pred == true:
            correct.append((i, pred, true))
        else:
            wrong.append((i, pred, true))
    
    print("Correct Predictions:")
    for idx, pred, true in correct:
        print(f"Index: {idx}, Predicted: {target_names[pred]}, Actual: {target_names[true]}")
    
    print("\nWrong Predictions:")
    for idx, pred, true in wrong:
        print(f"Index: {idx}, Predicted: {target_names[pred]}, Actual: {target_names[true]}")
    
    # Print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.2f}")

# Main function to run the k-NN classification
def main():
    # Load data
    X, y, target_names = load_iris_data()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train k-NN classifier
    knn = train_knn(X_train, y_train, n_neighbors=3)
    
    # Evaluate the classifier
    evaluate_knn(knn, X_test, y_test, target_names)

# Run the program
if __name__ == "__main__":
    main()
