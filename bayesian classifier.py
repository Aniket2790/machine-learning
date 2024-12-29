from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(documents, labels):
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    return vectorizer, X, labels

def train_naive_bayes(X_train, y_train):
     
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def classify_documents(model, vectorizer, new_documents):
    
    X_new = vectorizer.transform(new_documents)
    predictions = model.predict(X_new)
    return predictions

def main():
    # Example documents and their corresponding labels
    documents = [
        "I love programming in Python",
        "Python is a great language for data science",
        "Machine learning is fascinating",
        "I dislike writing documentation",
        "Data analysis is interesting",
        "I hate bugs in my code",
        "Programming is both challenging and rewarding",
        "Debugging is frustrating but necessary"
    ]
    labels = ["Positive", "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative"]

    # Preprocess the data
    vectorizer, X, y = preprocess_data(documents, labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the Naïve Bayes classifier
    model = train_naive_bayes(X_train, y_train)

    # Test the model and evaluate its performance
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Classify new documents
    new_documents = [
        "I enjoy solving problems with Python",
        "Debugging can be very annoying"
    ]
    predictions = classify_documents(model, vectorizer, new_documents)
    print("\nNew Document Classifications:")
    for doc, label in zip(new_documents, predictions):
        print(f"'{doc}' -> {label}")

if __name__ == "__main__":
    main()
