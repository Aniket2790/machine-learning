import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def load_heart_disease_data(csv_file):
   
    data = pd.read_csv(csv_file)
    return data

def construct_bayesian_network(data):
    
    # Define the structure of the Bayesian Network
    model = BayesianNetwork([
        ('Age', 'HeartDisease'),
        ('Sex', 'HeartDisease'),
        ('ChestPain', 'HeartDisease'),
        ('RestingBP', 'HeartDisease'),
        ('Cholesterol', 'HeartDisease'),
        ('HeartDisease', 'ExerciseInducedAngina')
    ])

    # Fit the model using Maximum Likelihood Estimation
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

def diagnose_heart_disease(model, query):
    
    infer = VariableElimination(model)
    result = infer.query(variables=['HeartDisease'], evidence=query)
    return result

def main():
    # Load the Heart Disease dataset
    csv_file = "heart_disease.csv"  # Replace with the actual path to the dataset
    data = load_heart_disease_data(csv_file)

    # Preprocess the data (example: encode categorical variables)
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'M' else 0)  # Convert Sex to binary
    data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert target to binary

    # Construct the Bayesian Network
    model = construct_bayesian_network(data)

    # Example query to diagnose heart disease
    query = {
        'Age': 45,
        'Sex': 1,  # 1 for Male
        'ChestPain': 3,  # Numeric encoding of chest pain type
        'RestingBP': 130,
        'Cholesterol': 250
    }

    # Perform inference
    result = diagnose_heart_disease(model, query)
    print("Diagnosis Result:")
    print(result)

if __name__ == "__main__":
    main()
