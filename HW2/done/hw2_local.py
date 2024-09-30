import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('CreditCard.csv')

# Data Preprocessing
# Encode categorical variables
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})

# Handle the missing values by dropping the rows with missing data
data = data.dropna()

# Extract features and target variable
X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values.astype(float)
y = data['CreditApprove'].values.astype(float)

# Initialize parameters
n_samples = X.shape[0]
w = np.array([-1, -1, -1, -1, -1, -1])  # Initial weight vector
error_history = []

def compute_error(w, X, y):
    f = np.dot(X, w)
    error = np.mean((f - y) ** 2)
    return error

# Compute initial error
current_error = compute_error(w, X, y)
error_history.append(current_error)
converged = False
round_number = 0

# hill climbing algorithm
print("Starting the hill climbing algorithm.")

while not converged:
    round_number += 1
    print(f"Round {round_number}: Current error = {current_error}")
    neighbors = []
    errors = []
    
    # Generate all adjacent solutions
    for i in range(len(w)):
        w_neighbor = w.copy()
        w_neighbor[i] = -w_neighbor[i]  # Flip the sign
        neighbors.append(w_neighbor)
        neighbor_error = compute_error(w_neighbor, X, y)
        errors.append(neighbor_error)
        print(f"Neighbor {i}: w = {w_neighbor}, error = {neighbor_error}")
    
    # Find the best neighbor
    min_error = min(errors)
    min_index = errors.index(min_error)
    best_neighbor = neighbors[min_index]
    
    # Check if the best neighbor is better than the current
    if min_error < current_error:
        w = best_neighbor
        current_error = min_error
        error_history.append(current_error)
        print(f"Updated w to {w} with error {current_error}")
    else:
        converged = True
        print("No better neighbor found. The algorithm has converged.")

lenght = 6
width = 10

# Plotting er(w) versus each round of search
plt.figure(figsize=(width, lenght))
plt.plot(range(len(error_history)), error_history, marker='o')
plt.title('Error vs. Round of Search')
plt.xlabel('Nth Round of Search')
plt.ylabel('Error: er(w)')
plt.grid(True)
plt.show()

# Present the optimal w vector and er(w)
print("Optimal weight vector w:", w)
print("Minimum error er(w):", current_error)
