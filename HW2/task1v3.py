import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # De/encode categorical variables
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
    data['CarOwner'] = data['CarOwner'].map({'dependentData': 1, 'N': 0})
    data['PropertyOwner'] = data['PropertyOwner'].map({'dependentData': 1, 'N': 0})
    
    # This will drop the missing values 
    data = data.dropna()
    
    # Separate features and target
    independentData = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']]
    dependentData = data['CreditApprove']
    
    return independentData.values, dependentData.values

def calculate_error(independentData, dependentData, w):
    predictions = np.dot(independentData, w)
    error = np.mean((predictions - dependentData) ** 2)
    return error

def hill_climbing(independentData, dependentData, max_iterations=5000):
    # number of elements in array
    n_elements = independentData.shape[1]
    # Initialize w as [-1, -1, -1, -1, -1, -1]
    w = np.array([-1] * n_elements)  
    error_rates = []
    
    for iteration in range(max_iterations):
        current_error = calculate_error(independentData, dependentData, w)
        error_rates.append(current_error)
        
        best_w = w.copy()
        best_error = current_error
        
        for i in range(n_elements):
            # Make a copy of the array
            w_new = w.copy()
            # Flip the sign of one element
            w_new[i] = -w_new[i]  
            new_error = calculate_error(independentData, dependentData, w_new)
            
            if new_error < best_error:
                best_w = w_new
                best_error = new_error
        
        if np.array_equal(w, best_w):
            break
        
        w = best_w
    
    return w, error_rates

# Main execution
file_path = 'CreditCard.csv'
independentData, dependentData = load_and_preprocess_data(file_path)

# Run hill climbing
optimal_w, error_rates = hill_climbing(independentData, dependentData)

# Plot error rate vs. iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(error_rates)), error_rates)
plt.title('Error Rate vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error Rate (MSE)')
plt.grid(True)
plt.savefig('error_rate_plot.png')
plt.close()

# Print results
print("Optimal w:", optimal_w)
print("Final error rate:", error_rates[-1])