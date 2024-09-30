import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Encode categorical variables
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
    data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
    data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})
    
    # Handle missing values (if any)
    data = data.dropna()
    
    # Separate features and target
    X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']]
    y = data['CreditApprove']
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y.values

def calculate_error_rate(X, y, w):
    predictions = np.sign(np.dot(X, w))
    error_rate = np.mean(predictions != y)
    return error_rate

def hill_climbing(X, y, max_iterations=5000):
    n_features = X.shape[1]
    w = np.random.randn(n_features)  # Random initial weights
    error_rates = []
    
    for iteration in range(max_iterations):
        current_error = calculate_error_rate(X, y, w)
        error_rates.append(current_error)
        
        best_w = w.copy()
        best_error = current_error
        
        for i in range(n_features):
            for step in [-0.1, 0.1]:  # Smaller step size
                w_new = w.copy()
                w_new[i] += step
                new_error = calculate_error_rate(X, y, w_new)
                
                if new_error < best_error:
                    best_w = w_new
                    best_error = new_error
        
        if np.array_equal(w, best_w):
            break
        
        w = best_w
    
    return w, error_rates

# Main execution
file_path = 'CreditCard.csv'
X, y = load_and_preprocess_data(file_path)

# Run hill climbing
optimal_w, error_rates = hill_climbing(X, y)

# Plot error rate vs. iterations and save the plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(error_rates)), error_rates)
plt.title('Error Rate vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.grid(True)
plt.savefig('error_rate_plot_improved.png')
print("Plot saved as 'error_rate_plot_improved.png'")

# Print results
print("Optimal w:", optimal_w)
print("Final error rate:", error_rates[-1])