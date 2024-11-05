#
# Template for Task 2: Logistic Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import libraries as needed 
# (No additional imports needed)
# --- end of task --- #

# -------------------------------------
# Load data 
data = np.loadtxt('diabetes.csv', delimiter=',')
[n, p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75 * n)
num_test = n - num_train
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]
sample_test = data[num_train:, 0:-1]
label_test = data[num_train:, -1]
# -------------------------------------

# Add a column of ones to include the bias term (intercept)
sample_train = np.hstack((np.ones((num_train, 1)), sample_train))
sample_test = np.hstack((np.ones((num_test, 1)), sample_test))

# --- Your Task --- #
# Pick a proper number of iterations 
num_iter = 1000
# Randomly initialize your w 
np.random.seed(0)  # For reproducibility
w = np.random.randn(sample_train.shape[1])
# --- end of task --- #

er_test = []

# --- Your Task --- #
# Implement the iterative learning algorithm for w
# At the end of each iteration, evaluate the updated w 
learning_rate = 0.01
m = sample_train.shape[0]

for iter in range(num_iter): 

    ## Update w
    # Compute the linear combination
    z = sample_train.dot(w)
    # Apply the sigmoid function
    h = 1 / (1 + np.exp(-z))
    # Compute the gradient
    gradient = (1 / m) * sample_train.T.dot(h - label_train)
    # Update weights
    w = w - learning_rate * gradient

    ## Evaluate testing error of the updated w 
    # Compute predictions on the test set
    z_test = sample_test.dot(w)
    h_test = 1 / (1 + np.exp(-z_test))
    # Convert probabilities to class labels (0 or 1)
    predictions_test = (h_test >= 0.5).astype(int)
    # Compute classification error
    er = np.mean(predictions_test != label_test)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Classification Error vs. Number of Iterations')
plt.grid(True)
plt.show()
