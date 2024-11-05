# Template for Task 1: Linear Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import libraries as needed 
# (No additional imports needed)
# --- end of task --- #

# -------------------------------------
# Load data 
data = np.loadtxt('crimerate.csv', delimiter=',')
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
    predictions = sample_train.dot(w)
    errors = predictions - label_train
    gradient = (1 / m) * sample_train.T.dot(errors)
    w = w - learning_rate * gradient

    ## Evaluate testing error of the updated w 
    # We should measure mean-square-error here
    predictions_test = sample_test.dot(w)
    er = np.mean((predictions_test - label_test) ** 2)
    er_test.append(er)
# --- end of task --- #

plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Number of Iterations')
plt.grid(True)
plt.show()
