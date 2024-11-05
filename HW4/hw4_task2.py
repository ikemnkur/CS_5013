import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error # couldnt find a replacement
# --- end of task --- #

# Load a data set for regression
# In array "data", each row represents a community 
# Each column represents an attribute of community 
# Last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Always use last 25% data for testing 
num_test = int(0.25 * n)
sample_test = data[n - num_test:, 0:-1]
label_test = data[n - num_test:, -1]

# --- Your Task --- #
# Now, pick the percentage of data used for training 
# Remember we should be able to observe overfitting with this pick 
# Note: maximum percentage is 0.75 
per = 0.5  # Using 50% of the available data for training
num_train = int(n * per)
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]
# --- end of task --- #

# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 8 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
for alpha in alpha_vec: 

    # Pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # Now train your model using (sample_train, label_train)
    model.fit(sample_train, label_train)
    
    # Now evaluate your training error (MSE) and store it in "er_train"
    pred_train = model.predict(sample_train)
    er_train = mean_squared_error(label_train, pred_train)
    er_train_alpha.append(er_train)
    
    # Now evaluate your testing error (MSE) and store it in "er_test"
    pred_test = model.predict(sample_test)
    er_test = mean_squared_error(label_test, pred_test)
    er_test_alpha.append(er_test)
    # --- end of task --- #
    
plt.plot(alpha_vec, er_train_alpha, label='Training Error')
plt.plot(alpha_vec, er_test_alpha, label='Testing Error')
plt.xlabel('Hyper-Parameter Alpha')
plt.ylabel('Prediction Error (MSE)')
plt.legend()
plt.xscale('log')  # Use logarithmic scale for alpha
plt.show()
