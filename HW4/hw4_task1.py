import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
# Now, vary the percentage of data used for training 
# Pick 8 values for array "num_train_per"
# You should aim to observe overfitting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
# num_train_per = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.65, 0.75]
num_train_per = [0.15, 0.25, 0.35, 0.45, 0.5, 0.54, 0.65, 0.75, 0.85, 0.9, 0.95]
# --- end of task --- #

er_train_per = []
er_test_per = []

for per in num_train_per: 
    # Create training data and label 
    num_train = int(n * per)
    sample_train = data[0:num_train, 0:-1]
    label_train = data[0:num_train, -1]
    
    # We will use linear regression model 
    model = LinearRegression()
    
    # --- Your Task --- #
    # Now, training your model using training data 
    # (sample_train, label_train)
    model.fit(sample_train, label_train)
    
    # Now, evaluate training error (MSE) of your model 
    # Store it in "er_train"
    pred_train = model.predict(sample_train)
    er_train = mean_squared_error(label_train, pred_train)
    er_train_per.append(er_train)
    
    # Now, evaluate testing error (MSE) of your model 
    # Store it in "er_test"
    pred_test = model.predict(sample_test)
    er_test = mean_squared_error(label_test, pred_test)
    er_test_per.append(er_test)
    # --- end of task --- #
        
plt.plot(num_train_per, er_train_per, label='Training Error')
plt.plot(num_train_per, er_test_per, label='Testing Error')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Prediction Error (MSE)')
plt.legend()
plt.show()
