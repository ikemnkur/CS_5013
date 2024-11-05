import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# --- end of task --- #

# Load a data set for classification 
# In array "data", each row represents a patient 
# Each column represents an attribute of patients 
# Last column is the binary label: 1 means the patient has diabetes, 0 means otherwise
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Always use last 25% data for testing 
num_test = int(0.25 * n)
sample_test = data[n - num_test:, 0:-1]
label_test = data[n - num_test:, -1]

# --- Your Task --- #
# Now, vary the percentage of data used for training 
# Pick 8 values for array "num_train_per" e.g., 0.5 means using 50% of the available data for training 
# You should aim to observe overfitting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
num_train_per = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.65, 0.75]
# --- end of task --- #

er_train_per = []
er_test_per = []

# for loop for training the model
for per in num_train_per: 

    # Create training data and label 
    num_training = int(n * per)
    sample_training = data[0:num_training, 0:-1]
    label_training = data[0:num_training, -1]
    
    # We will use logistic regression model 
    model = LogisticRegression(max_iter=1000)
    
    # --- Your Task --- #
    # Now, train your model using training data 
    # (sample_train, label_train)
    model.fit(sample_training, label_training)
    
    # Now, evaluate training error (not MSE) of your model 
    # Store it in "er_train"
    pred_training = model.predict(sample_training)
    er_training = 1 - accuracy_score(label_training, pred_training)
    er_train_per.append(er_training)
    
    # Now, evaluate testing error (not MSE) of your model 
    # Store it in "er_test"
    pred_test = model.predict(sample_test)
    er_test = 1 - accuracy_score(label_test, pred_test)
    er_test_per.append(er_test)
    # --- end of task --- #
    
# Plot Errors
plt.figure(1)    
plt.plot(num_train_per, er_train_per, label='Training Error')
plt.plot(num_train_per, er_test_per, label='Testing Error')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Error')
plt.legend()
plt.show()
