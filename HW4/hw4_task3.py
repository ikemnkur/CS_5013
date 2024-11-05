import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error # it say is depcrated but it seem to still work
from sklearn.utils import shuffle
# --- end of task --- #

# Load a data set for regression
# In array "data", each row represents a community
# Each column represents an attribute of the community
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
per = 0.6  # Using 60% of the available data for training
num_train = int(n * per)
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]
# --- end of task --- #

# --- Your Task --- #
# We will use a regression model called Ridge.
# This model has a hyper-parameter alpha. Larger alpha means a simpler model.
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values
# Suggestion: the first value should be very small and the last should be large
alpha_vec = [1e-4, 1e-2, 0.1, 1, 10]
# --- end of task --- #

er_valid_alpha = []

for alpha in alpha_vec: 

    # Pick ridge model, set its hyperparameter
    model = Ridge(alpha=alpha)
    
    # --- Your Task --- #
    # Implement k-fold cross-validation
    # on the training set to get the validation error for each candidate alpha value
    # Store it in "er_valid"
    
    # Shuffle training data
    sample_train_shuffled, label_train_shuffled = shuffle(sample_train, label_train, random_state=42)
    
    k = 5  # Number of folds
    num_train_shuffled = len(label_train_shuffled)
    fold_size = num_train_shuffled // k
    er_valid_folds = []
    
    for fold in range(k):
        # Define Validatiion indices
        val_start = fold * fold_size
        if fold == k - 1:
            val_end = num_train_shuffled  # Include all data in the last fold
        else:
            val_end = val_start + fold_size
        
        # Split validation set
        X_val = sample_train_shuffled[val_start:val_end]

        y_val = label_train_shuffled[val_start:val_end]
        
        # Split training set
        X_train_cv = np.concatenate((sample_train_shuffled[:val_start], sample_train_shuffled[val_end:]), axis=0)
        y_train_cv = np.concatenate((label_train_shuffled[:val_start], label_train_shuffled[val_end:]), axis=0)
        
        # Train model on k-1 folds
        model.fit(X_train_cv, y_train_cv)
        
        # Predict on validation fold
        y_pred_val = model.predict(X_val)
        
        # Compute validation error (MSE)
        er_fold = mean_squared_error(y_val, y_pred_val)
        er_valid_folds.append(er_fold)
        
    # Compute average validation error for this alpha
    er_valid = np.mean(er_valid_folds)
    er_valid_alpha.append(er_valid)
    # --- end of task --- #

# Now you should have obtained a validation error for each alpha value
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error
# Set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)]

# Now retrain your model on the entire training set using alpha_opt
# Then evaluate your model on the testing set
model = Ridge(alpha=alpha_opt)

# Train the model on the entire training set
model.fit(sample_train, label_train)

# Evaluate training error (MSE)
er_train = mean_squared_error(label_train, model.predict(sample_train))

# Evaluate testing error (MSE)
er_test = mean_squared_error(label_test, model.predict(sample_test))

# Print the optimal alpha and corresponding errors
print("Optimal Alpha :", alpha_opt)
print("Training Error with Alpha Opt.:", er_train)
print("Testing Error with Alpha_optimal:", er_test)

print(f"\nOptimal Alpha: {alpha_opt}")
print(f"Training Error (MSE): {er_train:.5f}")
print(f"Testing Error (MSE): {er_test:.5f}")

# Create Table 1 to display the validation errors for each alpha
print("Table 1:")
print("Hyper-Parameter Alpha\tValidation Error (MSE)")
for alpha, mse in zip(alpha_vec, er_valid_alpha):
    print(f"{alpha}\t\t\t{mse:.5f}")

plt.figure(figsize=(8,6))
plt.plot(alpha_vec, er_valid_alpha, marker='o', linestyle='-', color='b')
plt.xscale('log')  # Logarithmic scale for alpha
plt.xlabel('Hyper-Parameter Alpha')
plt.ylabel('Validation Error (MSE)')
plt.title('k-Fold Cross-Validation Error vs. Alpha')
plt.grid(True)
plt.show()