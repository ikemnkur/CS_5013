import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle


# --- end of task --- #

# Load an imbalanced data set 
# There are 50 positive class instances 
# There are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)

[num, p] = np.shape(data)

# To shuffle up the data to make sure that the training data is a accurate represnatation of the distrubution of classes
data_shuffled, labels_shuffled = shuffle(data[:, :-1], data[:, -1], random_state=42)
num_test = int(0.25 * num)
sample_test = data_shuffled[-num_test:]
label_test = labels_shuffled[-num_test:]
sample_train = data_shuffled[:-num_test]
label_train = labels_shuffled[:-num_test]

# Always use last 25% data for testing 
num_test = int(0.25 * num)
sample_test = data[num - num_test:, 0:-1]
label_test = data[num - num_test:, -1]

# Vary the percentage of data for training
num_train_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per: 

    # Create training data and label
    num_train = int(num * per)
    sample_train = data[0:num_train, 0:-1]
    label_train = data[0:num_train, -1]

    model = LogisticRegression(max_iter=1000)

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    model.fit(sample_train, label_train)
    
    # Evaluate model testing accuracy and store it in "acc_base"
    pred_test_base = model.predict(sample_test)
    acc_base = accuracy_score(label_test, pred_test_base)
    acc_base_per.append(acc_base)
    
    # Evaluate model testing AUC score and store it in "auc_base"
    prob_test_base = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, prob_test_base)
    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    
    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 
    
    # Implement logistic regression with class_weight='balanced'
    model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_balanced.fit(sample_train, label_train)
    
    # Evaluate model testing accuracy and store it in "acc_yours"
    pred_test_balanced = model_balanced.predict(sample_test)
    acc_yours = accuracy_score(label_test, pred_test_balanced)
    acc_yours_per.append(acc_yours)
    
    # Evaluate model testing AUC score and store it in "auc_yours"
    prob_test_balanced = model_balanced.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, prob_test_balanced)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #
    
pred_train_base = model.predict(sample_train)
print("Baseline Training Predictions:", np.bincount(pred_train_base.astype(int)))

pred_train_balanced = model_balanced.predict(sample_train)
print("Balanced Training Predictions:", np.bincount(pred_train_balanced.astype(int)))


print("Training set class distribution:", np.bincount(label_train.astype(int)))
print("Testing set class distribution:", np.bincount(label_test.astype(int)))


plt.figure(1)    
plt.plot(num_train_per, acc_base_per, label='Base Accuratcy')
plt.plot(num_train_per, acc_yours_per, label='My Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title('Model Accuracy vs Training Data Size')
plt.show()

plt.figure(2)
plt.plot(num_train_per, auc_base_per, label='Base AUC Score')
plt.plot(num_train_per, auc_yours_per, label='My AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.title('Model AUC Score vs Training Data Size')
plt.show()

# The Bonus Point Part: Impact of Hyper-Parameter on AUC Score
# Define different class_weight ratios for the positive class
class_weight_values = [{0:1, 1:wieght} for wieght in [1, 2, 5, 10, 20, 50]]

auc_hyperparameter = []

# Use the largest training size for the demonstration
per = num_train_per[-1]  # Use 80% of the training data
num_train = int(num * per)
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]

for cw in class_weight_values:
    model_custom = LogisticRegression(class_weight=cw, max_iter=1000)
    model_custom.fit(sample_train, label_train)
    
    prob_test_custom = model_custom.predict_proba(sample_test)[:, 1]
    auc = roc_auc_score(label_test, prob_test_custom)
    auc_hyperparameter.append(auc)

# Plotting the impact of the class_weight on the AUC score
weights = [cw[1] for cw in class_weight_values]

plt.figure(3)
plt.plot(weights, auc_hyperparameter, marker='o')
plt.xlabel('Class Weight for Positive Class')
plt.ylabel('Classification AUC Score')
plt.title('Impact of Class Weight on AUC Score')
plt.xscale('log')
plt.show()