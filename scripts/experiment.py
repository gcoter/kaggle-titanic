"""
Experiment
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import constants
import utils

data_folder = constants.data_folder
results_path = constants.results_path

# === Read data ===
train_data = pd.read_csv(data_folder + "train.csv", header=0)
test_data = pd.read_csv(data_folder + "test.csv", header=0)

# === Prepare data ===	
train_data = utils.prepare_dataset(train_data)
test_data = utils.prepare_dataset(test_data)

print('=== TRAIN DATA ===')
train_data.info()
print()
print(train_data.head(3))
print()
print('=== TEST DATA ===')
test_data.info()
print()
print(test_data.head(3))
print()

# Convert to numpy array
train_data_np = train_data.values
test_data_np = test_data.values

# === Feature selection ===
train_X = train_data_np[0::,2::]
train_Y = train_data_np[0::,1]

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=100, max_features=None)
clf = clf.fit(train_X,train_Y)

# print(clf.feature_importances_)
selection = SelectFromModel(clf, prefit=True)

train_X_new = selection.transform(train_X)
selected_indices = selection.get_support(indices=True)

print('Features Importances :',clf.feature_importances_)
print('Selected Indices :',selected_indices)

"""
# === MODEL ===
model = RandomForestClassifier(n_estimators=10, max_features=None)
model = model.fit(train_X_new,train_Y)

# Take the same decision trees and run it on the test data
test_X = test_data_np[0::,1::]
test_X_new = test_X[:,selected_indices]
predictions = model.predict(test_X_new)

# === Generate submission file ===
utils.generate_submission_file(test_data, predictions, results_path + 'experiment.csv')
"""