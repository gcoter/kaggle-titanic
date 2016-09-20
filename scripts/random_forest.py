"""
Random Forest

Based on this tutorial : https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests
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

# Convert to numpy array
""" [PassengerId, Survived, Pclass, SibSp, Parch, FareClass, Gender, AgeFill, Alpha] """
train_data_np = train_data.values

""" [PassengerId, Pclass, SibSp, Parch, FareClass, Gender, AgeFill, Alpha] """
test_data_np = test_data.values

# === MODEL ===
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
""" 
X : [Alpha]
Y : Survived
"""

forest = forest.fit(train_data_np[0::,-1].reshape(-1, 1),train_data_np[0::,1])

# Take the same decision trees and run it on the test data
predictions = forest.predict(test_data_np[0::,-1].reshape(-1, 1))

# === Generate submission file ===
utils.generate_submission_file(test_data, predictions, results_path + 'random_forest.csv')