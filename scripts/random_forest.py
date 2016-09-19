"""
Random Forest

Based on this tutorial : https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests
"""

from __future__ import print_function
import pandas as pd
import numpy as np

data_folder = "../data/"
results_path = "../results/"

# === Read train data ===
train_data = pd.read_csv(data_folder + "train.csv", header=0)

# === Prepare data ===
"""
ORIGINAL DATASET : PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

NEW DATASET : PassengerId,Pclass,Gender,AgeFill,FareClass
"""

num_Pclasses = len(np.unique(train_data['Pclass']))
num_fare_classes = 5

def prepare_dataset(dataframe):
	# Clean data
	dataframe.loc[dataframe['Fare'].isnull(),'Fare'] = dataframe['Fare'].mean()

	# Create different class for fare
	dataframe.loc[(dataframe['Fare'] >= 0) & (dataframe['Fare'] < 10),'FareClass'] = 0
	dataframe.loc[(dataframe['Fare'] >= 10) & (dataframe['Fare'] < 20),'FareClass'] = 1
	dataframe.loc[(dataframe['Fare'] >= 20) & (dataframe['Fare'] < 30),'FareClass'] = 2
	dataframe.loc[(dataframe['Fare'] >= 30) & (dataframe['Fare'] < 40),'FareClass'] = 3
	dataframe.loc[dataframe['Fare'] >= 40,'FareClass'] = 4

	# Conversions to int
	dataframe['PassengerId'] = dataframe['PassengerId'].astype(int)
	dataframe['FareClass'] = dataframe['FareClass'].astype(int)

	# Convert 'male' and 'female' to integers
	dataframe['Gender'] = dataframe['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

	# Fill the Age column
	dataframe['AgeFill'] = dataframe['Age']
	dataframe.loc[dataframe['AgeFill'].isnull(),'AgeFill'] = dataframe['AgeFill'].mean()

	# Drop Age
	dataframe = dataframe.drop(['Age'], axis=1)
	
	# Drop Fare
	dataframe = dataframe.drop(['Fare'], axis=1)

	# Drop unused columns (dtype=object)
	dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

	return dataframe
	
train_data = prepare_dataset(train_data)

# Convert to numpy array
""" [PassengerId, Survived, Pclass, SibSp, Parch, FareClass, Gender, AgeFill] """
train_data_np = train_data.values

# === Read test data ===
# Clean data
test_data = pd.read_csv(data_folder + "test.csv", header=0)
test_data = prepare_dataset(test_data)

# Convert to numpy array
""" [PassengerId, Pclass, SibSp, Parch, FareClass, Gender, AgeFill] """
test_data_np = test_data.values

# === MODEL ===
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
""" 
X : [Pclass, SibSp, Parch, FareClass, Gender, AgeFill]
Y : Survived
"""

forest = forest.fit(train_data_np[0::,2::],train_data_np[0::,1])

# Take the same decision trees and run it on the test data
prediction = forest.predict(test_data_np[0::,1::])

# === Generate submission file ===
submission = pd.DataFrame(columns=('PassengerId','Survived'))
submission['PassengerId'] = submission['PassengerId'].astype(int)
submission['Survived'] = submission['Survived'].astype(int)

for i in range(len(test_data)):
	passengerId = int(test_data['PassengerId'][i])
	survived = int(prediction[i]) # <-- make prediction here
	submission.loc[i] = [passengerId,survived] # Add row
	
submission.to_csv(results_path + 'random_forest.csv', index=False)