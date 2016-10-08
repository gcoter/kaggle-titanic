"""
ORIGINAL DATASET : PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,Survived

NEW DATASET : PassengerId,Survived,Pclass,FamilySizeClass,FareClass,Gender,AgeFillClass
"""

from __future__ import print_function
import pandas as pd

def convert_to_classes(dataframe,feature):
	quantile1 = dataframe.quantile(.25)[feature]
	median = dataframe.quantile(.5)[feature]
	quantile2 = dataframe.quantile(.75)[feature]
	
	dataframe.loc[(dataframe[feature] < quantile1),feature+'Class'] = 0
	dataframe.loc[(dataframe[feature] >= quantile1) & (dataframe[feature] < median),feature+'Class'] = 1
	dataframe.loc[(dataframe[feature] >= median) & (dataframe[feature] < quantile2),feature+'Class'] = 2
	dataframe.loc[(dataframe[feature] >= quantile2),feature+'Class'] = 3
	
	return dataframe

def prepare_dataset(dataframe):
	# Fill data
	dataframe['FareFill'] = dataframe['Fare']
	dataframe.loc[dataframe['FareFill'].isnull(),'FareFill'] = dataframe['FareFill'].mean()
	dataframe['AgeFill'] = dataframe['Age']
	dataframe.loc[dataframe['AgeFill'].isnull(),'AgeFill'] = dataframe['AgeFill'].mean()
	
	# Convert 'male' and 'female' to integers
	dataframe['Gender'] = dataframe['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
	
	# Create FamilySize feature
	dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch']
	
	print(dataframe[dataframe['FamilySize'].isnull()])
	
	# Convert to classes
	dataframe = convert_to_classes(dataframe,'FareFill')
	dataframe = convert_to_classes(dataframe,'AgeFill')
	dataframe = convert_to_classes(dataframe,'FamilySize')
	
	# Conversions to int
	dataframe['PassengerId'] = dataframe['PassengerId'].astype(int)
	dataframe['FareFillClass'] = dataframe['FareFillClass'].astype(int)
	dataframe['AgeFillClass'] = dataframe['AgeFillClass'].astype(int)
	dataframe['FamilySizeClass'] = dataframe['FamilySizeClass'].astype(int)
	
	# FEATURES : PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,Survived||FareFill,AgeFill,Gender,FamilySize,FareClass,AgeFillClass,FamilySizeClass
	
	# Drop unused classes
	dataframe = dataframe.drop(['Age'], axis=1)
	dataframe = dataframe.drop(['AgeFill'], axis=1)
	dataframe = dataframe.drop(['Fare'], axis=1)
	dataframe = dataframe.drop(['FareFill'], axis=1)
	dataframe = dataframe.drop(['FamilySize'], axis=1)
	dataframe = dataframe.drop(['SibSp'], axis=1)
	dataframe = dataframe.drop(['Parch'], axis=1)

	# Drop unused columns (dtype=object)
	dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
	
	# FEATURES : PassengerId,Survived,Pclass||Gender,FareFillClass,AgeFillClass,FamilySizeClass

	return dataframe
	
def generate_submission_file(test_data,predictions,file_path):
	submission = pd.DataFrame(columns=('PassengerId','Survived'))
	submission['PassengerId'] = submission['PassengerId'].astype(int)
	submission['Survived'] = submission['Survived'].astype(int)

	for i in range(len(test_data)):
		passengerId = int(test_data['PassengerId'][i])
		survived = int(predictions[i]) # <-- make prediction here
		submission.loc[i] = [passengerId,survived] # Add row
		
	submission.to_csv(file_path, index=False)
	print("Saved to", file_path)