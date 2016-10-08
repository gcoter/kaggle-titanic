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
	
	# Convert to classes
	dataframe = convert_to_classes(dataframe,'FareFill')
	dataframe = convert_to_classes(dataframe,'AgeFill')
	dataframe = convert_to_classes(dataframe,'FamilySize')
	
	# Create title feature from name
	dataframe['Title'] = dataframe['Name'].str.extract("(.*\.)", expand=False).str.split(",", expand=False).str.get(1).str.strip()
	
	""" 
	>>> train_data['Title'].value_counts()
	Mr.                          517
	Miss.                        182
	Mrs.                         124
	Master.                       40  <-- Rare
	Dr.                            7  <-- Rare
	Rev.                           6  <-- Rare
	Mlle.                          2  <-- Could be 'Miss.'
	Major.                         2  <-- Rare
	Col.                           2  <-- Rare
	Jonkheer.                      1  <-- Rare
	Sir.                           1  <-- Rare
	Mrs. Martin (Elizabeth L.      1  <-- Should be 'Mrs.'
	Don.                           1  <-- Rare
	Capt.                          1  <-- Rare
	the Countess.                  1  <-- Rare
	Ms.                            1  <-- Could be 'Mrs.'
	Mme.                           1  <-- Could be 'Mrs.'
	Lady.                          1  <-- Could be 'Miss.'
	"""
	
	frequent_titles = ['Mr.','Miss.','Mrs.']
	possible_titles = frequent_titles + ['Rare']
	# rare_titles = ['Master.','Dr.','Rev.','Major.','Col.','Jonkheer.','Sir.','Don.','Capt.','the Countess.']
	
	dataframe.loc[dataframe['Title'] == 'Mlle.', 'Title'] = 'Miss.'
	dataframe.loc[dataframe['Title'] == 'Ms.', 'Title'] = 'Mrs.'
	dataframe.loc[dataframe['Title'] == 'Mme.', 'Title'] = 'Mrs.'
	dataframe.loc[dataframe['Title'] == 'Lady.', 'Title'] = 'Miss.'
	
	dataframe.loc[~dataframe['Title'].isin(frequent_titles), 'Title'] = 'Rare'
	
	# Convert Title to numerical classes
	for i in range(len(possible_titles)):
		dataframe.loc[dataframe['Title'] == possible_titles[i], 'TitleClass'] = i
	
	# Conversions to int
	dataframe['PassengerId'] = dataframe['PassengerId'].astype(int)
	dataframe['FareFillClass'] = dataframe['FareFillClass'].astype(int)
	dataframe['AgeFillClass'] = dataframe['AgeFillClass'].astype(int)
	dataframe['FamilySizeClass'] = dataframe['FamilySizeClass'].astype(int)
	dataframe['TitleClass'] = dataframe['TitleClass'].astype(int)
	
	# FEATURES : PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,Survived||FareFill,AgeFill,Gender,FamilySize,FareClass,AgeFillClass,FamilySizeClass,Title,TitleClass
	
	# Drop unused classes
	dataframe = dataframe.drop(['Age'], axis=1)
	dataframe = dataframe.drop(['AgeFill'], axis=1)
	dataframe = dataframe.drop(['Fare'], axis=1)
	dataframe = dataframe.drop(['FareFill'], axis=1)
	dataframe = dataframe.drop(['FamilySize'], axis=1)
	dataframe = dataframe.drop(['SibSp'], axis=1)
	dataframe = dataframe.drop(['Parch'], axis=1)

	# Drop unused columns (dtype=object)
	dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1)
	
	# FEATURES : PassengerId,(Survived),Pclass||Gender,FareFillClass,AgeFillClass,FamilySizeClass,TitleClass
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