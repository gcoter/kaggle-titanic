"""
ORIGINAL DATASET : PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

NEW DATASET : PassengerId,Pclass,Gender,AgeFill,FareClass
"""

import pandas as pd

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
	
def generate_submission_file(test_data,predictions,file_path):
	submission = pd.DataFrame(columns=('PassengerId','Survived'))
	submission['PassengerId'] = submission['PassengerId'].astype(int)
	submission['Survived'] = submission['Survived'].astype(int)

	for i in range(len(test_data)):
		passengerId = int(test_data['PassengerId'][i])
		survived = int(predictions[i]) # <-- make prediction here
		submission.loc[i] = [passengerId,survived] # Add row
		
	submission.to_csv(file_path, index=False)