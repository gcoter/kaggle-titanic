"""
A simple model based on Sex, Class and Fare.

Based on those tutorials :
* https://www.kaggle.com/c/titanic/details/getting-started-with-python
* https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii
"""

from __future__ import print_function
import pandas as pd
import numpy as np

data_folder = "../data/"
results_path = "../results/"

# === READ DATA ===
train_data = pd.read_csv(data_folder + "train.csv", header=0)

# === Filter by Sex, Pclass and Fare ===
num_Pclasses = len(np.unique(train_data['Pclass']))

# Clean data
train_data.loc[train_data['Fare'].isnull(),'Fare'] = train_data['Fare'].mean()

# Create different class for fare
# $0-9 => 0
# $10-19 => 1
# $20-29 => 2
# $30-39 => 3
# $40-inf => 4

num_fare_classes = 5
		
train_data.loc[(train_data['Fare'] >= 0) & (train_data['Fare'] < 10),'FareClass'] = 0
train_data.loc[(train_data['Fare'] >= 10) & (train_data['Fare'] < 20),'FareClass'] = 1
train_data.loc[(train_data['Fare'] >= 20) & (train_data['Fare'] < 30),'FareClass'] = 2
train_data.loc[(train_data['Fare'] >= 30) & (train_data['Fare'] < 40),'FareClass'] = 3
train_data.loc[train_data['Fare'] >= 40,'FareClass'] = 4

# Conversions to int
train_data['PassengerId'] = train_data['PassengerId'].astype(int)
train_data['FareClass'] = train_data['FareClass'].astype(int)

survival_table = np.zeros((2,num_Pclasses,num_fare_classes))

for pclass in range(num_Pclasses):
	for fare_class in range(num_fare_classes):
		male_stats = train_data[(train_data['Sex'] == 'male') & (train_data['Pclass'] == pclass) & (train_data['FareClass'] == fare_class)]['Survived'].as_matrix()
		female_stats = train_data[(train_data['Sex'] == 'female') & (train_data['Pclass'] == pclass) & (train_data['FareClass'] == fare_class)]['Survived'].as_matrix()
		if len(male_stats) == 0:
			survival_table[0,pclass,fare_class] = 0
		else:
			survival_table[0,pclass,fare_class] = np.mean(male_stats)
		if len(female_stats) == 0:
			survival_table[1,pclass,fare_class] = 0
		else:
			survival_table[1,pclass,fare_class] = np.mean(female_stats)

survival_table[survival_table >= 0.5] = 1
survival_table[survival_table < 0.5] = 0

survival_table = survival_table.astype(int)

print(survival_table)

test_data = pd.read_csv(data_folder + "test.csv", header=0)

# Clean data
test_data.loc[test_data['Fare'].isnull(),'Fare'] = test_data['Fare'].mean()

# Create different class for fare
test_data.loc[(test_data['Fare'] >= 0) & (test_data['Fare'] < 10),'FareClass'] = 0
test_data.loc[(test_data['Fare'] >= 10) & (test_data['Fare'] < 20),'FareClass'] = 1
test_data.loc[(test_data['Fare'] >= 20) & (test_data['Fare'] < 30),'FareClass'] = 2
test_data.loc[(test_data['Fare'] >= 30) & (test_data['Fare'] < 40),'FareClass'] = 3
test_data.loc[test_data['Fare'] >= 40,'FareClass'] = 4

# Conversions to int
test_data['PassengerId'] = test_data['PassengerId'].astype(int)
test_data['FareClass'] = test_data['FareClass'].astype(int)

submission = pd.DataFrame(columns=('PassengerId','Survived'))
submission['PassengerId'] = submission['PassengerId'].astype(int)
submission['Survived'] = submission['Survived'].astype(int)

for i in range(len(test_data)):
	sex_index = 0
	if test_data['Sex'][i] == 'female':
		sex_index = 1
	
	passengerId = test_data['PassengerId'][i]
	survived = survival_table[sex_index,test_data['Pclass'][i]-1,test_data['FareClass'][i]]
	submission.loc[i] = [passengerId,survived] # Add row
	
submission.to_csv(results_path + 'simple_model.csv', index=False)