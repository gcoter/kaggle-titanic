import constants
import pandas as pd
import numpy as np
from six.moves.urllib.request import urlretrieve
import os.path
import logging

"""
This class contains method to manage data.

Its main goal is to return one training set, one validation set and one test set thanks to DataManager.get_datasets(files_paths).
"""
class DataManager(object):
	# === METHODS TO IMPLEMENT (if necessary) ===
	@staticmethod
	def construct_datasets(files_paths):
		"""
		This the main method of this class.
		
		It calls all other methods, when necessary, in order to return
		the training set, the validation set and the test set which will be used
		later to train models.
		
		This method should be edited if necessary.
		"""
		
		logging.info("Start constructing datasets")
		
		dataframes = DataManager.extract(files_paths)
		
		# If test data is in a separate file
		logging.info("Transforming train dataframe")
		train_dataframe = DataManager.transform_dataframe(dataframes["train.csv"]) # Be sure to put the right dataframe here
		logging.info("Transforming test dataframe")
		test_df = DataManager.transform_dataframe(dataframes["test.csv"]) # Be sure to put the right dataframe here
		logging.info("Splitting train dataframe into training and validation sets")
		train_df, valid_df = DataManager.split(train_dataframe,proportion=0.1,shuffle=True)
		
		# Separate X from Y
		train_df_X, train_df_Y = DataManager.get_X_and_Y(train_df)
		valid_df_X, valid_df_Y = DataManager.get_X_and_Y(valid_df)
		
		train_X, train_Y = train_df_X.values, train_df_Y.values
		valid_X, valid_Y = valid_df_X.values, valid_df_Y.values
		test_X = test_df.values
		
		DataManager.save_datasets(train_X, train_Y, valid_X, valid_Y, test_X)
		return train_X, train_Y, valid_X, valid_Y, test_X
	
	@staticmethod
	def extract(files_paths):
		logging.info("Extracting files")
		
		"""
		This function must return a list of Pandas DataFrames containing data extracted from file_paths.
		"""
		
		dataframes = {}
		
		for file_name in files_paths.keys():
			dataframes[file_name] = pd.read_csv(files_paths[file_name])
		
		return dataframes
	
	@staticmethod
	def filter(dataframe):
		logging.info("Filtering dataframe")
		
		""" 
		YOUR CODE HERE (if necessary)
		
		This function must remove unnecessary columns from dataframe and return it.
		"""
		return dataframe
		
	@staticmethod
	def clean(dataframe):
		logging.info("Cleaning dataframe")
		
		""" 
		This function must clear (and complete) data in dataframe and return it.
		"""
		
		dataframe['FareFill'] = dataframe['Fare']
		dataframe.loc[dataframe['FareFill'].isnull(),'FareFill'] = dataframe['FareFill'].mean()
		
		dataframe['AgeFill'] = dataframe['Age']
		dataframe.loc[dataframe['AgeFill'].isnull(),'AgeFill'] = dataframe['AgeFill'].mean()
		
		dataframe['EmbarkedFill'] = dataframe['Embarked']
		dataframe.loc[dataframe['EmbarkedFill'].isnull(),'EmbarkedFill'] = 'C'
		
		return dataframe
		
	@staticmethod
	def feature_engineering(dataframe):
		logging.info("Doing feature engineering on dataframe")
		
		"""
		This function must add new features.
		"""
		
		dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch']
		
		dataframe['Title'] = dataframe['Name'].str.extract("(.*\.)", expand=False).str.split(",", expand=False).str.get(1).str.strip()
		
		dataframe.loc[dataframe['Title'] == 'Mlle.', 'Title'] = 'Miss.'
		dataframe.loc[dataframe['Title'] == 'Ms.', 'Title'] = 'Mrs.'
		dataframe.loc[dataframe['Title'] == 'Mme.', 'Title'] = 'Mrs.'
		dataframe.loc[dataframe['Title'] == 'Lady.', 'Title'] = 'Miss.'
	
		dataframe.loc[~dataframe['Title'].isin(constants.FREQUENT_TITLES), 'Title'] = 'Rare'
		
		return dataframe
		
	@staticmethod
	def convert_to_classes(dataframe,feature):
		quantile1 = dataframe.quantile(.25)[feature]
		median = dataframe.quantile(.5)[feature]
		quantile2 = dataframe.quantile(.75)[feature]
		
		dataframe.loc[(dataframe[feature] < quantile1),feature+'Class'] = 0
		dataframe.loc[(dataframe[feature] >= quantile1) & (dataframe[feature] < median),feature+'Class'] = 1
		dataframe.loc[(dataframe[feature] >= median) & (dataframe[feature] < quantile2),feature+'Class'] = 2
		dataframe.loc[(dataframe[feature] >= quantile2),feature+'Class'] = 3
		
		return dataframe
		
	@staticmethod
	def numerize(dataframe):
		logging.info("Numerizing dataframe")
		
		"""
		This function must transform all non-numeric features (string, boolean, etc...) 
		into numeric features in dataframe and return it.
		"""
		
		dataframe['Gender'] = dataframe['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
		
		dataframe = DataManager.convert_to_classes(dataframe,'FareFill')
		dataframe = DataManager.convert_to_classes(dataframe,'AgeFill')
		dataframe = DataManager.convert_to_classes(dataframe,'FamilySize')
		
		for i in range(len(constants.POSSIBLE_TITLES)):
			dataframe.loc[dataframe['Title'] == constants.POSSIBLE_TITLES[i], 'TitleClass'] = i
			
		dataframe['EmbarkedClass'] = dataframe['EmbarkedFill'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
		
		# Conversions to int
		dataframe['PassengerId'] = dataframe['PassengerId'].astype(int)
		dataframe['FareFillClass'] = dataframe['FareFillClass'].astype(int)
		dataframe['AgeFillClass'] = dataframe['AgeFillClass'].astype(int)
		dataframe['FamilySizeClass'] = dataframe['FamilySizeClass'].astype(int)
		dataframe['TitleClass'] = dataframe['TitleClass'].astype(int)
		
		return dataframe
		
	@staticmethod
	def drop_unused(dataframe):
		""" This function must drop unused features """
		dataframe = dataframe.drop(['Age'], axis=1)
		dataframe = dataframe.drop(['Fare'], axis=1)
		dataframe = dataframe.drop(['FareFill'], axis=1)
		dataframe = dataframe.drop(['SibSp'], axis=1)
		dataframe = dataframe.drop(['Parch'], axis=1)
		dataframe = dataframe.drop(['Embarked'], axis=1)
		dataframe = dataframe.drop(['EmbarkedFill'], axis=1)
		
		# Drop unused columns (dtype=object)
		dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Title'], axis=1)
		
		return dataframe
	
	@staticmethod
	def get_X_and_Y(dataframe):
		logging.debug("Getting X and Y")
		
		"""
		This function must return X (input data) and Y (label for instance) as dataframes
		"""
		
		X = dataframe.drop("Survived", axis=1)
		Y = dataframe["Survived"]
		return X, Y
	
	# === OTHER METHODS ===
	@staticmethod
	def file_exists(filepath):
		""" Helper function to check whether a file exists. """
		res = os.path.isfile(filepath)
		if res:
			logging.debug(filepath + " found")
		else:
			logging.debug(filepath + " not found")
		return res

	@staticmethod
	def get_datasets(files_paths):
		""" Load or construct datasets and return them as numpy arrays. """
		
		# Check whether datasets backups already exist
		if DataManager.file_exists(constants.TRAIN_X_PATH) and DataManager.file_exists(constants.TRAIN_Y_PATH) and DataManager.file_exists(constants.VALID_X_PATH) and DataManager.file_exists(constants.VALID_Y_PATH) and DataManager.file_exists(constants.TEST_X_PATH):
			return DataManager.load_datasets()
		else:
			return DataManager.construct_datasets(files_paths)
	
	@staticmethod	
	def transform_dataframe(dataframe):
		logging.debug("Transforming dataframe")
	
		"""
		This method applies all required methods to return a usable dataframe.
		"""
		dataframe = DataManager.filter(dataframe)
		dataframe = DataManager.clean(dataframe)
		dataframe = DataManager.feature_engineering(dataframe)
		dataframe = DataManager.numerize(dataframe)
		dataframe = DataManager.drop_unused(dataframe)
		
		return dataframe
			
	@staticmethod
	def shuffle(dataframe):
		""" Randomly permutes elements along axis in dataframe. """
		logging.debug("Shuffling dataframe")
		dataframe.iloc[np.random.permutation(len(dataframe))]
		dataframe.reset_index(drop=True)
		return dataframe
		
	@staticmethod
	def split(dataframe,proportion=0.1,shuffle=True):
		""" 
		Divide dataframe into two dataframes (usually train_df and valid_df) and return them
		
		If shuffle is True, dataframe is first shuffled before being divided.
		"""
		logging.debug("Splitting dataframe")
		
		if shuffle:
			dataframe = DataManager.shuffle(dataframe)
		
		split_index = int(proportion * len(dataframe))
		dataframe_right = dataframe[split_index:]
		dataframe_left = dataframe[:split_index]
		
		logging.debug("split_index = " + str(split_index))
		logging.debug("dataframe_right : " + str(dataframe_right.shape))
		logging.debug("dataframe_left : " + str(dataframe_left.shape))
		
		return dataframe_right, dataframe_left
		
	@staticmethod
	def save_np_array(np_array,filepath):
		""" Save a numpy array to filepath. """
		return np.save(filepath, np_array)
	
	@staticmethod
	def load_np_array(filepath):
		""" Load and return a numpy array. """
		return np.load(filepath)
		
	@staticmethod
	def save_datasets(train_X, train_Y, valid_X, valid_Y, test_X):
		""" Save train_df, valid_df and test_df as numpy arrays. """
		logging.info("Saving training X to " + constants.TRAIN_X_PATH)
		DataManager.save_np_array(train_X, constants.TRAIN_X_PATH)
		logging.info("Saving training Y to " + constants.TRAIN_Y_PATH)
		DataManager.save_np_array(train_Y, constants.TRAIN_Y_PATH)
		
		logging.info("Saving validation X to " + constants.VALID_X_PATH)
		DataManager.save_np_array(valid_X, constants.VALID_X_PATH)
		logging.info("Saving validation Y to " + constants.VALID_Y_PATH)
		DataManager.save_np_array(valid_Y, constants.VALID_Y_PATH)
		
		logging.info("Saving test X to " + constants.TEST_X_PATH)
		DataManager.save_np_array(test_X, constants.TEST_X_PATH)
		
	@staticmethod
	def load_datasets():
		""" Save training, validation and test sets as numpy arrays. """
		logging.info("Loading training X from " + constants.TRAIN_X_PATH)
		train_X = DataManager.load_np_array(constants.TRAIN_X_PATH)
		logging.info("Loading training Y from " + constants.TRAIN_Y_PATH)
		train_Y = DataManager.load_np_array(constants.TRAIN_Y_PATH)
		
		logging.info("Loading validation X from " + constants.VALID_X_PATH)
		valid_X = DataManager.load_np_array(constants.VALID_X_PATH)
		logging.info("Loading validation Y from " + constants.VALID_Y_PATH)
		valid_Y = DataManager.load_np_array(constants.VALID_Y_PATH)
		
		logging.info("Loading test X from " + constants.TEST_X_PATH)
		test_X = DataManager.load_np_array(constants.TEST_X_PATH)
		return train_X, train_Y, valid_X, valid_Y, test_X