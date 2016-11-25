from __future__ import print_function
import logging
import time
import numpy as np
import pandas as pd

"""
This class trains and compares different models on the same data.
"""
class Trainer(object):
	def __init__(self, models, train_X, train_Y, valid_X, valid_Y, test_X):
		self.models = models
		self.train_X = train_X
		self.train_Y = train_Y
		self.valid_X = valid_X
		self.valid_Y = valid_Y
		self.test_X = test_X
		self.results = []
	
	@staticmethod
	def accuracy(predicted_Y, true_Y):
		logging.debug("Accuracy between predicted_Y: " + str(predicted_Y.shape) + " and true_Y: " + str(true_Y.shape))
		logging.debug("Accuracy: " + str(predicted_Y[0:10]) + " VS " + str(true_Y[0:10]))
		return float('%.2f'%(100.0 * np.sum(predicted_Y == true_Y) / predicted_Y.shape[0])) # TO BE TESTED
	
	def evaluate(self, model):
		predicted_train_Y = model.predict(self.train_X)
		predicted_valid_Y = model.predict(self.valid_X)
		return Trainer.accuracy(predicted_train_Y,self.train_Y), Trainer.accuracy(predicted_valid_Y,self.valid_Y)
		
	def train(self):
		logging.info("Start training all models")
		self.results = [] # [{model,train_accuracy,valid_accuracy}]
		for model in self.models:
			logging.info("Training " + model.name + "(" + model.hyperparameters_as_string() + ")...")
			begin_time = time.time()
			model.fit(self.train_X,self.train_Y)
			end_time = time.time()
			training_duration = int(end_time - begin_time)
			train_accuracy, valid_accuracy = self.evaluate(model)
			self.results.append({"model" : model, "train_accuracy" : train_accuracy, "valid_accuracy" : valid_accuracy, "training_duration" : training_duration})
		return self.results
		
	def print_results(self):
		logging.info("Printing results")
		fields = ["Hyperparameters","Training Accuracy (%)","Validation Accuracy (%)","Training Duration (s)"]
		row_format = "{:>30}" * (len(fields) + 1)
		logging.info(row_format.format("", *fields))
		for result in self.results:
			model = result["model"]
			name = model.name
			hyperparameters_string = model.hyperparameters_as_string()
			train_accuracy = result["train_accuracy"]
			valid_accuracy = result["valid_accuracy"]
			training_duration = result["training_duration"]
			logging.info(row_format.format(name, hyperparameters_string, train_accuracy, valid_accuracy, training_duration))
			
	def get_best_model(self):
		best_model = None
		max = 0.0
		
		for result in self.results:
			valid_accuracy = result["valid_accuracy"]
			if valid_accuracy > max:
				best_model = result["model"]
				max = valid_accuracy
		
		return best_model
	
	def generate_submission_file(self, file_path):
		best_model = self.get_best_model()
		
		print("Model used for generating submission file:",best_model.name,"(",best_model.hyperparameters_as_string(),")")
		
		predictions = best_model.predict(self.test_X[:,1:]) # Ignore PassengerId
	
		submission = pd.DataFrame(columns=('PassengerId','Survived'))
		submission['PassengerId'] = submission['PassengerId'].astype(int)
		submission['Survived'] = submission['Survived'].astype(int)

		for i in range(len(self.test_X)):
			passengerId = int(self.test_X[i][0])
			survived = int(predictions[i]) # <-- make prediction here
			submission.loc[i] = [passengerId,survived] # Add row
			
		submission.to_csv(file_path, index=False)
		print("Saved to", file_path)