from __future__ import print_function
import logging
import time
import numpy as np

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
	
	def evaluate(self,model):
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