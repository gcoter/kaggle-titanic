import cPickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from abc import abstractmethod

"""
In this script, you can define subclasses of Model.
"""
class Model(object):
	@staticmethod
	def load(file_path):
		return cPickle.load(open(file_path, 'rb'))
		
	def __init__(self,name,**kwargs):
		self.name = name
		self.hyperparameters = kwargs
		self.parameters = []
		self.model_object = None
		
	def save(self,file_path):
		logging.info("Model saved to " + file_path)
		cPickle.dump(self, open(file_path, 'wb'))
		
	def hyperparameters_as_string(self):
		res = ""
		for name in self.hyperparameters.keys():
			value = self.hyperparameters[name]
			res += name + "=" + str(value) + ","
		return res.strip()[:-1]
		
	@abstractmethod
	def fit(self,X,Y):
		""" 
		This method fits (modifies self.parameters) the model to the data (using self.hyperparameters).
		"""
		return
		
	@abstractmethod
	def predict(self,X):
		""" 
		This method predicts Y given X.
		
		Returns predicted Y.
		"""
		return
		
class SKModel(Model):
	def __init__(self,name,**kwargs):
		Model.__init__(self,name,**kwargs)

	def fit(self,X,Y):
		self.model_object.fit(X,Y)
		
	def predict(self,X):
		return self.model_object.predict(X)
		
class SKLearnLogisticRegression(SKModel):
	def __init__(self,**kwargs):
		SKModel.__init__(self,"SKLearnLogisticRegression",**kwargs)
		self.model_object = LogisticRegression(**kwargs)
		
class SKRandomForest(SKModel):
	def __init__(self,**kwargs):
		SKModel.__init__(self,"SKRandomForest",**kwargs)
		self.model_object = RandomForestClassifier(**kwargs)