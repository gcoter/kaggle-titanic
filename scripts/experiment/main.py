from __future__ import print_function
import constants
from datamanager import DataManager
import sys
import logging
from trainer import Trainer
import models

"""
Main code
"""
def get_command_line_arguments():
	argsdict = {}

	for farg in sys.argv:
		if farg.startswith('--'):
			(arg,val) = farg.split("=")
			arg = arg[2:]
			
			if arg not in argsdict:
				argsdict[arg] = val
				
	return argsdict

argsdict = get_command_line_arguments()

# Defines loglevel
if "log" in argsdict:
	loglevel = argsdict["log"]
else:
	loglevel = "INFO"

numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

# === MAIN CODE ===
train_X, train_Y, valid_X, valid_Y, test_X = DataManager.get_datasets(constants.FILES_PATHS)

logging.info("Train X: " + str(train_X.shape))
logging.info("Train Y: " + str(train_Y.shape))
logging.info("Validation X: " + str(valid_X.shape))
logging.info("Validation Y: " + str(valid_Y.shape))
logging.info("Test X: " + str(test_X.shape))

C_values = [1.0] + [10*i for i in range(1,10)]
n_estimators_values = [10] + [100*i for i in range(1,10)]

# Put the models you want to test here
models_to_test = [models.SKLearnLogisticRegression(C=C_value,max_iter=10000) for C_value in C_values]
models_to_test += [models.SKRandomForest(n_estimators = n_estimators_value) for n_estimators_value in n_estimators_values]
models_to_test += [models.SKNaiveBayes()]

trainer = Trainer(models_to_test, train_X, train_Y, valid_X, valid_Y, test_X)
results = trainer.train()
trainer.print_results()

trainer.generate_submission_file(constants.RESULTS_FOLDER + "submission.csv")