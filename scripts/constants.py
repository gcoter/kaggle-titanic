DATA_FOLDER = "../data/" # Data folder
RAW_DATA_FOLDER = DATA_FOLDER + "raw/" # Sub folder containing raw files
SETS_DATA_FOLDER = DATA_FOLDER + "sets/" # Sub folder containing training, validation and test set backups as numpy arrays
TRAIN_X_PATH = SETS_DATA_FOLDER + "train_X.npy" # Training set backup file path
TRAIN_Y_PATH = SETS_DATA_FOLDER + "train_Y.npy"
VALID_X_PATH = SETS_DATA_FOLDER + "valid_X.npy" # Validation set backup file path
VALID_Y_PATH = SETS_DATA_FOLDER + "valid_Y.npy"
TEST_X_PATH = SETS_DATA_FOLDER + "test_X.npy" # Test set backup file path

RESULTS_FOLDER = "../results/"

# Dictionnary: keys are file names and values are paths.
FILES_PATHS = {
	"train.csv" : RAW_DATA_FOLDER + "train.csv", # Example
	"test.csv" : RAW_DATA_FOLDER + "test.csv" # Example
}

# Constants
FREQUENT_TITLES = ['Mr.','Miss.','Mrs.']
POSSIBLE_TITLES = FREQUENT_TITLES + ['Rare']