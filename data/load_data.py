#/bin/python
"""
Data sets:
	'load_DeepSea',
	'load_MotifSimulation_categorical'
	'load_MotifSimulation_categorical'
"""

def load_data(model_name, filepath, options=[]):

	# load and build model parameters
	if model_name == "DeepSea":
		from .DeepSea import DeepSea
		if "num_include" in options:
			num_include = options["num_include"]
		else:
			num_include = []
		if "class_range" in options:
			class_range = options["class_range"]
		else:
			class_range = range(918)
		train, valid, test = DeepSea(filepath, class_range, num_include)

	elif model_name == "Basset":
		from .Basset import Basset
		if "num_include" in options:
			num_include = options["num_include"]
		else:
			num_include = []
		if "class_range" in options:
			class_range = options["class_range"]
		else:
			class_range = range(164)
		train, valid, test = Basset(filepath, class_range, num_include)

	elif model_name == "MotifSimulation_binary":
		from .MotifSimulation_binary import MotifSimulation_binary
		train, valid, test = MotifSimulation_binary(filepath)
	
	elif model_name == "MotifSimulation_correlated":
		from .MotifSimulation_correlated import MotifSimulation_correlated
		train, valid, test = MotifSimulation_correlated(filepath)

	elif model_name == "MotifSimulation_categorical":
		from .MotifSimulation_categorical import MotifSimulation_categorical 
		train, valid, test = MotifSimulation_categorical(filepath)

	elif model_name == "RNA_compete":
		from .RNA_compete import RNA_compete 
		train, valid, test = RNA_compete(filepath)

	return train, valid, test