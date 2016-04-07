#/bin/python
"""
Data sets:
	'load_DeepSea',
	'load_MotifSimulation_categorical'
"""

def load_data(name, filepath, options):

	# load and build model parameters
	if model_name == "load_DeepSea":
		import load_DeepSea
		if "num_include" in options:
			num_include = options["num_include"]
		else:
			num_include = 4400000
		if "class_range" in options:
			class_range = options["class_range"]
		else:
			class_range = range(918)
		train, valid, test = load_DeepSea(filepath, num_include, class_range)

	elif model_name == "load_MotifSimulation_binary":
		import load_MotifSimulation_binary
		train, valid, test = load_MotifSimulation_binary(filepath)

	elif model_name == "load_MotifSimulation_categorical":
		import load_MotifSimulation_categorical
		train, valid, test = load_MotifSimulation_categorical(filepath)


	return layers, input_var, target_var, optimization