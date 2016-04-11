#/bin/python

def load_model(model_name, shape, num_labels):

	# load and build model parameters
	if model_name == "binary_genome_motif_model":
		from .binary_genome_motif_model import binary_genome_motif_model
		model_layers, input_var, target_var, optimization = binary_genome_motif_model(shape, num_labels)

	elif model_name == "categorical_genome_motif_model":
		from .categorical_genome_motif_model import categorical_genome_motif_model
		model_layers, input_var, target_var, optimization = categorical_genome_motif_model(shape, num_labels)

	elif model_name == "genome_motif_model":
		from .genome_motif_model import genome_motif_model
		model_layers, input_var, target_var, optimization = genome_motif_model(shape, num_labels)

	elif model_name == "deepsea_model":
		from .simple_genome_motif_model import deepsea_model
		model_layers, input_var, target_var, optimization = deepsea_model(shape, num_labels)

	return model_layers, input_var, target_var, optimization

	