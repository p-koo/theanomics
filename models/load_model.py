#/bin/python

def load_model(model_name, shape, num_labels):

	# load and build model parameters
	if model_name == "binary_genome_motif_model":
		from .binary_genome_motif_model import binary_genome_motif_model
		network, input_var, target_var, optimization = binary_genome_motif_model(shape, num_labels)

	elif model_name == "categorical_genome_motif_model":
		from .categorical_genome_motif_model import categorical_genome_motif_model
		network, input_var, target_var, optimization = categorical_genome_motif_model(shape, num_labels)

	elif model_name == "genome_motif_model":
		from .genome_motif_model import genome_motif_model
		network, input_var, target_var, optimization = genome_motif_model(shape, num_labels)

	elif model_name == "deepsea_model":
		from .deepsea_model import deepsea_model
		network, input_var, target_var, optimization = deepsea_model(shape, num_labels)

	elif model_name == "jaspar_motif_model":
		from .jaspar_motif_model import jaspar_motif_model
		network, input_var, target_var, optimization = jaspar_motif_model(shape, num_labels)

	elif model_name == "jaspar_motif_model2":
		from .jaspar_motif_model2 import jaspar_motif_model2
		network, input_var, target_var, optimization = jaspar_motif_model2(shape, num_labels)

	elif model_name == "cyclic_genome_motif_model":
		from .cyclic_genome_motif_model import cyclic_genome_motif_model
		network, input_var, target_var, optimization = cyclic_genome_motif_model(shape, num_labels)


	elif model_name == "inception_genome_motif_model":
		from .inception_genome_motif_model import inception_genome_motif_model
		network, input_var, target_var, optimization = inception_genome_motif_model(shape, num_labels)


	elif model_name == "recurrent_inception_motif_model":
		from .recurrent_inception_motif_model import recurrent_inception_motif_model
		network, input_var, target_var, optimization = recurrent_inception_motif_model(shape, num_labels)


	elif model_name == "conv_LSTM_model":
		from .conv_LSTM_model import conv_LSTM_model
		network, input_var, target_var, optimization = conv_LSTM_model(shape, num_labels)

	elif model_name == "MNIST_CNN_model":
		from .MNIST_CNN_model import MNIST_CNN_model
		network, input_var, target_var, optimization = MNIST_CNN_model(shape, num_labels)

	elif model_name == "test_motif_model":
		from .test_motif_model import test_motif_model
		network, input_var, target_var, optimization = test_motif_model(shape, num_labels)


		

	return network, input_var, target_var, optimization

	