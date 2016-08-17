#/bin/python
import os, sys
import numpy as np
sys.path.append('..')
from src.neuralnetwork import NeuralNet, NeuralTrainer
import src.train as fit 
import load_data
from models.genome_motif_model import model

np.random.seed(247) # for reproducibility


# -----------------------------------------------------------------------------------------------------
# dataset to use
# H1-hESC   CTCF  GABP SP1  SRF  TAF1 YY1                   ATF2 TCF12 Rad21 NANOG CEBPB JunD ATF3
#           [262, 265, 278, 281, 282, 287]                  [259, 284 480 268,]468, 267,  260, 
#
# HepG2     CTCF  GABP SP1  SRF  TAF1 YY1                     SP2  FOXA1 JunD ATF3 
#           [299, 305, 321, 323, 324, 328]                  [ , 322, ,302, 309, 295, ]
#
# K562      CTCF  GABP SP1  SRF  TAF1 YY1                     GATA2 STAT5AMax CEBPB ATF3 
#           [338, 345, 359, 361, 363, 369]                  346, 362,348,337, 333, 
#
# all
# [262, 265, 282, 287, 278, 281, 299, 305, 324, 328, 321, 333, 338, 345, 363, 369, 359, 361] 
#
# just CTCF
# K562 H1-hESC HepG2                                       
# [415, 299, 338]                                              
#
# -----------------------------------------------------------------------------------------------------
# dataset not used
# A549      CTCF GABP YY1  FOXA1 CEBPB Max                               
#           [173, 179, 195, 178, 402, 403]                               
#
# GM12878   ATF3 CTCF  GABP TAF1  YY1              SP1 SRF           ATF2 FOXM1 STAT5A JunD Max STAT1 STAT3 ZNF274 
#           [205, 415, 216, 240, 244]              237, 238,         [204 215, ]239 420, , 421, 439, 440, 448
#
# HeLa-S3   GABP CTCF CEBPB JunD STAT1  MAfK Max                 
#           [291, 737 503, 518, 535, 519, 520]                       509,  519, 520, 531,  538, 545, 
#
# range(129,330) big range
# HeLa-S3
# GABP NRSF AP-2alpha AP-2gamma BAF155 BAF170 BDP1 BRCA1 BRF1 BRF2 Brg1 CEBPB 
# c-Fos CHD2 c-Jun c-Myc COREST E2F1 E2F4 E2F6 ELK1 ELK4 GTF2F1 HA-E2F1 Ini1 
# IRF3 JunD MafK Max MAZ Mxi1 NF-YA NF-YB Nrf1 p300 Pol2(phosphoS2) Pol2 PRDM1 
# Rad21 RFX5 RPC155 SMC3 SPT20 STAT1 STAT3 TBP TCF7L2
# np.hstack([291, 292, range(494,538)])
#
# Dnase
# A549 GM12878 H1-hESC HeLA-S3 HepG2 K562 MCF-7 
# [52, 53, 54, 55, 56, 61, 63]
#------------------------------------------------------------------------------
# load data

# class = ['H1-hESC','HepG2', 'K562', 'combine', 'CTCF', 'all']

datapath = '/media/peter/storage/DeepSea/'
filepath = os.path.join(datapath,'TF_dataset.hdf5')
train, valid, test = load_data.Encode_TF(filepath, tf_index=0)

#-------------------------------------------------------------------------------------

# build network
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = train[1].shape[1]
network, input_var, target_var, optimization = model(shape, num_labels)

# build neural network class
nnmodel = NeuralNet(network, input_var, target_var)
nnmodel.inspect_layers()

# set output file paths
output_name = 'test'
filepath = os.path.join(datapath, 'Results', output_name)
nntrainer = NeuralTrainer(nnmodel, optimization, save='best', filepath=filepath)

# train model
fit.train_minibatch(nntrainer, data={'train': train, 'valid': valid}, 
                              batch_size=100, num_epochs=500, patience=10, verbose=1)

# load best model --> lowest cross-validation error
nntrainer.set_best_parameters()

# test model
nntrainer.test_model(test, batch_size, "test")

# save all performance metrics (train, valid, test)
nntrainer.save_all_metrics(filepath)

















