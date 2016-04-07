#!/bin/python
import os.path
import pandas as pd
import numpy as np
from six.moves import cPickle

motifpath = 'top10align_motifs/'   # directory where motif files are located
motiflist = 'motif.pickle'         # output filename

# get all motif files in motifpath directory
listdir = os.listdir(motifpath)

# parse motifs
motif_set = []
for files in listdir:
    df = pd.read_table(os.path.join(motifpath,files))
    motif_set.append(df.iloc[0::,1::].transpose())

# save motifs    
f = open(motiflist, 'wb')
cPickle.dump(motif_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

