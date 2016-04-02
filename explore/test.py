import sys
import numpy as np
import time

print("Starting training...")
bar_length = 20
start_time = time.time()
train_loss = 0
train_accuracy = 0
for index in range(1000):

    # progress bar
    remaining_time = (time.time()-start_time)*(1000-index-1)/(index+1)
    percent = (index+1.)/1000
    progress = '-'*int(round(percent*bar_length))
    spaces = ' '*int(round((1000-index-1)/1000*bar_length))
    sys.stdout.write("[%s] %.1f%% -- est.time=%d s -- loss=%.4f -- accuracy=%.3f \n" \
    %(progress+spaces, percent*100, remaining_time, np.random.uniform(), np.random.uniform()))
    sys.stdout.flush()


