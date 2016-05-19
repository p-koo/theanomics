
"""
------------------------------------------------
N=100000

random:


correlated:


corrected:







-------------------------------------------------
--------------------------------------
Comparison of correlated vs random labels
--------------------------------------
N = 100000
Model: Basset (3 conv layers)

correlated
  valid loss:		  0.32906
  valid accuracy:	0.88787+/-0.02727
  valid auc-roc:	0.85525+/-0.07487
  valid auc-pr:		0.70259+/-0.10433

Epoch 1 out of 500 
[==============================] 100.0% -- time=0s -- loss=0.31827 -- accuracy=88.17%  
  valid loss:   0.29385
  valid accuracy: 0.89175+/-0.02742
  valid auc-roc:  0.85755+/-0.07188
  valid auc-pr:   0.70745+/-0.10271


random
  valid loss:     0.26782
  valid accuracy: 0.89853+/-0.01073
  valid auc-roc:  0.89646+/-0.02320
  valid auc-pr:   0.75200+/-0.04491

--------------------------------------
N = 100000
Model: deep (7 conv layers)

correlated:
  test loss:      0.26788
  test accuracy:  0.89599+/-0.02814
  test auc-roc:   0.88072+/-0.07354
  test auc-pr:    0.72204+/-0.11390

  test loss:      0.26770
  test accuracy:  0.89611+/-0.02818
  test auc-roc:   0.88083+/-0.07383
  test auc-pr:    0.72233+/-0.11381

random
  valid loss:     0.23036
  valid accuracy: 0.90839+/-0.01586
  valid auc-roc:  0.92753+/-0.02201
  valid auc-pr:   0.79954+/-0.05636

  test loss:      0.21332
  test accuracy:  0.91687+/-0.02138
  test auc-roc:   0.93734+/-0.02664
  test auc-pr:    0.82591+/-0.06886

  valid loss:   10.19320
  valid accuracy: 0.89179+/-0.00949
  valid auc-roc:  0.87734+/-0.01941
  valid auc-pr:   0.71958+/-0.03673
saving model parameters to: /home/peter/Data/SequenceMotif/Results/random_2_epoch_0.pickle
Epoch 2 out of 500 
[==============================] 100.0% -- time=0s -- loss=10.20822 -- accuracy=82.09%   
  valid loss:   10.08800
  valid accuracy: 0.89157+/-0.01154
  valid auc-roc:  0.88629+/-0.01847
  valid auc-pr:   0.73411+/-0.04228

decorr (GLS):
  test loss:      0.29031
  test accuracy:  0.89378+/-0.02789
  test auc-roc:   0.86402+/-0.07155
  test auc-pr:    0.70588+/-0.10900

decorr (multiBernoulli):
  valid loss:   
  valid accuracy: 0.89637+/-0.02746
  valid auc-roc:  0.88165+/-0.07128
  valid auc-pr:   0.72347+/-0.10733

  epoch 1
  valid loss:   18.74883
  valid accuracy: 0.89161+/-0.02724
  valid auc-roc:  0.86654+/-0.07309
  valid auc-pr:   0.71234+/-0.10714



--------------------------------------
N = 100000
Model: deep (9 conv layers)

correlated:

Epoch 1 out of 500 
[==============================] 100.0% -- time=0s -- loss=0.31398 -- accuracy=88.20%   
  valid loss:   0.28457
  valid accuracy: 0.89073+/-0.02690
  valid auc-roc:  0.86856+/-0.07204
  valid auc-pr:   0.71783+/-0.10721




random:




--------------------------------------
N = 300000
Model: Basset (3 conv layers)

correlated:
  test loss:      0.27111
  test accuracy:  0.89672+/-0.02793
  test auc-roc:   0.87529+/-0.07220
  test auc-pr:    0.72121+/-0.10939

random:
  test loss:      0.19494
  test accuracy:  0.92659+/-0.01660
  test auc-roc:   0.94477+/-0.01883
  test auc-pr:    0.85591+/-0.04864

--------------------------------------
N = 300000
Model: deep (7 conv layers)

correlated
  test loss:      0.26205
  test accuracy:  0.89782+/-0.02818
  test auc-roc:   0.88656+/-0.07149
  test auc-pr:    0.73059+/-0.11109

  test loss =     0.26441
  acc             0.89629 +/- 0.02778
  roc             0.885+/-0.0713
  pr              0.72874 +/- 0.1108

  test loss:    0.26240
  test accuracy:  0.89748+/-0.02789
  test auc-roc: 0.88581+/-0.07128
  test auc-pr:    0.72928+/-0.11108

random
  test loss:      0.17300
  test accuracy:  0.93249+/-0.01993
  test auc-roc:   0.95876+/-0.01724
  test auc-pr:    0.87862+/-0.05194

  test loss:      0.17258
  test accuracy:  0.93278+/-0.01994
  test auc-roc:   0.95892+/-0.01727
  test auc-pr:    0.87959+/-0.05185

  test loss:    0.18502
  test accuracy:  0.92743+/-0.01690
  test auc-roc: 0.95314+/-0.01627
  test auc-pr:    0.86416+/-0.04679

--------------------------------------
Comparison of deeper networks
--------------------------------------
N=300000, random

Model: Basset (3 conv layers)
  test loss:      0.19494
  test accuracy:  0.92659+/-0.01660
  test auc-roc:   0.94477+/-0.01883
  test auc-pr:    0.85591+/-0.04864

Model: deep (7 conv layers)
  test loss:      0.17300
  test accuracy:  0.93249+/-0.01993
  test auc-roc:   0.95876+/-0.01724
  test auc-pr:    0.87862+/-0.05194

Model: deep (9 conv layers)
  test loss:      0.17565
  test accuracy:  0.93367+/-0.01925
  test auc-roc:   0.95506+/-0.01819
  test auc-pr:    0.87779+/-0.05163


Model: super deep (11 conv layers)
  test loss:      0.18627
  test accuracy:  0.92790+/-0.02193
  test auc-roc:   0.95051+/-0.01954
  test auc-pr:    0.86235+/-0.05798





--------------------------------------
--------------------------------------
Comparison of deeper networks

--------------------------------------
--------------------------------------

RNAcompete

Model: 3 conv layer

OLS:
mu
  test loss:        1.77265
  test Pearson's R: 0.83048+/-0.20848
  test rsquare:     0.46605+/-0.19379
  test slope:       1.05217+/-0.05448
zero
  test loss:        1.78310
  test Pearson's R: 0.83136+/-0.20720
  test rsquare:     0.46786+/-0.19331
  test slope:       1.05399+/-0.06156
log
  test loss:        0.03985
  test Pearson's R: 0.81860+/-0.22368
  test rsquare:     0.43981+/-0.20366
  test slope:       1.02875+/-0.07990

GLS:
mu
  test loss:        0.79750
  test Pearson's R: 0.83180+/-0.20433
  test rsquare:     0.46658+/-0.18343
  test slope:       1.25920+/-0.06662

zero  
  test loss:        0.79824
  test Pearson's R: 0.83419+/-0.20199
  test rsquare:     0.47273+/-0.18401
  test slope:       1.19756+/-0.05781

log
  test loss:        0.82407
  test Pearson's R: 0.82669+/-0.21514
  test rsquare:     0.45758+/-0.20030
  test slope:       1.18893+/-0.06197

compare ols and gls by making distribution of errors on test set

__________________________________________________________
__________________________________________________________
__________________________________________________________
__________________________________________________________
OLD

random
  test loss:		0.19582
  test accuracy:	0.93177+/-0.02346
  test auc-roc:	0.90736+/-0.03641
  test auc-pr:		0.72602+/-0.09486

  test loss:		0.18212
  test accuracy:	0.93527+/-0.02120
  test auc-roc:	0.92287+/-0.03157
  test auc-pr:		0.75277+/-0.08848


correlated
  test loss:		0.19108
  test accuracy:	0.93210+/-0.02161
  test auc-roc:	0.91711+/-0.03147
  test auc-pr:		0.73934+/-0.08664

  test loss:		0.18065
  test accuracy:	0.93428+/-0.02234
  test auc-roc:	0.92796+/-0.02906
  test auc-pr:		0.76041+/-0.08327

random 300000
  test loss:		0.16663
  test accuracy:	0.94084+/-0.01945
  test auc-roc:	0.93659+/-0.02787
  test auc-pr:		0.79017+/-0.08711

correlated
3 conv layer
  test loss:		0.15933
  test accuracy:	0.94169+/-0.02506
  test auc-roc:	0.94500+/-0.02649
  test auc-pr:		0.80914+/-0.07878

5 conv layer
  test loss:		0.15302
  test accuracy:	0.94342+/-0.02494
  test auc-roc:	0.95012+/-0.02519
  test auc-pr:		0.82071+/-0.07227

7 conv layer
  test loss:		0.13757
  test accuracy:	0.94913+/-0.02334
  test auc-roc:	0.96317+/-0.02082
  test auc-pr:		0.85490+/-0.06482



"""