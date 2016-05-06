
"""

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

random
  valid loss:     0.23036
  valid accuracy: 0.90839+/-0.01586
  valid auc-roc:  0.92753+/-0.02201
  valid auc-pr:   0.79954+/-0.05636

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

random
  test loss:      0.17300
  test accuracy:  0.93249+/-0.01993
  test auc-roc:   0.95876+/-0.01724
  test auc-pr:    0.87862+/-0.05194

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
Comparison of deeper networks
--------------------------------------
DNAse (0-40)
Model: deep (7 conv layers)




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