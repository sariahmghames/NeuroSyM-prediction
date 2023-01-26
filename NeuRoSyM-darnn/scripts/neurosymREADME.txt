# Neuro-Symbolic approah for the Input and Time Attention Series Prediction

Navigate to the directoty scripts_longterm/ to start the training
To Get started you can launch the shell scripts:
- peds_pre_caf_new1.sh for the 48 times steps prediction
- peds_pre_caf_new2.sh for the 80 time steps prediction

To switch between the following models:
- no input attention
- input attention as per the original DA-RNN paper
- input attention with a-priori information on the possible effect of each input time series on the first one (the one we predict), provided by 
the injection of CND-driven attention weights (for human motion prediction application) directly after the series embedding.

please adjust the parameters in the neurosym_train.py:
alpha_update = True/False
inp_att_enabled = True/False
temporal_att_enabled = True/False
run_validation = True/False
run_test = True/False
run_train = True/False

in addition to the hyperparams in the configuraion files in the direcory conf/
