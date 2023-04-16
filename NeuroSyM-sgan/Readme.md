## NeuRoSyM Prediction framework 


Note: 
All .py files that have similar name but differ with "_informed" are imported from the original SGAN repository (https://github.com/agrimgupta92/sgan), the newly proposed neuro-symbolic approach for sgan is implemented in the "_informed" files where some enherits utility files from the original sgan. 




## Original SGAN loaded hyperparameters (for investigation)

loaded model args are : AttrDict({'encoder_h_dim_d': 48, 'neighborhood_size': 2.0, 'pool_every_timestep': False, 'clipping_threshold_g': 2.0, 'delim': 'tab', 'dataset_name': 'hotel', 'print_every': 100, 'skip': 1, 'loader_num_workers': 4, 'd_steps': 1, 'batch_size': 64, 'num_epochs': 200, 'num_layers': 1, 'best_k': 20, 'obs_len': 8, 'pred_len': 8, 'g_steps': 1, 'g_learning_rate': 0.0001, 'l2_loss_weight': 1.0, 'grid_size': 8, 'bottleneck_dim': 8, 'checkpoint_name': 'checkpoint', 'gpu_num': '0', 'restore_from_checkpoint': 1, 'dropout': 0.0, 'noise_mix_type': 'global', 'decoder_h_dim_g': 32, 'pooling_type': 'pool_net', 'use_gpu': 1, 'num_iterations': 8512, 'batch_norm': False, 'noise_type': 'gaussian', 'clipping_threshold_d': 0, 'encoder_h_dim_g': 32, 'checkpoint_every': 300, 'd_learning_rate': 0.001, 'checkpoint_start_from': None, 'timing': 0, 'mlp_dim': 64, 'num_samples_check': 5000, 'd_type': 'global', 'noise_dim': (8,), 'embedding_dim': 16})


## NeurRoSyM SGAN model training 

python scripts/train_informed.py --dataset_name zara1 --pool_every_timestep False --noise_dim 8 --checkpoint_name checkpoint_alpha_cnd_8ts_zara1 --bottleneck_dim 8 --encoder_h_dim_d 48 --batch_size 10 --encoder_h_dim_g 32 --embedding_dim 16 --mlp_dim 64 --decoder_h_dim_g 32 --num_epochs 200 --num_iterations 8512 --noise_mix_type 'global' --noise_type 'gaussian' --labels_dir "scripts/sgan/data/" --filename "cnd_labels.txt" --obs_len 8 --pred_len 8 --g_learning_rate 0.0001 --d_learning_rate 0.001

Note: to check the complete list of hyperparameters used for training each neurosym-sgan on each dataset, one can load the arrguments of the saved models provided in the directory neurosym-sgan-models/



## SGAN model evaluation

python scripts/evaluate_model.py --model_path scripts/models/sgan-p-models/hotel_8_model.pt --dset_type "test"


## NeurRoSyM SGAN model evaluation

python scripts/evaluate_model_informed.py --model_path scripts/models/neurosym-sgan-models/alpha_cnd_8ts_zara1/checkpoint_alpha_cnd_8ts_zara1_with_model.pt --dset_type "test"



