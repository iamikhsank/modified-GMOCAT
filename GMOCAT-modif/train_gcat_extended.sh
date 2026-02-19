python ./launch_gcat.py -environment GCATEnv -T 20 -ST [5,10,15,20] -agent GCATAgent -FA GCAT -seed 145 -emb_dim 128 -cdm_bs 32 \
-training_epoch 5 -train_bs 8 -test_bs 8 -learning_rate 0.001 -policy_epoch 1 \
-gamma 0.5 -n_head 1 -n_block 1 -dropout_rate 0.1 -graph_block 1 \
-morl_weights [1,1,1] -CDM NCD -data_name dbekt22 -gpu_no 0 -cdm_lr 0.02 -cdm_epoch 2 -device cpu
