python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'worstoffdro' \
       --optimizer 'Adam' \
       --learning_rate 0.0005 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 501 \
       --weight_decay 0.001  \
       --lab_split 0.3 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.5,.4,.1 \
       --gpu_ids '3'
       
