prog_ep1en3 () {
eps=0.001
python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'worstoffdro' \
       --optimizer 'Adam' \
       --learning_rate 0.0001 \
       --batch_size 39640 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.01  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.0001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.40,.07 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '4' 
}       
prog_ep1en2 () {
eps=0.01
python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'worstoffdro' \
       --optimizer 'Adam' \
       --learning_rate 0.0001 \
       --batch_size 39640 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.01  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.0001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.40,.07 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '5' 
}       
prog_ep1en1 () {
eps=0.1
python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'worstoffdro' \
       --optimizer 'Adam' \
       --learning_rate 0.0001 \
       --batch_size 39640 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.01  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.0001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.40,.07 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '6' 
}       
prog_ep1en0 () {
eps=1.0
python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'worstoffdro' \
       --optimizer 'Adam' \
       --learning_rate 0.0001 \
       --batch_size 39640 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.01  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.0001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.40,.07 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '7' 
}       
prog_ep1en3 & prog_ep1en2 & prog_ep1en1 & prog_ep1en0;
