prog_ep1en3 () {
eps=0.001
python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 50 \
       --weight_decay 0.1  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.44,.41,.14,.01 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '0' 
}
prog_ep1en2 () {
eps=0.01
python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 50 \
       --weight_decay 0.1  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.44,.41,.14,.01 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '1' 
}
prog_ep1en1 () {
eps=0.1
python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 50 \
       --weight_decay 0.1  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.44,.41,.14,.01 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '2' 
}
prog_ep1en0 () {
eps=1.0
python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 50 \
       --weight_decay 0.1  \
       --lab_split 0.1 \
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.44,.41,.14,.01 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon" \
       --flag_run_all \
       --gpu_ids '3' 
}
prog_ep1en3 & prog_ep1en2 & prog_ep1en1 & prog_ep1en0;
