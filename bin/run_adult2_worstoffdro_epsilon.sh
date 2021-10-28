prog_ep1en3 () {
    eps=0.001
    python train_and_eval_loop.py \
	   --dataset 'Adult2' \
	   --method 'worstoffdro' \
	   --model_type 'fullyconn' \
	   --latent_dim 32 \
	   --optimizer 'Adam' \
	   --batch_size 128 \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.00001 \
	   --weight_decay 0.001 \
	   --lab_split 0.1 \
	   --worstoffdro_stepsize 0.0001 \
	   --worstoffdro_lambda 1.0 \
	   --worstoffdro_latestart 0 \
	   --worstoffdro_marginals=.63,.27,.05,.05 \
	   --epsilon="$eps" \
	   --ckpt_prefix "results_epsilon" \
	   --flag_run_all \
	   --gpu_ids '0'
}
prog_ep1en2 () {
    eps=0.01
    python train_and_eval_loop.py \
	   --dataset 'Adult2' \
	   --method 'worstoffdro' \
	   --model_type 'fullyconn' \
	   --latent_dim 32 \
	   --optimizer 'Adam' \
	   --batch_size 128 \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.00001 \
	   --weight_decay 0.001 \
	   --lab_split 0.1 \
	   --worstoffdro_stepsize 0.0001 \
	   --worstoffdro_lambda 1.0 \
	   --worstoffdro_latestart 0 \
	   --worstoffdro_marginals=.63,.27,.05,.05 \
	   --epsilon="$eps" \
	   --ckpt_prefix "results_epsilon" \
	   --flag_run_all \
	   --gpu_ids '1'
}
prog_ep1en1 () {
    eps=0.1
    python train_and_eval_loop.py \
	   --dataset 'Adult2' \
	   --method 'worstoffdro' \
	   --model_type 'fullyconn' \
	   --latent_dim 32 \
	   --optimizer 'Adam' \
	   --batch_size 128 \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.00001 \
	   --weight_decay 0.001 \
	   --lab_split 0.1 \
	   --worstoffdro_stepsize 0.0001 \
	   --worstoffdro_lambda 1.0 \
	   --worstoffdro_latestart 0 \
	   --worstoffdro_marginals=.63,.27,.05,.05 \
	   --epsilon="$eps" \
	   --ckpt_prefix "results_epsilon" \
	   --flag_run_all \
	   --gpu_ids '2'
}
prog_ep1en0 () {
    eps=1.0
    python train_and_eval_loop.py \
	   --dataset 'Adult2' \
	   --method 'worstoffdro' \
	   --model_type 'fullyconn' \
	   --latent_dim 32 \
	   --optimizer 'Adam' \
	   --batch_size 128 \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.00001 \
	   --weight_decay 0.001 \
	   --lab_split 0.1 \
	   --worstoffdro_stepsize 0.0001 \
	   --worstoffdro_lambda 1.0 \
	   --worstoffdro_latestart 0 \
	   --worstoffdro_marginals=.63,.27,.05,.05 \
	   --epsilon="$eps" \
	   --ckpt_prefix "results_epsilon" \
	   --flag_run_all \
	   --gpu_ids '3'
}
prog_ep1en3 & prog_ep1en2 & prog_ep1en1 & prog_ep1en0;
