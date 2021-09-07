for lr in 0.001 0.0001 0.00001
do
    for wd in 0.01 0.001 0.0001
    do
	for param in 0.01 0.001 0.0001
	do
	python train_and_eval_loop.py \
	       --dataset 'CMNIST' \
	       --model_type 'mlp' \
	       --latent_dim 390 \
	       --method 'worstoffdro' \
	       --optimizer 'Adam' \
	       --learning_rate "$lr" \
	       --batch_size 39640 \
	       --noflag_saveckpt \
	       --num_epoch 501 \
	       --weight_decay "$wd"  \
	       --lab_split 0.1 \
	       --worstoffdro_stepsize "$param" \
	       --worstoffdro_lambda 1.0 \
	       --worstoffdro_latestart 0 \
	       --worstoffdro_marginals=.53,.40,.07 \
	       --ckpt_prefix "results_all" \
	       --flag_run_all \
	       --gpu_ids '7'
	done
    done
done
       
