for lr in 0.001 0.0001 0.00001
do
    for wd in 0.01 0.001 0.0001
    do
	for param in 0.01 0.001 0.0001
	do
	    for sd in 43
	    do
	    python train_and_eval_loop.py \
		   --dataset 'CMNIST' \
		   --model_type 'mlp' \
		   --latent_dim 390 \
		   --method 'groupdro' \
		   --optimizer 'Adam' \
		   --learning_rate "$lr" \
		   --batch_size 39640 \
		   --noflag_saveckpt \
		   --num_epoch 501 \
		   --weight_decay "$wd"  \
		   --groupdro_stepsize "$param" \
		   --seed "$sd" \
		   --ckpt_prefix "results_all" \
		   --flag_run_all &
	    done
	done
    done
done
wait
for lr in 0.001 0.0001 0.00001
do
    for wd in 0.01 0.001 0.0001
    do
	for param in 0.01 0.001 0.0001
	do
	    for sd in 44
	    do
	    python train_and_eval_loop.py \
		   --dataset 'CMNIST' \
		   --model_type 'mlp' \
		   --latent_dim 390 \
		   --method 'groupdro' \
		   --optimizer 'Adam' \
		   --learning_rate "$lr" \
		   --batch_size 39640 \
		   --noflag_saveckpt \
		   --num_epoch 501 \
		   --weight_decay "$wd"  \
		   --groupdro_stepsize "$param" \
		   --seed "$sd" \
		   --ckpt_prefix "results_all" \
		   --flag_run_all &
	    done
	done
    done
done
wait
