for lr in 0.001 0.0001 0.00001
do
    for wd in 0.01 0.001 0.0001
    do
	for sd in 43 44
	do
	python train_and_eval_loop.py \
	       --dataset 'CMNIST' \
	       --model_type 'mlp' \
	       --latent_dim 390 \
	       --method 'erm' \
	       --optimizer 'Adam' \
	       --learning_rate "$lr" \
	       --batch_size 39640 \
	       --noflag_saveckpt \
	       --num_epoch 1 \
	       --weight_decay "$wd" \
	       --seed "$sd" \
	       --ckpt_prefix "results_temp" \
	       --flag_run_all &
	done
    done
done
wait
