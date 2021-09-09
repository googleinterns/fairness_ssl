for lr in 0.001 0.0001 0.00001
do
    for wd in 0.1 0.01 0.001
    do
	for param in 0.01 0.001 0.0001
	do
	    python train_and_eval_loop.py \
		   --dataset 'Adult2' \
		   --method 'groupdro' \
		   --model_type 'fullyconn' \
		   --latent_dim 32 \
		   --optimizer 'Adam' \
		   --batch_size 128 \
		   --noflag_saveckpt \
		   --num_epoch 2 \
		   --learning_rate "$lr" \
		   --weight_decay "$wd" \
		   --groupdro_stepsize "$param" \
		   --ckpt_prefix "results_temp" \
		   --flag_run_all \
		   --gpu_ids '0'
	done
    done
done
			   