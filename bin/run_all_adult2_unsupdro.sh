for lr in 0.001 0.0001 0.00001
do
    for wd in 0.1 0.01 0.001
    do
	for param in 0.9 0.8 0.7
	do
	    python train_and_eval_loop.py \
		   --dataset 'Adult2' \
		   --method 'unsupdro' \
		   --model_type 'fullyconn' \
		   --latent_dim 32 \
		   --optimizer 'Adam' \
		   --batch_size 128 \
		   --noflag_saveckpt \
		   --num_epoch 201 \
		   --learning_rate "$lr" \
		   --weight_decay "$wd" \
		   --unsupdro_eta "$param" \
		   --ckpt_prefix "results_all" \
		   --flag_run_all \
		   --gpu_ids "1"
	done
    done
done
