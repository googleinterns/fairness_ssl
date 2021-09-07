for lr in 0.001 0.0001 0.00001
do
    for wd in 0.01 0.001 0.0001
    do
	for param in 0.9 0.8 0.7
	do 
	    python train_and_eval_loop.py \
		   --dataset 'CMNIST' \
		   --model_type 'mlp' \
		   --latent_dim 390 \
		   --method 'unsupdro' \
		   --optimizer 'Adam' \
		   --learning_rate "$lr" \
		   --noflag_saveckpt \
		   --batch_size 39640 \
		   --num_epoch 5 \
		   --weight_decay "$wd"  \
		   --unsupdro_eta "$param" \
		   --ckpt_prefix "results_temp" \
		   --flag_run_all \
		   --gpu_ids '5'
	done
    done
done
