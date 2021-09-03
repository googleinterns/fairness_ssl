for lr in 0.001 0.01
do
    python train_and_eval_loop.py \
	   --dataset 'Adult2' \
	   --method 'erm' \
	   --model_type 'fullyconn' \
	   --latent_dim 32 \
	   --optimizer 'Adam' \
	   --batch_size 128 \
	   --noflag_saveckpt \
	   --noflag_usegpu \
	   --num_epoch 2 \
	   --learning_rate "$lr" \
	   --weight_decay 0.01  \
	   --ckpt_prefix 'results_all'\
	   --flag_run_all
done
    
