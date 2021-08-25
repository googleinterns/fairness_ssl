for param in 0.95 0.9 0.6 0.3 0.1
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
	   --learning_rate 0.0001 \
	   --weight_decay 0.01 \
	   --unsupdro_eta "$param" \
	   --gpu_ids '4'
done
