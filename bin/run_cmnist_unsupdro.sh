for param in 0.95 0.9 0.6 0.3 0.1
do 
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'unsupdro' \
	   --optimizer 'Adam' \
	   --learning_rate 0.0005 \
	   --noflag_saveckpt \
	   --batch_size 39640 \
	   --num_epoch 501 \
	   --weight_decay 0.001  \
	   --unsupdro_eta "$param" \
	   --gpu_ids '5' 
done
