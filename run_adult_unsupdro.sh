for eta in 0.9 0.6 0.3 0.1
do
    python train_and_eval_loop.py \
	   --dataset 'Adult' \
	   --method 'unsupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 51 \
	   --learning_rate 0.001 \
	   --unsupdro_eta "$eta"
done


