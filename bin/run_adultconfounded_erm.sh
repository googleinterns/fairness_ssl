for wd in 1.0 0.1 0.01
do
    python train_and_eval_loop.py \
	   --dataset 'AdultConfounded' \
	   --method 'erm' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.001 \
	   --weight_decay "$wd"
done


