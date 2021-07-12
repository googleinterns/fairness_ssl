for ratio in 0.1
do
    python train_and_eval_loop.py \
	   --dataset 'German' \
	   --method 'groupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --learning_rate 0.01 \
	   --groupdro_stepsize 0.001 \
	   --lab_split "$ratio"  \
	   --flag_debug
done
