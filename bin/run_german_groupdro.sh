for stepsz in 0.01 0.001 0.0001
do
    python train_and_eval_loop.py \
	   --dataset 'German' \
	   --method 'groupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --learning_rate 0.01 \
	   --groupdro_stepsize "$stepsz"
done


