for stepsz in 0.9 0.5 0.1 0.01 0.001 0.05 0.005
do
    python train_and_eval_loop.py \
	   --dataset 'Adult' \
	   --method 'groupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 51 \
	   --learning_rate 0.001 \
	   --groupdro_stepsize "$stepsz"
done


