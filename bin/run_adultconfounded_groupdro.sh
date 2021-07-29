for stepsz in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
do
    python train_and_eval_loop.py \
	   --dataset 'AdultConfounded' \
	   --method 'groupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 201 \
	   --learning_rate 0.001 \
	   --groupdro_stepsize "$stepsz"
done


