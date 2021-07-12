for ratio in 0.1 0.3 0.6 0.9
do
    python train_and_eval_loop.py \
	   --dataset 'Adult' \
	   --method 'groupdro' \
	   --noflag_usegpu \
	   --noflag_saveckpt \
	   --num_epoch 51 \
	   --learning_rate 0.001 \
	   --groupdro_stepsize 0.001 \
	   --lab_split "$ratio" 
done    


