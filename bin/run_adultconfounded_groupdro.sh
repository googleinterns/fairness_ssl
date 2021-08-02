for wd in 1.0 0.1 0.01
do
    for stepsz in 0.1 0.01 0.001 0.0001	      
    do 
	python train_and_eval_loop.py \
	       --dataset 'AdultConfounded' \
	       --method 'groupdro' \
	       --noflag_usegpu \
	       --noflag_saveckpt \
	       --num_epoch 201 \
	       --learning_rate 0.001 \
	       --groupdro_stepsize "$stepsz" \
	       --weight_decay "$wd" 
    done
done


