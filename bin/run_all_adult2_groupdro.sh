for lr in 0.01 0.001
do	  
    for sd in 42 43 44
    do	  
	python train_and_eval_loop.py \
	       --dataset 'Adult2' \
	       --method 'groupdro' \
	       --model_type 'fullyconn' \
	       --latent_dim 32 \
	       --optimizer 'Adam' \
	       --batch_size 128 \
	       --noflag_saveckpt \
	       --num_epoch 2 \
	       --learning_rate "$lr" \
	       --weight_decay 0.01 \
	       --groupdro_stepsize 0.001 \
	       --gpu_ids '5' \
	       --seed "$sd" \
	       --ckpt_prefix 'results_all' \
	       --flag_run_all
    done
done
			   
