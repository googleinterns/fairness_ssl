for lr in 0.001 0.0001 0.00001
do
    for wd in 0.1 0.01 0.001
    do
        for param in 0.01 0.001 0.0001
        do
	    for sd in 43 44
	    do
		python train_and_eval_loop.py \
                       --dataset 'Adult2' \
                       --method 'worstoffdro' \
                       --model_type 'fullyconn' \
                       --latent_dim 32 \
                       --optimizer 'Adam' \
                       --batch_size 128 \
                       --noflag_saveckpt \
                       --num_epoch 201 \
                       --learning_rate "$lr" \
                       --weight_decay "$wd" \
                       --lab_split 0.1 \
                       --worstoffdro_stepsize "$param" \
                       --worstoffdro_lambda 1.0 \
                       --worstoffdro_latestart 0 \
                       --worstoffdro_marginals=.63,.27,.05,.05 \
		       --seed "$sd" \
                       --ckpt_prefix "results_all" \
                       --flag_run_all &
	    done
        done
    done
done
wait
