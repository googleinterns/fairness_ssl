for param1 in 0.001 0.01 0.0001
do
    for param2 in 200000 20000 2000000
    do
	for sd in 42 43 44
	do
	    python train_and_eval_loop.py \
		   --dataset 'CMNIST' \
		   --model_type 'mlp' \
		   --latent_dim 390 \
		   --method 'eiil' \
		   --optimizer 'Adam' \
		   --eiil_refmodel_epochs 501 \
		   --eiil_phase1_steps 10000 \
		   --eiil_phase1_lr "$param1" \
		   --eiil_phase2_method 'eiil_phase2_irm' \
		   --eiil_phase2_penalwt "$param2" \
		   --eiil_phase2_annliter 190 \
		   --learning_rate 0.0001 \
		   --batch_size 39640 \
		   --noflag_saveckpt \
		   --num_epoch 501 \
		   --weight_decay 0.01 \
		   --noshuffle_train \
		   --seed "$sd" \
		   --ckpt_prefix "results_all" \
		   --flag_run_all \
		   --gpu_ids '0,1,2,3'
	done
    done
done
       
