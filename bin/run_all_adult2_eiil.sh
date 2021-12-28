for param1 in 0.001 0.01 0.0001
do
    for param2 in 200000 20000 2000000
    do
	for sd in 42 43 44
	do
	    python train_and_eval_loop.py \
		   --dataset 'Adult2' \
		   --model_type 'fullyconn' \
		   --latent_dim 32 \
		   --method 'eiil' \
		   --optimizer 'Adam' \
		   --eiil_refmodel_epochs 201 \
		   --eiil_phase1_steps 10000 \
		   --eiil_phase1_lr "$param1" \
		   --eiil_phase2_method 'eiil_phase2_irm' \
		   --eiil_phase2_penalwt "$param2" \
		   --eiil_phase2_annliter 90 \
		   --learning_rate 0.00001 \
		   --batch_size 128 \
		   --noflag_saveckpt \
		   --num_epoch 201 \
		   --weight_decay 0.001 \
		   --noshuffle_train \
		   --seed "$sd" \
		   --ckpt_prefix "results_all" \
		   --flag_run_all \
		   --gpu_ids '0,1,2,3'
	done
    done
done
       
