python train_and_eval_loop.py \
       --dataset 'Adult2' \
       --model_type 'fullyconn' \
       --latent_dim 32 \
       --method 'eiil' \
       --optimizer 'Adam' \
       --eiil_refmodel_epochs 5 \
       --eiil_phase1_steps 10 \
       --eiil_phase1_lr 0.1 \
       --eiil_phase2_penalwt 191258 \
       --eiil_phase2_annliter 190 \
       --learning_rate 0.0001 \
       --batch_size 128 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.001  \
       --lab_split 1.0 \
       --noshuffle_train \
       --gpu_ids '0' 
       
