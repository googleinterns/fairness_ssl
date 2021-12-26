python train_and_eval_loop.py \
       --dataset 'CMNIST' \
       --model_type 'mlp' \
       --latent_dim 390 \
       --method 'eiil' \
       --optimizer 'Adam' \
       --eiil_refmodel_epochs 501 \
       --eiil_phase1_steps 10000 \
       --eiil_phase1_lr 0.001 \
       --eiil_phase2_penalwt 191258 \
       --eiil_phase2_annliter 190 \
       --learning_rate 0.0001 \
       --batch_size 39640 \
       --noflag_saveckpt \
       --num_epoch 501 \
       --weight_decay 0.001  \
       --lab_split 0.1 \
       --gpu_ids '0' 
       
