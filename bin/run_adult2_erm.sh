python train_and_eval_loop.py \
       --dataset 'Adult2' \
       --method 'erm' \
       --model_type 'fullyconn' \
       --latent_dim 32 \
       --optimizer 'Adam' \
       --batch_size 128 \
       --noflag_saveckpt \
       --noflag_usegpu \
       --num_epoch 201 \
       --learning_rate 0.0001 \
       --weight_decay 0.01 

