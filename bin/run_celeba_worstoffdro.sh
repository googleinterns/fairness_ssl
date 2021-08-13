python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 150 \
       --weight_decay 0.1  \
       --lab_split 0.3 \
       --worstoffdro_lambda 0.01 \
       --worstoffdro_latestart 50 \
       --gpu_ids '5' 
       
