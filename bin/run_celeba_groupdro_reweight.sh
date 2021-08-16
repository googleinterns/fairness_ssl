python train_and_eval_loop.py \
       --dataset 'CelebA' \
       --model_type 'resnet50' \
       --method 'groupdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 50 \
       --weight_decay 0.1  \
       --flag_reweight \
       --gpu_ids '3'
       
