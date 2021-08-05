python train_and_eval_loop.py \
       --dataset 'Waterbirds' \
       --model_type 'resnet50' \
       --method 'erm' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 300 \
       --weight_decay 1.0 
