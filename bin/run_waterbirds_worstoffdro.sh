python train_and_eval_loop.py \
       --dataset 'Waterbirds' \
       --model_type 'resnet50' \
       --method 'worstoffdro' \
       --optimizer 'SGD' \
       --learning_rate 1e-5 \
       --noflag_saveckpt \
       --batch_size 128 \
       --num_epoch 300 \
       --weight_decay 1.0  \
       --lab_split 0.3 \
       --worstoffdro_lambda 0.01 \
       --worstoffdro_latestart 100 \
       --gpu_ids '6'
       
