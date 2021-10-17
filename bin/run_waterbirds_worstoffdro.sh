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
       --lab_split 0.1 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.25,.07,.15 \
       --gpu_ids '2'  \
       --get_dataset_from_lmdb \
       --flag_debug
       
