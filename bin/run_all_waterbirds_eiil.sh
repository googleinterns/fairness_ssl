python train_and_eval_loop.py \
       --dataset 'Waterbirds' \
       --model_type 'resnet50' \
       --method 'eiil' \
       --optimizer 'SGD' \
       --eiil_refmodel_epochs 1 \
       --eiil_phase1_steps 20000 \
       --eiil_phase1_lr 0.01 \
       --eiil_phase2_method 'eiil_phase2_groupdro' \
       --learning_rate 0.00001 \
       --batch_size 128 \
       --noflag_saveckpt \
       --num_epoch 300 \
       --weight_decay 1.0 \
       --noshuffle_train \
       --ckpt_prefix "results_all" \
       --flag_run_all \
       --gpu_ids '0,1,2,3' 
       
