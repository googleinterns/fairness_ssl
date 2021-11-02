prog_ep1en6 () {
eps=1e-6
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
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.25,.07,.15 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon2" \
       --flag_run_all \
       --gpu_ids '4'
}
prog_ep1en5 () {
eps=1e-5
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
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.25,.07,.15 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon2" \
       --flag_run_all \
       --gpu_ids '5'
}
prog_ep1en4 () {
eps=1e-4
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
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.25,.07,.15 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon2" \
       --flag_run_all \
       --gpu_ids '6'
}
prog_ep2en0 () {
eps=2.0
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
       --worstoffdro_stepsize 0.001 \
       --worstoffdro_lambda 1.0 \
       --worstoffdro_latestart 0 \
       --worstoffdro_marginals=.53,.25,.07,.15 \
       --epsilon="$eps" \
       --ckpt_prefix "results_epsilon2" \
       --flag_run_all \
       --gpu_ids '7'
}
prog_ep1en6 & prog_ep1en5 & prog_ep1en4 & prog_ep2en0
