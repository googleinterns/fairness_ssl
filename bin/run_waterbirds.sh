######### Worstoff DRO Method ###########
run_worstoffdro () {
    lr=0.0001
    wd=1.0
    param=0.001
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'Waterbirds' \
	   --model_type 'resnet50' \
	   --method 'worstoffdro' \
	   --optimizer 'SGD' \
	   --learning_rate "$lr" \
	   --noflag_saveckpt \
	   --batch_size 128 \
	   --num_epoch 300 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --worstoffdro_stepsize "$param" \
	   --worstoffdro_marginals=.53,.25,.07,.15 \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Group DRO Method ###########
run_groupdro () {
    lr=0.00001
    wd=0.1
    param=0.001
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'Waterbirds' \
	   --model_type 'resnet50' \
	   --method 'groupdro' \
	   --optimizer 'SGD' \
	   --learning_rate "$lr" \
	   --noflag_saveckpt \
	   --batch_size 128 \
	   --num_epoch 300 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --groupdro_stepsize "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Group DRO Oracle Method ###########
run_groupdro_oracle () {
    lr=0.00001
    wd=1.0
    param=0.001
    lab_split=1.0
    python train_and_eval_loop.py \
	   --dataset 'Waterbirds' \
	   --model_type 'resnet50' \
	   --method 'groupdro' \
	   --optimizer 'SGD' \
	   --learning_rate "$lr" \
	   --noflag_saveckpt \
	   --batch_size 128 \
	   --num_epoch 300 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --groupdro_stepsize "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Unsup DRO Method ###########
run_unsupdro () {
    lr=0.0001
    wd=0.1
    param=0.0001
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'Waterbirds' \
	   --model_type 'resnet50' \
	   --method 'unsupdro' \
	   --optimizer 'SGD' \
	   --learning_rate "$lr" \
	   --noflag_saveckpt \
	   --batch_size 128 \
	   --num_epoch 300 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --unsupdro_eta "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### ERM Method ###########
run_erm () {
    lr=0.0001
    wd=0.1
    python train_and_eval_loop.py \
	   --dataset 'Waterbirds' \
	   --model_type 'resnet50' \
	   --method 'erm' \
	   --optimizer 'SGD' \
	   --learning_rate "$lr" \
	   --noflag_saveckpt \
	   --batch_size 128 \
	   --num_epoch 300 \
	   --weight_decay "$wd"  \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

run_worstoffdro;
run_groupdro;
run_groupdro_oracle;
run_unsupdro;
run_erm;
