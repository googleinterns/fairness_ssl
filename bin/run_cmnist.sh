######### Worstoff DRO Method ###########
run_worstoffdro () {
    lr=0.0001
    wd=0.01
    param=0.0001
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'worstoffdro' \
	   --optimizer 'Adam' \
	   --learning_rate "$lr" \
	   --batch_size 39640 \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --worstoffdro_stepsize "$param" \
	   --worstoffdro_marginals=.53,.40,.07 \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Group DRO Method ###########
run_groupdro () {
    lr=0.001
    wd=0.01
    param=0.001
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'groupdro' \
	   --optimizer 'Adam' \
	   --learning_rate "$lr" \
	   --batch_size 39640 \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --groupdro_stepsize "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Group DRO Oracle Method ###########
run_groupdro_oracle () {
    lr=0.0001
    wd=0.001
    param=0.001
    lab_split=1.0
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'groupdro' \
	   --optimizer 'Adam' \
	   --learning_rate "$lr" \
	   --batch_size 39640 \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --groupdro_stepsize "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### Unsup DRO Method ###########
run_unsupdro () {
    lr=0.00001
    wd=0.001
    param=0.4
    lab_split=0.1
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'unsupdro' \
	   --optimizer 'Adam' \
	   --learning_rate "$lr" \
	   --batch_size 39640 \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --unsupdro_eta "$param" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

######### ERM Method ###########
run_erm () {
    lr=0.001
    wd=0.01
    python train_and_eval_loop.py \
	   --dataset 'CMNIST' \
	   --model_type 'mlp' \
	   --latent_dim 390 \
	   --method 'erm' \
	   --optimizer 'Adam' \
	   --learning_rate "$lr" \
	   --batch_size 39640 \
	   --noflag_saveckpt \
	   --num_epoch 501 \
	   --weight_decay "$wd"  \
	   --lab_split "$lab_split" \
	   --ckpt_prefix "results" \
	   --flag_run_all
}

run_worstoffdro;
run_groupdro;
run_groupdro_oracle;
run_unsupdro;
run_erm;
