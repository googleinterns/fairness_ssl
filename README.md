# Invariant Learning with Partial Group Labels


-   **This is not an officially supported Google product.**

## Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env
. env/bin/activate
pip install -r requirements.txt
```

-   The code has been tested on Ubuntu 18.04 with CUDA 9.1.


## Datasets

Download or generate the datasets as follows:

-   Waterbirds: Download a [tarball](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) of the dataset. Place the contents under `data/dataset/celeba_dataset` directory.
-   CMNIST: Download the MNIST dataset from this [website](http://yann.lecun.com/exdb/mnist/). Place the contents in `data/dataset/mnist_dataset/` directory.
-   Adult: The dataset can be downloaded from [UCI repository](https://archive.ics.uci.edu/ml/datasets/adult). Place the contents in `data/dataset/raw/` directory. 
-   CelebA: Download the dataset from [kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset). Place it under `data/datasets/` directory.

## Running experiment
Several run scripts are provided in the `bin/*` directory. The files `bin/run_*` indicate a single hyper-param run whereas `bin/run_all_*` indicate all hyper-param runs. 

An example command with relevant flags is provided below. Details on each flag is available in the file `train_and_eval_loop.py`.


    ```bash
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
        --worstoffdro_marginals=.53,.25,.07,.15 \
        --epsilon=0.001 \
        --ckpt_prefix "results" \
        --flag_run_all 
     ```
