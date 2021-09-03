"""Training and evaluation binary that is implemented as a custom loop."""


"""Train and evaluation loop."""


from absl import app
from absl import flags

from erm import ERM
from groupdro import GroupDRO
from unsupdro import UnsupDRO
from worstoffdro import WorstoffDRO

from util.utils import HParams

import os
import pdb

# Dataset.
flags.DEFINE_enum(name='dataset', default='Adult',
                  enum_values=['Adult', 'German', 'Waterbirds', 'AdultConfounded', 'CelebA', 'CMNIST', 'Adult2'],
                  help='dataset.')
flags.DEFINE_integer(name='dataseed', default=0,
                     help='random seed for dataset construction.')
# Model.
flags.DEFINE_enum(name='model_type', default='fullyconn',
                  enum_values=['fullyconn', 'mlp', 'resnet50'], help='model type.')
flags.DEFINE_integer(name='latent_dim', default=64,
                     help='latent dims for fully connected network')
flags.DEFINE_bool(name='flag_usegpu', default=True, help='To use GPU or not')
flags.DEFINE_string(name='gpu_ids', default='0,1,2,3,4,5,6,7', help='gpu_ids')
flags.DEFINE_bool(name='flag_saveckpt', default=True, help='To save checkpoints or not')

# Optimization.
flags.DEFINE_enum(name='method', default='erm',
                  enum_values=['erm', 'groupdro', 'unsupdro', 'worstoffdro'],
                  help='method.')
flags.DEFINE_integer(name='seed', default=42, help='random seed for optimizer.')
flags.DEFINE_enum(name='optimizer', default='Adam',
                  enum_values=['Adam', 'SGD'],
                  help='optimization method.')
flags.DEFINE_enum(name='scheduler', default='',
                  enum_values=['', 'linear', 'step', 'cosine', 'plateau'],
                  help='learning rate scheduler.')
flags.DEFINE_float(name='learning_rate', default=0.0001,
                   help='learning rate.')
flags.DEFINE_float(name='weight_decay', default=0.00001,
                   help='L2 weight decay.')
flags.DEFINE_boolean(name='resume', default=True,
                     help='resume training from checkpoint if true.')
flags.DEFINE_integer(name='batch_size', default=64, help='batch size.')
flags.DEFINE_integer(name='num_epoch', default=100, help='number of epoch.')

# Misc.
flags.DEFINE_string(name='ckpt_prefix', default='results',
                    help='path to checkpoint.')
flags.DEFINE_string(name='ckpt_path', default='',
                    help='path to save model.')

# Experiment modes.
flags.DEFINE_bool(name='flag_debug', default=False, help='Enables Debug Mode')
flags.DEFINE_bool(name='flag_singlebatch', default=False, help='Enables Debug Mode')
flags.DEFINE_bool(name='flag_run_all', default=False, help='Enables hyper-param search mode')

# DRO hyper-params
flags.DEFINE_float(name='groupdro_stepsize', default=0.01, help='step size.')
flags.DEFINE_bool(name='flag_reweight', default=False, help='To reweight groups for waterbirds dataset')
flags.DEFINE_float(name='unsupdro_eta', default=0.9, help='step size.')
flags.DEFINE_float(name='worstoffdro_stepsize', default=0.01, help='step size for parameter update')
flags.DEFINE_float(name='worstoffdro_lambda', default=0.01,
                   help='regularization for labelled and unlabelled.')
flags.DEFINE_integer(name='worstoffdro_latestart', default=0,
                     help='epoch at which the unlab loss will be added.')
flags.DEFINE_list(name='worstoffdro_marginals', default=['.25','.25','.25','.25'],
                  help='Marginal probabilities for each group.')

# SSL Parameters
flags.DEFINE_float(name='lab_split', default=1.0,
                   help='The ratio of labelled samples in the dataset')



FLAGS = flags.FLAGS


def get_trainer(hparams):
    """Gets trainer for the required method."""

    if hparams.method == 'erm':
        trainer = ERM(hparams)
    elif hparams.method == 'groupdro':
        trainer = GroupDRO(hparams)
    elif hparams.method == 'unsupdro':
        trainer = UnsupDRO(hparams)
    elif hparams.method == 'worstoffdro':
        trainer = WorstoffDRO(hparams)
    else:
        raise NotImplementedError

    return trainer


def main(unused_argv):
    
    # Set parameters.
    hparams = HParams({
            flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
        })

    # Select the GPU machine to run the experiment
    if hparams.flag_usegpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu_ids # do not import torch
  
    # Obtain the code for necessary method
    trainer = get_trainer(hparams)

    # Run the training routine
    trainer.get_config()
    trainer.train()


if __name__ == '__main__':
    app.run(main)

