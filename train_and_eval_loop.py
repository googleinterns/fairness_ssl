"""Training and evaluation binary that is implemented as a custom loop."""


"""Train and evaluation loop."""


from absl import app
from absl import flags

from erm import ERM
from groupdro import GroupDRO
from unsupdro import UnsupDRO

from util.utils import HParams

import pdb

# Dataset.
flags.DEFINE_enum(name='dataset', default='Adult',
                  enum_values=['Adult', 'German', 'Compas', 'LawSchools'],
                  help='dataset.')
flags.DEFINE_integer(name='dataseed', default=0,
                     help='random seed for dataset construction.')
# Model.
flags.DEFINE_enum(name='model_type', default='fullyconn',
                  enum_values=['fullyconn', 'resnet'], help='model type.')
flags.DEFINE_integer(name='latent_dim', default=64,
                     help='latent dims for fully connected network')
flags.DEFINE_bool(name='flag_usegpu', default=True, help='To use GPU or not')
flags.DEFINE_bool(name='flag_saveckpt', default=True, help='To save checkpoints or not')

# Optimization.
flags.DEFINE_enum(name='method', default='erm',
                  enum_values=['erm', 'groupdro', 'unsupdro'],
                  help='method.')
flags.DEFINE_integer(name='seed', default=42, help='random seed for optimizer.')
flags.DEFINE_enum(name='optimizer', default='Adam',
                  enum_values=['Adam'],
                  help='optimization method.')
flags.DEFINE_enum(name='scheduler', default='',
                  enum_values=['', 'linear', 'step', 'cosine'],
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

# Debug mode.
flags.DEFINE_bool(name='flag_debug', default=False, help='Enables Debug Mode')

# DRO hyper-params
flags.DEFINE_float(name='groupdro_stepsize', default=0.01,
                   help='soft penalty step size.')
flags.DEFINE_float(name='unsupdro_eta', default=0.9,
                   help='soft penalty step size.')


FLAGS = flags.FLAGS


def get_trainer(hparams):
    """Gets trainer for the required method."""

    if hparams.method == 'erm':
        trainer = ERM(hparams)
    elif hparams.method == 'groupdro':
        trainer = GroupDRO(hparams)
    elif hparams.method == 'unsupdro':
        trainer = UnsupDRO(hparams)
    else:
        raise NotImplementedError

    return trainer


def main(unused_argv):

    # Set parameters.
    hparams = HParams({
            flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
        })
    
    # Obtain the code for necessary method
    trainer = get_trainer(hparams)

    # Run the training routine
    trainer.get_config()
    trainer.train()


if __name__ == '__main__':
    app.run(main)

