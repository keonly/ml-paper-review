import argparse
import torch


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(
           description='Trains a CIFAR Classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--dataset',
            type=str,
            default='cifar10',
            choices=['cifar10', 'cifar100'],
            help='Choose between CIFAR-10, CIFAR-100.')
        parser.add_argument(
            '--model',
            '-m',
            type=str,
            default='wrn',
            choices=['wrn', 'allconv', 'densenet', 'resnext'],
            help='Choose architecture.')
        # Optimization options
        parser.add_argument(
            '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
        parser.add_argument(
            '--learning-rate',
            '-lr',
            type=float,
            default=0.1,
            help='Initial learning rate.')
        parser.add_argument(
            '--batch-size', '-b', type=int, default=128, help='Batch size.')
        parser.add_argument('--eval-batch-size', type=int, default=1000)
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
        parser.add_argument(
            '--decay',
            '-wd',
            type=float,
            default=0.0005,
            help='Weight decay (L2 penalty).')
        # WRN Architecture options
        parser.add_argument(
            '--layers', default=40, type=int, help='total number of layers')
        parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
        parser.add_argument(
            '--droprate', default=0.0, type=float, help='Dropout probability')
        # AugMix options
        parser.add_argument(
            '--mixture-width',
            default=3,
            type=int,
            help='Number of augmentation chains to mix per augmented example')
        parser.add_argument(
            '--mixture-depth',
            default=-1,
            type=int,
            help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
        parser.add_argument(
            '--aug-severity',
            default=3,
            type=int,
            help='Severity of base augmentation operators')
        parser.add_argument(
            '--no-jsd',
            '-nj',
            action='store_true',
            help='Turn off JSD consistency loss.')
        parser.add_argument(
            '--all-ops',
            '-all',
            action='store_true',
            help='Turn on all operations (+brightness,contrast,color,sharpness).')
        # Checkpointing options
        parser.add_argument(
            '--save',
            '-s',
            type=str,
            default='./snapshots',
            help='Folder to save checkpoints.')
        parser.add_argument(
            '--resume',
            '-r',
            type=str,
            default='',
            help='Checkpoint path for resume / test.')
        parser.add_argument('--evaluate', action='store_true', help='Eval only.')
        parser.add_argument(
            '--print-freq',
            type=int,
            default=50,
            help='Training loss print frequency (batches).')
        # Acceleration
        parser.add_argument(
            '--num-workers',
            type=int,
            default=4,
            help='Number of pre-fetching threads.')

    def parse_args(self):
        self.args = self.parser.parse_args()
        self.args.device = torch.device('cuda:{}'.format(self.args.gpu_ids[0])) if self.args.gpu_ids else torch.device('cpu')
        return self.args

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')