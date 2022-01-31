import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # ------ model settings ------ #
        self.parser.add_argument("--in_channels", default=3, type=int, help="Number of channels of input")
        self.parser.add_argument("--patch_size", default=16, type=int, help="patch size")
        self.parser.add_argument("--emb_size", default=768, type=int, help="embedding size")
        self.parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
        self.parser.add_argument("--layers", default=12, type=int, help="number of mlpmixerblock")
        self.parser.add_argument("--num_classes", default=10, type=int, help="number of classes of training data")
        self.parser.add_argument("--tokens_mlp_dim", default=384, type=int, help="mlp hidden dimension for token mixing layer")
        self.parser.add_argument("--channels_mlp_dim", default=3072, type=int, help="mlp hidden dimension for channel mixing layer")

        # -------- training settings -------- #
        self.parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written.")
        self.parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
        self.parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for dataloader")
        self.parser.add_argument("--eval_step", default=100, type=int, help="Run prediction on validation set every so many steps.")
        self.parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
        self.parser.add_argument("--momentum", default=0.9, type=float, help="The momentum for SGD.")
        self.parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
        self.parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training epochs to perform.")
        self.parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--net', type=str, default='Mixer', help="model name for training")

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
