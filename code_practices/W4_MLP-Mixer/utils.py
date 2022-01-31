import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
import os
import torchvision.models as models
from model import MlpMixer
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.


def select_model(args):
    
    if args.net == 'Mixer':
        net = MlpMixer(args.in_channels, 
                       args.emb_size, 
                       args.num_classes, 
                       args.patch_size, 
                       args.img_size, 
                       args.layers, 
                       args.tokens_mlp_dim, 
                       args.channels_mlp_dim)
        
    elif args.net == 'ResNet50':
        net = models.resnet50(num_classes=args.n_classes)
        
    elif args.net == 'ResNet101':
        net = models.resnet101(num_classes=args.n_classes)
        
    else:
        print('net name is wrong')

    return net

        
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def save_model(args, model):
    model_ckpt = model.state_dict()
    ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
    torch.save(model_ckpt, ckpt_path)
    

def get_dataloader(args):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR10(root="./data",
                                train=True,
                                download=True,
                                transform=transform_train)
    testset = datasets.CIFAR10(root="./data",
                                train=False,
                                download=True,
                                transform=transform_test)
            
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers)
    
    return trainloader, testloader