import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import seed_everything, select_model, WarmupLinearSchedule, save_model, get_dataloader
from options import Options
import os


def valid(args, global_step):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    
    for step, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            logits = model(inputs)

            eval_loss = criterion(logits, labels)

            _, preds = logits.max(1)
            
            test_loss += eval_loss.item()
            correct += preds.eq(labels).sum()
    
    test_loss = test_loss / len(testloader)
    accuracy = correct.float() / len(testloader.dataset)
    print('\nEval) global step: {}, Average loss: {:.4f}, Eval Accuracy: {:.4f}'.format(
        global_step, 
        test_loss,
        accuracy))

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    writer.add_scalar("test/loss", scalar_value=test_loss, global_step=global_step)
    
    return accuracy


def train(args):
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(trainloader,
                                desc="Training (X / X Steps) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)
        
        for step, data in enumerate(epoch_iterator):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss.item())
            )
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            if global_step % args.eval_step == 0 :
                accuracy = valid(args, global_step)
                if best_acc < accuracy:
                    save_model(args, model)
                    best_acc = accuracy
                model.train()

            if global_step % args.num_steps == 0:
                break
        if global_step % args.num_steps == 0:
                break

    print("Best Accuracy: ", best_acc)
    
if __name__ == '__main__':
    args = Options().parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.net))
    seed_everything(args.seed)
    
    trainloader, testloader = get_dataloader(args)
    model = select_model(args).to(args.device)
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    
    train(args) 