import os
import glob, pylab, pandas as pd
import pydicom, numpy as np
import random
import shutil
import json
import time
import argparse
import torchvision


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from pathlib import Path
from dataset import get_rsna_data, get_imagenet_data
from loss import FocalLoss, F1_Loss
# from train import one_epoch_train, model_eval
from resnet import resnet50


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://211.184.186.64:16023', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--saved-dir', default='trained_models/', type=str,
                    help='distributed backend')
parser.add_argument('--isrsna', action='store_true',
                    help='imagenet or rsna')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def main() :
    args = parser.parse_args()
    print('isrsna', args.isrsna)
    
    if args.saved_dir and not os.path.isdir(args.saved_dir):
        mkdir(args.saved_dir)    
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])   
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('num of gpu', ngpus_per_node)
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print('single node training')
        main_worker(args.gpu, ngpus_per_node, args)    
        
def main_worker(gpu, ngpus_per_node, args) :
    
    args.gpu = gpu
    print('gpu', args.gpu)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)    

    print('isrsna', args.isrsna)
    if args.isrsna  :
        train_loader ,test_loader = get_rsna_data(args)
    else :
        train_loader ,test_loader = get_imagenet_data(args)
    
#     model= torchvision.models.resnet50(pretrained=True)
    backbone_model= torchvision.models.resnet50(pretrained=True)
    model = resnet50(pretrained=False)
    print(model.conv1.weight[0,0,0])
    model.load_state_dict(backbone_model.state_dict(), strict=False)
    
    criterion = nn.CrossEntropyLoss()
    
    print(model.conv1.weight[0,0,0])    
    if args.isrsna :
        if os.path.isfile('trained_models/imagenet/model_best.pt') :
            checkpoint = torch.load('trained_models/imagenet/model_best.pt')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print('load pretrained model with imagenet locally')
            
        num_ftrs = model.fc1.in_features
        model.fc1 = nn.Linear(num_ftrs, 2) # target label is 2
        criterion = FocalLoss(alpha=0.97, gamma=2, reduce=True)
        # criterion = F1_Loss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.to(args.gpu)

    criterion = criterion.to(args.gpu)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#     scheduler = lr_scheduler.LambdaLR(
#         optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1)
#     )    
    
    train_loss = []
    val_acc = []
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, args)  

        train_loss.append(epoch_loss)
        val_acc.append(acc1)
        print('************train_loss {} val_acc {}*************'.format(epoch_loss, acc1))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

    #     if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    #             and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
#             'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args)

    
def save_checkpoint(state, is_best, args, filename='checkpoint.pt'):
    args = parser.parse_args()
    torch.save(state, args.saved_dir+filename)
    if is_best:
        shutil.copyfile(args.saved_dir+filename, args.saved_dir+'model_best.pt')        
    
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_loss = 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

	 # compute output
        output = model(images)
        loss = criterion(output, target)
        total_loss += loss.item()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(train_loader)/20) == 0:
            progress.display(i)
    return total_loss/len(train_loader)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    device = torch.device('cuda')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True) 

	     # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#             if i % (len(val_loader)/10) == 0:
#                 progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
    
def mkdir(path):
    try:
        os.makedirs(path)
        print(path+' is maded')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise    
    
if __name__ == '__main__':
#     args = parser.parse_args()
#     print(args.isrsna)
    main()
