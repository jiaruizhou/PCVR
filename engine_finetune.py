# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import csv
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from timm.data import Mixup
from util.metrics import accuracy, macro_average_precision, plot_roc, plot_pr_curve, micro_average_precision, weighted_macro_average_precision, weighted_micro_average_precision, new_macro_average_precision
import util.misc as misc
import util.lr_sched as lr_sched
# from torchmetrics import Accuracy
import os
from util.tax_entry import genus_dict

import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self,  gamma=2,alpha=.75,reduction="mean"):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = alpha
            self.gamma = gamma
            self.cross_entropy=torch.nn.CrossEntropyLoss()
            self.reduction=reduction
    def forward(self, inputs, targets):
            eps= 1e-7
            CE_loss = F.cross_entropy(inputs, targets, reduction='none',weight=self.alpha)     #self.cross_entropy(inputs, targets)#   
            targets = targets.type(torch.long)        
            pt = torch.exp(-CE_loss+eps)        
            focal_loss = (1-pt)**self.gamma * CE_loss     
            # if self.alpha is not None:
            #     alpha_weights = torch.ones_like(inputs)  # 初始化为1
            #     alpha_weights[:, self.alpha.size(0)] = self.alpha
            #     focal_loss = alpha_weights * focal_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
 
        
class Focal_Loss(torch.nn.Module):
    
    def __init__(self,alpha=0.75,gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma=gamma
        self.weight=alpha
    def forward(self,preds,labels):
        eps=1e-7
        preds = F.softmax(preds, dim=-1)
        y_pred =preds.view((preds.size()[0],preds.size()[1])) #B*C*H*W->B*C*(H*W)

        target=labels.view(y_pred.size(),-1) #B*C*H*W->B*C*(H*W)

        ce=-1*torch.log(y_pred+eps)*target
        floss=torch.pow((1-y_pred),self.gamma)*ce
        floss=torch.mul(floss,self.weight)
        floss=torch.sum(floss,dim=1)
        return floss.mean()#torch.mean(floss)

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None, model_without_ddp=None, 
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch="err")
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        targets1, targets2, targets3 = targets
        targets1, targets2, targets3 = targets1.to(device, non_blocking=True), targets2.to(device, non_blocking=True), targets3.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            
            outputs1, outputs2, outputs3,_ = model(samples)
            loss1 = criterion(outputs1, targets1)
            loss2 = criterion(outputs2, targets2)
            loss3 = criterion(outputs3, targets3)
            loss = (args.weight1*loss1 + args.weight2*loss2 + args.weight3*loss3) 
            

        loss_value = loss.item()
        if args.only==0:
            loss_value1 = loss1.item()
            loss_value2 = loss2.item()
            loss_value3 = loss3.item()
        
            if not math.isfinite(loss_value):
                print("Loss is {} Loss1 is {} Loss2 is {} Loss3 is {}, stopping training".format(loss_value,loss_value1,loss_value2,loss_value3))
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch="err")
                sys.exit(0)
        elif not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch="err")
            sys.exit(0)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if args.only==0:
            loss_value_reduce1 = misc.all_reduce_mean(loss_value1)
            loss_value_reduce2 = misc.all_reduce_mean(loss_value2)
            loss_value_reduce3 = misc.all_reduce_mean(loss_value3)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            if args.only==0:
                log_writer.add_scalar('loss1', loss_value_reduce1, epoch_1000x)
                log_writer.add_scalar('loss2', loss_value_reduce2, epoch_1000x)
                log_writer.add_scalar('loss3', loss_value_reduce3, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from util.tax_entry import genus_dict


@torch.no_grad()
def evaluate(args, data_loader, model, device):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode

    model.eval()
    if args.unknown_loss :# and args.data=="final"
        outputs = [[], [], [],[]]
    else:
        outputs = [[], [], [],[]]
    targets = [[], [], []]
    for batch in metric_logger.log_every(data_loader, 40, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target1, target2, target3 = target
        targets[0].extend(target1)
        targets[1].extend(target2)
        targets[2].extend(target3)
        target1, target2, target3 = target1.to(device, non_blocking=True), \
            target2.to(device, non_blocking=True), target3.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            outputs[0].extend(output[0])
            outputs[1].extend(output[1])
            outputs[2].extend(output[2])
            loss1 = criterion(output[0], target1)
            loss2 = criterion(output[1], target2)

            loss = (criterion(output[0], target1)+ \
                criterion(output[1], target2)+criterion(output[2], target3))/3
                
        metric_logger.update(loss=loss.item())
        metric_logger.meters['loss0'].update(loss1.item())
        metric_logger.meters['loss1'].update(loss2.item())

    metric_logger.synchronize_between_processes()

    rank_dicts = {"supk": 0, "phyl": 1, "genus": 2}
    # classes = [5, 44, 156]

    for i, rank in enumerate(rank_dicts.keys()):
        print("*********** For {} ***********".format(rank))
        acc1, acc5 = accuracy(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu(), topk=(1, 5))
        macro_avep, aver, avef = macro_average_precision(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        micro_avep = micro_average_precision(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        weighted_macro = weighted_macro_average_precision(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        weighted_micro = weighted_micro_average_precision(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        # ACC = Accuracy(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        print(" RESULTS acc1:{} acc5:{} \nmacro_avep:{} micro_avep:{} weighted macro_avep:{} weighted micro_avep:{} \naver:{} avef:{}".format(acc1, acc5, macro_avep, micro_avep, weighted_macro, weighted_micro, aver, avef))
        
        metric_logger.meters['avep{}'.format(i)].update(macro_avep)
        metric_logger.meters['aver{}'.format(i)].update(aver)
        metric_logger.meters['avef{}'.format(i)].update(avef)
        metric_logger.meters['acc1_{}'.format(i)].update(acc1.item())
        metric_logger.meters['acc5_{}'.format(i)].update(acc5.item())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def get_encoded(args, data_loader, model, device):
    model.eval()
    features = []
    labels1 = []
    labels2 = []
    labels3 = []
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'get_latent'
    # switch to evaluation mode
    model.eval()
    # flat=torch.nn.Flatten(start_dim=0,end_dim=1)
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        idx = batch[1]
        images = images.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            _, _, _, latent = model(images)
            features.extend(latent.unsqueeze(1).cpu())
            labels1.extend(idx[0])
            labels2.extend(idx[1])
            labels3.extend(idx[2])
    features = torch.stack(features)
    labels1 = torch.stack(labels1).unsqueeze(1)
    labels2 = torch.stack(labels2).unsqueeze(1)
    labels3 = torch.stack(labels3).unsqueeze(1)
    labels = torch.cat([labels1, labels2, labels3], dim=1)
    return features, labels
