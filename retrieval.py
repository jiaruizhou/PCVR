# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.custom_datasets import Finetune_Dataset_All
import models_mae
import models_vit
from engine_finetune import train_one_epoch, get_encoded


def get_args_parser0():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    # Optimizer parameters
    

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./outputs/output_dir_retrieval/pretrained', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_retrieval', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--data', default="", type=str)
    parser.add_argument('--mask', default=0, type=int)
    parser.add_argument('--topk', default=5, type=int)
    # parser.add_argument('--data', default="", type=str)
    parser.add_argument('--kmer', default=5, type=int)  # input size 2**k x 2**k
    parser.add_argument('--tax_rank', default="phylum")  #['superkingdom', 'kingdom', 'phylum', 'family']
    parser.add_argument('--unknown_loss',default=0,type=int)
    parser.add_argument('--class_weights',default=0,type=int)
    parser.add_argument('--focal_loss',default=0,type=int)
    return parser


def get_args_parser1():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N', help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs/output_dir_retrieval/pretrained', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs/output_dir_retrieval/pretrained', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--pred', action='store_true', help='Perform evaluation only')

    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # my settings
    parser.add_argument('--data', default="", type=str)
    parser.add_argument('--only', default=0, type=int)
    parser.add_argument('--mask', default=0, type=int)
    parser.add_argument('--topk', default=5, type=int)
    # parser.add_argument('--data', default="", type=str)
    parser.add_argument('--kmer', default=5, type=int)  # input size 2**k x 2**k
    parser.add_argument('--tax_rank', default="phylum")  #['superkingdom', 'kingdom', 'phylum', 'family']
    parser.add_argument('--unknown_loss',default=0,type=int)
    parser.add_argument('--class_weights',default=0,type=int)
    parser.add_argument('--focal_loss',default=0,type=int)
    return parser


def main0(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str("7")
    # os.environ["SLURM_PROCID"] = "1"
    if args.data == "all":
        args.data_path = ["./data/" + data_name + "/" for data_name in ["similar", "non_similar", "final"]]
    args.data_path="./data/"+args.data
    args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_db = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="train")
    dataset_q = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="test")

    # dataset_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_db = torch.utils.data.SequentialSampler(dataset_db)
        sampler_q = torch.utils.data.SequentialSampler(dataset_q)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_db = torch.utils.data.DataLoader(
        dataset_db,
        sampler=sampler_db,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_q = torch.utils.data.DataLoader(dataset_q, sampler=sampler_q, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    # define the model
    if "pretrained" in args.output_dir:
        model = models_vit.__dict__[args.model](#"vit_large_patch16"
            args,
            num_classes=5,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    with torch.no_grad():
        db_features, db_labels = get_encoded(args, data_loader_db, model, device)
        q_features, q_labels = get_encoded(args, data_loader_q, model, device)
        torch.save(db_features, args.output_dir + "db_glb.npy")
        torch.save(db_labels, args.output_dir + "db_y_glb.npy")
        torch.save(q_features, args.output_dir + "q_glb.npy")
        torch.save(q_labels, args.output_dir + "q_y_glb.npy")

def main1(args,output_dir):

  # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    if not args.pred:
        args.data_path = "./data/" + args.data + "/"

    output_dir += "/{}".format(args.data)
    args.log_dir = output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # import csv
    # from util.tax_entry import new_genus_dict,new_phyl_dict,new_supk_dict,genus_dict,phyl_dict,supk_dict
    # test_genus=torch.load("./data/final/test_cls_phylum.npy")
    # train_genus=torch.load("./data/final/train_cls_phylum.npy")
    # test_genus=['unknown' if i not in phyl_dict.keys() else i for i in test_genus]
    # train_genus=['unknown' if i not in phyl_dict.keys() else i for i in train_genus]
    # print(phyl_dict.values())
    # for cls in phyl_dict.keys():
        
    #     with open("./final_phylum_count2.csv", 'a+') as f:
    #         csv_write = csv.writer(f)
    #         csv_write.writerow([cls,train_genus.count(cls),test_genus.count(cls)])
    # return
    cudnn.benchmark = True
    dataset_db = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="train")
    dataset_q = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="test")

    # dataset_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_db = torch.utils.data.SequentialSampler(dataset_db)
        sampler_q = torch.utils.data.SequentialSampler(dataset_q)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_db = torch.utils.data.DataLoader(
        dataset_db,
        sampler=sampler_db,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_q = torch.utils.data.DataLoader(dataset_q, sampler=sampler_q, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    # define the model
    if "pretrained" in args.output_dir:
        model = models_vit.__dict__[args.model](#"vit_large_patch16"
            args,
            num_classes=5,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    
    model_without_ddp = model

    import util.lr_decay as lrd
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    with torch.no_grad():
        db_features, db_labels = get_encoded(args, data_loader_db, model, device)
        q_features, q_labels = get_encoded(args, data_loader_q, model, device)
        torch.save(db_features, output_dir +"/db_glb.npy")
        torch.save(db_labels, output_dir + "/db_y_glb.npy")
        torch.save(q_features, output_dir+ "/q_glb.npy")
        torch.save(q_labels, output_dir + "/q_y_glb.npy")

import torch.nn.functional as F


def retreival(args, qvecs, dbvecs, q_label, db_label, topk):
    batch_size = 100
    qvecs = F.normalize(qvecs, dim=-1).squeeze(1)
    dbvecs = F.normalize(dbvecs, dim=-1).squeeze(1)
    start_time = time.time()
    batch_size = min(batch_size, qvecs.size(0))
    if qvecs.size(0) % batch_size == 0:
        num = qvecs.size(0) // batch_size
    else:
        num = qvecs.size(0) // batch_size + 1
    print('>> Total iteration:{}'.format(num))

    results=[[],[],[]]
    with torch.no_grad():
        for i in range(num):
            start = i * batch_size
            end = min(start + batch_size, qvecs.size(0))
            query = qvecs[start:end]
            score = torch.einsum("bd,kd->bk", query, dbvecs)
            _, topk_indice = torch.topk(score, topk, dim=-1)
            results[0].extend(db_label[:, rank_dict['supk']][topk_indice])
            
            results[1].extend(db_label[:, rank_dict['phyl']][topk_indice])
            results[2].extend(db_label[:, rank_dict['genus']][topk_indice])

            if (i + 1) % 4000 == 0 or (i + 1) == qvecs.size(0):
                print('\r>>>> {}/{} done...'.format(i + 1, num), end='')

    
    torch.save(results,args.output_dir+"/{}/results.npy".format(args.data))
    # accuracy
    acc1_1,acc5_1=retreival_acc(torch.stack(results[0]), q_label[:, rank_dict['supk']], tk=(1, 5))
    print("Acc@1 {} Acc@5 {} on {}".format(acc1_1, acc5_1, "supk"))

    acc1_2,acc5_2=retreival_acc(torch.stack(results[1]), q_label[:, rank_dict['phyl']], tk=(1, 5))
    print("Acc@1 {} Acc@5 {} on {}".format(acc1_2, acc5_2, "phyl"))

    acc1_3,acc5_3=retreival_acc(torch.stack(results[2]), q_label[:, rank_dict['genus']], tk=(1, 5))
    print("Acc@1 {} Acc@5 {} on {}".format( acc1_3 ,  acc5_3 , "genus"))
    
    # avep
    from util.metrics import macro_average_precision_for_retrieval
    avep_1,_,_=macro_average_precision_for_retrieval(torch.stack(results[0])[:,0], q_label[:, rank_dict['supk']])
    print("Avep {} {}".format(avep_1, "supk"))
    avep_2,_,_=macro_average_precision_for_retrieval(torch.stack(results[1])[:,0], q_label[:, rank_dict['phyl']])
    print("Avep {} {}".format(avep_2, "phyl"))
    avep_3,_,_=macro_average_precision_for_retrieval(torch.stack(results[2])[:,0], q_label[:, rank_dict['genus']])
    print("Avep {} {}".format(avep_3, "genus"))
    
    
    write_log(args, [args.kmer, args.data, "supk", acc1_1.item(),acc5_1.item(),avep_1.item()])
    write_log(args, [args.kmer, args.data, "phyl", acc1_2.item(),acc5_2.item(),avep_2.item()])
    write_log(args, [args.kmer, args.data, "genus",  acc1_3.item(), acc5_3.item(),avep_3.item()])


def retreival_acc(output, target, tk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(tk), output.size()[1])
    batch_size = target.size(0)
    pred = output.t()  # [topk,batch]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # torch.any(torch.eq(True, correct, dim=1))  #pred.eq(target.reshape(1, -1).expand_as(pred))
    return [torch.any(correct[:min(k, maxk)], dim=0).float().sum(0) * 100. / batch_size for k in tk]

from sklearn.metrics import confusion_matrix, precision_score
def macro_average_precision(y_pred, y_true):
    classes = set(list(y_true.cpu().numpy()))  #.union(set(y_pred.numpy()))
    # 初始化宏平均精度
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    macro_avg_f1 = 0.0
    # 计算每个类别的精度并累加
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        # 计算混淆矩阵
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()

        # 计算精度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # 累加精度
        macro_avg_precision += precision
        macro_avg_recall += recall
        macro_avg_f1 += f1

    # 计算宏平均精度
    macro_avg_precision /= len(classes)  #ttp / (ttp + tfp)
    macro_avg_recall /= len(classes)
    macro_avg_f1 /= len(classes)
    return macro_avg_precision * 100.0, macro_avg_recall * 100, macro_avg_f1 * 100

import os
import csv

def create_csv(args):
    path = args.output_dir+"/retrieval.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ["kmer", "dataset", "tax_rank", "Acc@1", "Acc@5","avep"]  # "AveP", "AveR", "AveF", "loss" "eval_loss", "eval_Acc", "test_loss",
        csv_write.writerow(data_row)


def write_log(args, log):
    if os.path.exists(args.output_dir+ "/retrieval.csv") is False:
        create_csv(args)
    with open(args.output_dir+ "/retrieval.csv", 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(log)


rank_dict = {"supk": 0, "phyl": 1, "genus": 2}
if __name__ == '__main__':
    
    args = get_args_parser1()
    args = args.parse_args()
    args.output_dir=args.output_dir+"/"+args.tax_rank
    print(args.output_dir+"/{}".format(args.data))
    if not os.path.exists(args.output_dir+ "/{}".format(args.data) + "/db_glb.npy"):
        main1(args,args.output_dir)
    dbvecs = torch.load(args.output_dir + "/{}".format(args.data) + "/db_glb.npy")
    print(dbvecs[0].shape)
    db_label = torch.load(args.output_dir +"/{}".format(args.data) + "/db_y_glb.npy")
    qvecs = torch.load(args.output_dir +"/{}".format(args.data) + "/q_glb.npy")
    q_label = torch.load(args.output_dir +"/{}".format(args.data) + "/q_y_glb.npy")
    print("retrieval on {}".format(args.data))
    retreival(args, qvecs, dbvecs, q_label, db_label, args.topk)

