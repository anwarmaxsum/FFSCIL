# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np
import random
from timm.utils import accuracy
from timm.optim import create_optimizer

import dualpromptlib.utils as utils

import copy 


from piplib.xpip_engine_utils import *

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    iter=0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):

        if iter == 0:
            print(list(set(target.tolist())))
            print(input.shape)

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        iter = iter + 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_with_available_classes(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, available_classes=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    if available_classes==None :
        if task_id > 0:
            min_c = args.base_classes + ((task_id-1)*args.fs_classes)
            max_c = args.base_classes + ((task_id)*args.fs_classes)
        else:
            min_c = 0
            max_c = args.base_classes 
        all_classes = [item for item in range(min_c, max_c)]
        print("all classes:")
        print(all_classes)
        available_classes = random.sample(all_classes, args.available_classes)
    
    
    print("available classes:")
    print(available_classes)
    # print(original_model)

    iter=0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        # if task_id > 0:
        # print(set(target))
        tmp = copy.deepcopy(target)
        filter = tmp.apply_(lambda x: x in available_classes).bool()
        # print(target)
        # print(filter)
        input=input[filter.tolist()]
        target=target[filter.tolist()]
        # print(input.shape)

        #Not any data in the filtered
        if target.shape[0]==0:
            continue

        if iter == 0:
            print(list(set(target.tolist())))

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']

            else:
                cls_features = None
        
        feat_mix, y_a, y_b, rep_lam, mixup_indexes = mixup_batch_representations(cls_features,target)

        # res = model.forward_features(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        # feat_mix, y_a, y_b, rep_lam, mixup_indexes = mixup_batch_representations(res['x'],target)
        # res['x'] = feat_mix

        # output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
      
        output = model(input, task_id=task_id, cls_features=feat_mix, train=set_training_mode)
        # output = model.forward_head(res)

        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


        loss = rep_lam * criterion(logits, y_a) + (1-rep_lam) * criterion(logits, y_b)
        # loss = criterion(logits, target) # base criterion (CrossEntropyLoss)



        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        iter = iter + 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_one_epoch_with_available_classes_v2(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, available_classes=None, 
                    global_prototype=None, global_prototype_var=None,  args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    if available_classes==None :
        if task_id > 0:
            min_c = args.base_classes + ((task_id-1)*args.fs_classes)
            max_c = args.base_classes + ((task_id)*args.fs_classes)
        else:
            min_c = 0
            max_c = args.base_classes 

        all_classes = [item for item in range(min_c, max_c)]
        print("all classes:")
        print(all_classes)
        available_classes = random.sample(all_classes, args.available_classes)
    
    print("available classes:")
    print(available_classes)
    # print(original_model)

    iter=0
    proto = None
    proto_label = None
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        # if task_id > 0:
        # print(set(target))
        tmp = copy.deepcopy(target)
        filter = tmp.apply_(lambda x: x in available_classes).bool()
        # print(target)
        # print(filter)
        input=input[filter.tolist()]
        target=target[filter.tolist()]
        # print(input.shape)

        #Not any data in the filtered
        if target.shape[0]==0:
            continue

        if iter == 0:
            print(list(set(target.tolist())))

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
       
            else:
                cls_features = None
        
        if iter == 0:        
            proto = torch.tensor(np.array(list(global_prototype.values())), dtype=input.dtype).to(device, non_blocking=True)
            proto_std = torch.square(torch.tensor(np.array(list(global_prototype_var.values())), dtype=input.dtype).to(device, non_blocking=True))
            proto_label = torch.tensor(np.array(list(global_prototype.keys())),dtype=target.dtype).to(device, non_blocking=True)

            d =  torch.rand(proto.shape, dtype=proto.dtype).to(device, non_blocking=True) / 3.0 
            proto1 = proto + (d*proto_std)
            proto2 = proto + (-1.0*d*proto_std)

            proto = torch.cat([proto,proto1,proto2],dim=0)
            proto_label = torch.cat([proto_label,proto_label,proto_label],dim=0)

        # output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        # output = model.forward_with_proto(input, task_id=task_id, cls_features=cls_features, proto=proto, train=set_training_mode)
        # logits = output['logits']
        feat_mix, y_a, y_b, rep_lam, mixup_indexes = mixup_batch_representations(cls_features,target)
        output = model.forward_with_proto(input, task_id=task_id, cls_features=feat_mix, proto=proto, train=set_training_mode)
        logits = output['logits']

        target_a = torch.cat([y_a,proto_label],dim=0)
        target_b = torch.cat([y_b,proto_label],dim=0)
        # print(logits.shape)
        # print(target.shape)

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        # loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        # print(target)
        loss = rep_lam * criterion(logits, target_a) + (1-rep_lam) * criterion(logits, target_b)



        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1, acc5 = accuracy(logits, target_a, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        iter = iter + 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_pertask(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, args = None,):
    

    # create matrix to save end-of-task accuracies 
    print("Train per Task [TaskID]: " + str(task_id+1))
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # for task_id in range(args.num_tasks):
    # Transfer previous learned prompt params to the new prompt
    if args.prompt_pool and args.shared_prompt_pool:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            if (prev_end > args.size) or (cur_end > args.size):
                pass
            else:
                for n in range(len(models)): 
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            # models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            # models[n].e_prompt.prompt.grad.zero_()
                            models[n].e_prompt.prompt[cur_idx] = models[n].e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].parameters()
                
    # Transfer previous learned prompt param keys to the new prompt
    if args.prompt_pool and args.shared_prompt_key:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            for n in range(len(models)):
                with torch.no_grad():
                    if args.distributed:
                        # models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        # models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 
    # Create new optimizer for each task to clear optimizer status
    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])

    available_classes_all=[]

    
    # Set available classes
    if task_id > 0:
        min_c = args.base_classes + ((task_id-1)*args.fs_classes)
        max_c = args.base_classes + ((task_id)*args.fs_classes)
    else:
        min_c = 0
        max_c = args.base_classes 
    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    for epoch in range(args.epochs):
        for n in range(len(models)):            

            train_stats = train_one_epoch_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                        available_classes=available_classes_all[n], args=args,)

    clients_prototype = []
    clients_prototype_var = []
    for n in range(len(models)):            
        prototype_dict, prototype_var =  get_prototype_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                    data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                    device=device, epoch=args.epochs, max_norm=args.clip_grad, 
                                    set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                    available_classes=available_classes_all[n], args=args,)
        
        clients_prototype.append(prototype_dict)
        clients_prototype_var.append(prototype_var)

    
    return clients_prototype, clients_prototype_var


def train_pertask_v2(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, global_prototype=None, global_prototype_var=None, args = None,):
    

    # create matrix to save end-of-task accuracies 
    print("Train per Task [TaskID]: " + str(task_id+1))
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # for task_id in range(args.num_tasks):
    # Transfer previous learned prompt params to the new prompt
    if args.prompt_pool and args.shared_prompt_pool:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            if (prev_end > args.size) or (cur_end > args.size):
                pass
            else:
                for n in range(len(models)): 
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            # models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            # models[n].e_prompt.prompt.grad.zero_()
                            models[n].e_prompt.prompt[cur_idx] = models[n].e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].parameters()
                
    # Transfer previous learned prompt param keys to the new prompt
    if args.prompt_pool and args.shared_prompt_key:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            for n in range(len(models)):
                with torch.no_grad():
                    if args.distributed:
                        # models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        # models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 
    # Create new optimizer for each task to clear optimizer status
    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])

    available_classes_all=[]

    
    # Set available classes
    if task_id > 0:
        min_c = args.base_classes + ((task_id-1)*args.fs_classes)
        max_c = args.base_classes + ((task_id)*args.fs_classes)
    else:
        min_c = 0
        max_c = args.base_classes 
    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    for epoch in range(args.epochs):
        for n in range(len(models)):            

            train_stats = train_one_epoch_with_available_classes_v2(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                        available_classes=available_classes_all[n], global_prototype=global_prototype, global_prototype_var=global_prototype_var, args=args,)
        

    clients_prototype = []
    clients_prototype_var = []
    for n in range(len(models)):            
        prototype_dict, prototype_var =  get_prototype_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                    data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                    device=device, epoch=args.epochs, max_norm=args.clip_grad, 
                                    set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                    available_classes=available_classes_all[n], args=args,)
        clients_prototype.append(prototype_dict)
        clients_prototype_var.append(prototype_var)

    
    return clients_prototype, clients_prototype_var



def train_fs_one_epoch_with_available_classes(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, available_classes=None, 
                    global_prototype=None, global_prototype_var=None,  args = None,):
    

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    if available_classes==None :
        if task_id > 0:
            min_c = args.base_classes + ((task_id-1)*args.fs_classes)
            max_c = args.base_classes + ((task_id)*args.fs_classes)
        else:
            min_c = 0
            max_c = args.base_classes 

        all_classes = [item for item in range(min_c, max_c)]
        print("all classes:")
        print(all_classes)
        available_classes = random.sample(all_classes, args.available_classes)
    

    available_classes = sorted(available_classes)
    print("available classes to be trained:")
    print(available_classes)
    # print(original_model)
    label_converter={}
    label_reverser={}
    l_con=0
    for l in sorted(available_classes):
        label_converter[l] = l_con
        label_reverser[l_con] = l
        l_con = l_con + 1


    iter=0
    proto = None
    proto_label = None

   
    # print(optimizer.param_groups)

    # print(optimizer.state_dict())
    # for n, p in optimizer.state_dict():
        # print(n)

    model.train(set_training_mode)
    original_model.eval()
    x_entropy = torch.nn.CrossEntropyLoss()



    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        tmp = copy.deepcopy(target)
        filter = tmp.apply_(lambda x: x in available_classes).bool()

        input=input[filter.tolist()]
        target=target[filter.tolist()]

        target1 = copy.deepcopy(target).apply_(lambda x: label_converter[x]).cuda()

        if target.shape[0]==0:
            continue

        if iter == 0:
            print(list(set(target.tolist())))

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()
        # print(input.shape)
        # print(target.shape)
        # print(target)
        # print(target1)


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
       
            else:
                cls_features = None
    
        # cls_features1 = torch.unsqueeze(cls_features, 0).cuda()
        # # target1 = target
        # logit_querys, classifiers = model.fs_head(cls_features1, cls_features1, target1, target1, args.available_fs_classes, args.fs_shots)
        # print(classifiers[-1].shape)
        # # print(len(logit_querys))
        # # print(logit_querys[-1].shape)
        # # print(len(classifiers))
        # # print(classifiers[-1].shape)
        # logit_query = logit_querys[-1]
        # loss = x_entropy(logit_query.reshape(-1,  args.available_fs_classes), target1.reshape(-1))
        # acc = count_accuracy(logit_query.reshape(-1, args.available_fs_classes), target1.reshape(-1))
        # print("Check acc: "+str(acc))

        if(iter ==0):
            ava_proto_lable, unava_proto_lable = [], []
            if (global_prototype is not None):
                all_proto_label =  list(global_prototype.keys())
                ava_proto_label = [x for x in available_classes if x in all_proto_label]
                unava_proto_label = [x for x in available_classes if x not in all_proto_label]

                # proto = torch.tensor(np.array(list(global_prototype.values())), dtype=input.dtype).to(device, non_blocking=True)
                # proto_std = torch.square(torch.tensor(np.array(list(global_prototype_var.values())), dtype=input.dtype).to(device, non_blocking=True))
                # proto_label = torch.tensor(np.array(list(global_prototype.keys())),dtype=target.dtype).to(device, non_blocking=True)
                proto = torch.tensor(np.array([global_prototype[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True)
                proto_std = torch.square(torch.tensor(np.array([global_prototype_var[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True))
                proto_label = torch.tensor(np.array(ava_proto_label),dtype=target.dtype).to(device, non_blocking=True)


        if(global_prototype is None):
            query_proto = model.forward_get_prelogits(input, task_id, cls_features, train=True)
            query_label = target1
        else:
            if(iter == 0):
                query_proto = proto
                query_label = proto_label
                if(len(unava_proto_lable) > 0):
                    for c in unava_proto_lable:
                        proto_c = torch.mean(cls_features[target==c],dim=0)
                        query_proto = torch.cat((query_proto,proto_c),dim=0)
                        query_label = torch.cat((query_proto,torch.tensor(c).cuda()),dim=0)

                query_proto_kshot = []
                query_label_kshot = []
                for i in range(args.fs_shots):
                    query_proto_kshot.append(query_proto)
                    query_label_kshot.append(query_label)

                query_label_kshot = torch.cat(query_label_kshot,dim=0)
                query_proto_kshot = torch.cat(query_proto_kshot,dim=0)


        logit_query, opt_proto = model.forward_fs(input,  target1, query_proto_kshot, query_label_kshot, args.available_fs_classes, args.fs_shots, task_id, cls_features, True)
        print(logit_query.shape)
        print(opt_proto.shape)



        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1, acc5 = accuracy(logit_query.reshape(-1,  args.available_fs_classes), target1.reshape(-1), topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        iter = iter + 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_fs_one_epoch_with_available_classes_v2(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, available_classes=None, 
                    global_prototype=None, global_prototype_var=None, all_global_prototype=None, all_global_prototype_var=None,  args = None,):
    

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    if available_classes==None :
        if task_id > 0:
            min_c = args.base_classes + ((task_id-1)*args.fs_classes)
            max_c = args.base_classes + ((task_id)*args.fs_classes)
        else:
            min_c = 0
            max_c = args.base_classes 

        all_classes = [item for item in range(min_c, max_c)]
        print("all classes:")
        print(all_classes)
        available_classes = random.sample(all_classes, args.available_classes)
    

    available_classes = sorted(available_classes)
    print("available classes to be trained:")
    print(available_classes)
    # print(original_model)
    label_converter={}
    label_reverser={}
    l_con=0
    for l in sorted(available_classes):
        label_converter[l] = l_con
        label_reverser[l_con] = l
        l_con = l_con + 1


    iter=0
    proto = None
    proto_label = None

   
    # print(optimizer.param_groups)

    # print(optimizer.state_dict())
    # for n, p in optimizer.state_dict():
        # print(n)

    model.train(set_training_mode)
    original_model.eval()
    x_entropy = torch.nn.CrossEntropyLoss()

    optimal_proto = {}




    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        tmp = copy.deepcopy(target)
        filter = tmp.apply_(lambda x: x in available_classes).bool()

        input=input[filter.tolist()]
        target=target[filter.tolist()]

        target1 = copy.deepcopy(target).apply_(lambda x: label_converter[x]).cuda()

        if target.shape[0]==0:
            continue

        if iter == 0:
            print(list(set(target.tolist())))

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()
        # print(input.shape)
        # print(target.shape)
        # print(target)
        # print(target1)


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
       
            else:
                cls_features = None
    
        # cls_features1 = torch.unsqueeze(cls_features, 0).cuda()
        # # target1 = target
        # logit_querys, classifiers = model.fs_head(cls_features1, cls_features1, target1, target1, args.available_fs_classes, args.fs_shots)
        # print(classifiers[-1].shape)
        # # print(len(logit_querys))
        # # print(logit_querys[-1].shape)
        # # print(len(classifiers))
        # # print(classifiers[-1].shape)
        # logit_query = logit_querys[-1]
        # loss = x_entropy(logit_query.reshape(-1,  args.available_fs_classes), target1.reshape(-1))
        # acc = count_accuracy(logit_query.reshape(-1, args.available_fs_classes), target1.reshape(-1))
        # print("Check acc: "+str(acc))

        if(iter ==0):
            ava_proto_lable, unava_proto_lable = [], []
            if (all_global_prototype is not None):
                all_proto_label =  list(global_prototype.keys())
                ava_proto_label = [x for x in available_classes if x in all_proto_label]
                unava_proto_label = [x for x in available_classes if x not in all_proto_label]

                # print("available and unavailable proto:")
                # print(ava_proto_label)
                # print(unava_proto_label)
                
                # proto = torch.tensor(np.array(list(global_prototype.values())), dtype=input.dtype).to(device, non_blocking=True)
                # proto_std = torch.square(torch.tensor(np.array(list(global_prototype_var.values())), dtype=input.dtype).to(device, non_blocking=True))
                # proto_label = torch.tensor(np.array(list(global_prototype.keys())),dtype=target.dtype).to(device, non_blocking=True)

                # proto = torch.tensor(np.array([global_prototype[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True)
                # proto_std = torch.square(torch.tensor(np.array([global_prototype_var[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True))
                # proto_label = torch.tensor(np.array(ava_proto_label),dtype=target.dtype).to(device, non_blocking=True)

                proto = torch.tensor(np.array([all_global_prototype[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True)
                proto_std = torch.square(torch.tensor(np.array([all_global_prototype_var[x] for x in ava_proto_label]), dtype=input.dtype).to(device, non_blocking=True))
                proto_label = torch.tensor(np.array(ava_proto_label),dtype=target.dtype).to(device, non_blocking=True)


                # if (all_global_prototype is not None):
                all_proto = {}
                for k in all_global_prototype.keys():
                    all_proto[k] = torch.tensor(np.array(all_global_prototype[k]), dtype=input.dtype).to(device, non_blocking=True)


        if(global_prototype is None):
            query_proto = model.forward_get_prelogits(input, task_id, cls_features, train=True)
            query_label = target1
        else:
            if(iter == 0):
                query_proto = proto
                query_label = proto_label
                if(len(unava_proto_label) > 0):
                    for c in unava_proto_label:
                        proto_c = torch.mean(cls_features[target==c],dim=0).unsqueeze(0)
                        # print(proto_c.shape)
                        query_proto = torch.cat((query_proto,proto_c),dim=0)
                        query_label = torch.cat((query_label,torch.tensor([c]).cuda()))

                query_proto_kshot = []
                query_label_kshot = []
                for i in range(args.fs_shots):
                    query_proto_kshot.append(query_proto)
                    query_label_kshot.append(query_label)

                query_label_kshot = torch.cat(query_label_kshot,dim=0)
                query_proto_kshot = torch.cat(query_proto_kshot,dim=0)

        # print(query_proto_kshot.shape)
        # print(query_label_kshot.shape)
        # print(input.shape)
        # print(target1.shape)
        # print(query_label_kshot)
        # print(target1)
        # print(target)


        output, logit_query, opt_proto = model.forward_fs(input,  target1, query_proto_kshot, query_label_kshot, args.available_fs_classes, args.fs_shots, task_id, cls_features, True)
        pre_logits = output['pre_logits']
        # print(pre_logits.shape)
        # print(logit_query.shape)
        # print(opt_proto.shape)

        pre_logits, y_a, y_b, rep_lam, mixup_indexes = mixup_batch_representations(pre_logits,target)


        for k in range(opt_proto.shape[0]):
            optimal_proto[label_reverser[k]] = opt_proto[k]

        logits = None
        isFirst = True
        for j in range(0,pre_logits.shape[0]):
            logit = classify_with_proto(pre_logits[j], all_proto, int(max(target).item())+1)
            if isFirst:
                logits = logit
                isFirst= False
            else:
                logits=torch.cat((logits,logit), dim=0)


        print(logits.shape)
        print(target.shape)
        # loss = criterion(logits, target)
        loss = rep_lam * criterion(logits, y_a) + (1-rep_lam) * criterion(logits, y_b)

        # loss_q = x_entropy(logit_query.reshape(-1,  args.available_fs_classes), target1)
        # print("Support proto CE Loss: "+str(loss))
        # print("Query proto CE Loss: "+str(loss_q))
        # # if torch.isfinite(loss_q) and not torch.isnan(loss_q):
        # if not math.isfinite(loss.item()):
        #     loss = torch.tensor(0.0).cuda()
        # if not math.isfinite(loss_q.item()):
        #     loss_q = torch.tensor(0.0).cuda()
        # print("Add Query Loss")
        # loss += 0.5*loss_q 


        # loss = x_entropy(logits, target)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))


        # loss += x_entropy(logit_query.reshape(-1,  args.available_fs_classes), target1.reshape(-1))
        # q_acc = count_accuracy(logit_query.reshape(-1, args.available_fs_classes), target1.reshape(-1))
        # print("Query Acc: "+str(q_acc))

        if args.pull_constraint and 'reduce_sim' in output:
            if torch.isfinite(output['reduce_sim']) and not torch.isnan(output['reduce_sim']) :
                print("Add match loss")
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        # acc1, acc5 = accuracy(logit_query.reshape(-1,  args.available_fs_classes), target1.reshape(-1), topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        iter = iter + 1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, optimal_proto

def train_fs_pertask(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, global_prototype=None, global_prototype_var=None, args = None,):
    

    # create matrix to save end-of-task accuracies 
    print("Train per Task [TaskID]: " + str(task_id+1))
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # for task_id in range(args.num_tasks):
    # Transfer previous learned prompt params to the new prompt
    if args.prompt_pool and args.shared_prompt_pool:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            if (prev_end > args.size) or (cur_end > args.size):
                pass
            else:
                for n in range(len(models)): 
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            # models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            # models[n].e_prompt.prompt.grad.zero_()
                            models[n].e_prompt.prompt[cur_idx] = models[n].e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].parameters()
                
    # Transfer previous learned prompt param keys to the new prompt
    if args.prompt_pool and args.shared_prompt_key:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            for n in range(len(models)):
                with torch.no_grad():
                    if args.distributed:
                        # models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        # models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 

    #Swithc to NODE head instead of FC
    if(task_id > 0):
        for n in range(len(models)):
            model = models[n]
            if not model.entered_fs_task:
                model.init_fs_head(args.available_fs_classes)
                #Freeze FC head
                for n, p in model.named_parameters():
                    if n.startswith("head"):
                        p.requires_grad = False

                model.entered_fs_task = True
                if not args.reinit_optimizer:
                    optimizers[n] = create_optimizer(args, model)

    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])

    available_classes_all=[]
    # Set available classes
    if task_id > 0:
        min_c = args.base_classes + ((task_id-1)*args.fs_classes)
        max_c = args.base_classes + ((task_id)*args.fs_classes)
    else:
        min_c = 0
        max_c = args.base_classes 

    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    clients_optimal_proto = [[]]*len(models)
    for epoch in range(args.epochs):
        for n in range(len(models)):            

            train_stats, optimal_proto = train_fs_one_epoch_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                        available_classes=available_classes_all[n], global_prototype=global_prototype, global_prototype_var=global_prototype_var, args=args,)
            
            clients_optimal_proto[n] = optimal_proto      

    clients_prototype = []
    clients_prototype_var = []
    for n in range(len(models)):            
        prototype_dict, prototype_var =  get_prototype_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                    data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                    device=device, epoch=args.epochs, max_norm=args.clip_grad, 
                                    set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                    available_classes=available_classes_all[n], args=args,)
        clients_prototype.append(prototype_dict)
        clients_prototype_var.append(prototype_var)

    
    # return clients_prototype, clients_prototype_var
    return clients_optimal_proto, clients_prototype_var


def train_fs_pertask_v2(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, 
                    global_prototype=None, global_prototype_var=None, 
                    all_global_prototype=None, all_global_prototype_var=None, args = None,):
    

    # create matrix to save end-of-task accuracies 
    print("Train per Task [TaskID]: " + str(task_id+1))
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # for task_id in range(args.num_tasks):
    # Transfer previous learned prompt params to the new prompt
    if args.prompt_pool and args.shared_prompt_pool:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            if (prev_end > args.size) or (cur_end > args.size):
                pass
            else:
                for n in range(len(models)): 
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            # models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            # models[n].e_prompt.prompt.grad.zero_()
                            models[n].e_prompt.prompt[cur_idx] = models[n].e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].parameters()
                
    # Transfer previous learned prompt param keys to the new prompt
    if args.prompt_pool and args.shared_prompt_key:
        if task_id > 0:
            prev_start = (task_id - 1) * args.top_k
            prev_end = task_id * args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * args.top_k

            for n in range(len(models)):
                with torch.no_grad():
                    if args.distributed:
                        # models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        # models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 

    #Swithc to NODE head instead of FC
    if(task_id > 0):
        for n in range(len(models)):
            model = models[n]
            if not model.entered_fs_task:
                model.init_fs_head(args.available_fs_classes)
                #Freeze FC head
                for n, p in model.named_parameters():
                    if n.startswith("head"):
                        p.requires_grad = False

                for n, p in model.head.named_parameters():
                    # if n.startswith("head"):
                    p.requires_grad = False

                model.entered_fs_task = True
                if not args.reinit_optimizer:
                    optimizers[n] = create_optimizer(args, model)

    # if(task_id > 0):
    #     for n in range(len(models)):
    #         model = models[n]
    #         model.init_fs_head(args.available_fs_classes)
    #         #Freeze FC head
    #         for n, p in model.named_parameters():
    #             if n.startswith("head"):
    #                 p.requires_grad = False

    #         for n, p in model.head.named_parameters():
    #             # if n.startswith("head"):
    #             p.requires_grad = False

    #         model.entered_fs_task = True
               

    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])

    available_classes_all=[]
    # Set available classes
    if task_id > 0:
        min_c = args.base_classes + ((task_id-1)*args.fs_classes)
        max_c = args.base_classes + ((task_id)*args.fs_classes)
    else:
        min_c = 0
        max_c = args.base_classes 

    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    for epoch in range(args.epochs):
        for n in range(len(models)):            

            train_stats = train_fs_one_epoch_with_available_classes_v2(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                        available_classes=available_classes_all[n], global_prototype=global_prototype, global_prototype_var=global_prototype_var, 
                                        all_global_prototype=all_global_prototype, all_global_prototype_var=all_global_prototype_var,args=args,)
        

    clients_prototype = []
    clients_prototype_var = []
    for n in range(len(models)):            
        prototype_dict, prototype_var =  get_prototype_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                    data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                    device=device, epoch=args.epochs, max_norm=args.clip_grad, 
                                    set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                    available_classes=available_classes_all[n], args=args,)
        clients_prototype.append(prototype_dict)
        clients_prototype_var.append(prototype_var)

    
    return clients_prototype, clients_prototype_var




def generate_prototype_only(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, args = None,):
    
    # create matrix to save end-of-task accuracies 
    print("Generate Prototypes Only On [TaskID]: " + str(task_id+1))
    

    # Set available classes
    available_classes_all=[]

    if task_id > 0:
        min_c = args.base_classes + ((task_id-1)*args.fs_classes)
        max_c = args.base_classes + ((task_id)*args.fs_classes)
    else:
        min_c = 0
        max_c = args.base_classes 
    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    clients_prototype = []
    clients_prototype_var = []
    for n in range(len(models)):            
        prototype_dict, prototype_var =  get_prototype_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                    data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                    device=device, epoch=args.epochs, max_norm=args.clip_grad, 
                                    set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                    available_classes=available_classes_all[n], args=args,)
        
        clients_prototype.append(prototype_dict)
        clients_prototype_var.append(prototype_var)

    
    return clients_prototype, clients_prototype_var


def get_prototype_with_available_classes(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, available_classes=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    print("Generate prototype")
    if available_classes==None :
        if task_id > 0:
            min_c = args.base_classes + ((task_id-1)*args.fs_classes)
            max_c = args.base_classes + ((task_id)*args.fs_classes)
        else:
            min_c = 0
            max_c = args.base_classes 
        all_classes = [item for item in range(min_c, max_c)]
        # print("all classes:")
        # print(all_classes)
        available_classes = random.sample(all_classes, args.available_classes)
    
    print("available classes:")
    print(available_classes)
    # print(original_model)

    iter=0

    prototype_dict={}
    prototype_var={}

    # for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
    cl_data = [0 for i in range(max(available_classes)+1)]
    # print(cl_data)
    for input, target in data_loader:
        # if task_id > 0:
        # print(set(target))
        tmp = copy.deepcopy(target)
        filter = tmp.apply_(lambda x: x in available_classes).bool()
        # print(target)
        # print(filter)
        input=input[filter.tolist()]
        target=target[filter.tolist()]


        if target.shape[0]==0:
            continue

        if iter == 0:
            print(list(set(target.tolist())))

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # filter = sum(target==i for i in available_classes).bool()


        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        pre_logists = model.forward_get_prelogits(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)


        for c in available_classes:
            pl_class = pre_logists[target==c]
            if(pl_class.shape[0]==0):
                continue
            if(cl_data[c]==0):
                # prototype_dict[c] = torch.mean(pl_class, dim=0).cpu().detach().numpy()

                prototype_dict[c] = torch.mean(pl_class, dim=0).cpu().detach().numpy()
                prototype_var[c] = torch.var(pl_class, dim=0).cpu().detach().numpy()
                # cl_data[c]=cl_data[c]+pl_class.shape[0]
            else:
                # prototype_dict[c] = ((cl_data[c] * prototype_dict[c]) + (torch.mean(pl_class,dim=0).cpu().detach().numpy() * pl_class.shape[0]))/(cl_data[c]+pl_class.shape[0])

                mu1 =  prototype_dict[c]
                mu2 =  torch.mean(pl_class,dim=0).cpu().detach().numpy()
                var1 = prototype_var[c]
                var2 = torch.var(pl_class, dim=0).cpu().detach().numpy()
                n1 = cl_data[c]
                n2 = pl_class.shape[0]

                prototype_dict[c] = ((n1*mu1)+(n2*mu2))/(n1+n2)
                prototype_var[c] = ((((var1+(mu1*mu1))*n1) + ((var2+(mu2*mu2))*n2))/(n1+n2)) - (prototype_dict[c]*prototype_dict[c])


            cl_data[c]=cl_data[c]+pl_class.shape[0]
                # cl_data[c]=cl_data[c]+pl_class.shape[0]


        del pre_logists
        del input
        del target

        iter = iter + 1

    return prototype_dict, prototype_var



@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    true_labels=[]
    pred_labels=[]
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, true_labels, pred_labels



@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, test_sizes=None, args=None,):
    # stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    stat_matrix = np.zeros((3, task_id+1))

    test_sizes = np.array(test_sizes[0:task_id+1])
    # test_sizes = test_sizes[0:task_id+1]

    true_labels=[]
    pred_labels=[]
    for i in range(task_id+1):
        test_stats, true_labels_t, pred_labels_t = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        
        true_labels.extend(true_labels_t)
        pred_labels.extend(pred_labels_t)
    # avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    avg_stat = np.divide(np.sum(stat_matrix*test_sizes, axis=1), np.sum(test_sizes))
    twavg_stat = np.divide(np.sum(stat_matrix, axis=1),  task_id+1)

    if task_id > 0:
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = np.divide(np.sum(stat_matrix[:,1:]*test_sizes[1:], axis=1), np.sum(test_sizes[1:]))
    else :
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = stat_matrix[:,0]

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)


    return test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels
    

def evaluate_server_global_model(model, model_without_ddp, original_model, 
                    criterion, data_loader, optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, task_id=0, test_sizes=None, args = None,):
    
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    # acc_all_model=[]
    # for n in range(len(models)):  
    test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, test_sizes=test_sizes, args=args)


    # acc_global_model=np.array(avg_stat[0])
    # print("Task: [" +str(task_id+1)+"] Server Global Model Acc@1: " + str(acc_global_model))
    print('-----------------------------------------------------------------------------------------------------')
    result_str = "[Global Model Average Top1-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[0], base_classes_stat[0], novel_classes_stat[0], twavg_stat[0], (base_classes_stat[0]+novel_classes_stat[0])/2)
    print(result_str)
    result_str = "[Global Model Average Top5-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[1], base_classes_stat[1], novel_classes_stat[1], twavg_stat[1], (base_classes_stat[1]+novel_classes_stat[1])/2)
    print(result_str)
    result_str = "[Global Model Average TestLoss till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[2], base_classes_stat[2], novel_classes_stat[2], twavg_stat[2], (base_classes_stat[2]+novel_classes_stat[2])/2)
    print(result_str)

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    print("Confussion Matrix at task-"+str(task_id+1))
    print(conf_mat.tolist())




@torch.no_grad()
def evaluate2(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    true_labels=[]
    pred_labels=[]
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features,train=True)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, true_labels, pred_labels



@torch.no_grad()
def evaluate_till_now2(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, test_sizes=None, args=None,):
    # stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    stat_matrix = np.zeros((3, task_id+1))

    test_sizes = np.array(test_sizes[0:task_id+1])
    # test_sizes = test_sizes[0:task_id+1]

    true_labels=[]
    pred_labels=[]
    for i in range(task_id+1):
        test_stats, true_labels_t, pred_labels_t = evaluate2(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        
        true_labels.extend(true_labels_t)
        pred_labels.extend(pred_labels_t)
    # avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    avg_stat = np.divide(np.sum(stat_matrix*test_sizes, axis=1), np.sum(test_sizes))
    twavg_stat = np.divide(np.sum(stat_matrix, axis=1),  task_id+1)

    if task_id > 0:
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = np.divide(np.sum(stat_matrix[:,1:]*test_sizes[1:], axis=1), np.sum(test_sizes[1:]))
    else :
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = stat_matrix[:,0]

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)


    return test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels


# def classify_with_proto(data, paras):
#     scale_factor =  10
#     logits = scale_factor * torch.nn.functional.cosine_similarity(
#         data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
#         paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
#         dim=-1)
#     return logits


# def classify(data, paras):
#     scale_factor =  10
#     logits = scale_factor * torch.nn.functional.cosine_similarity(
#         data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
#         paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
#         dim=-1)
#     return logits
# def count_accuracy(logits, label):
#     pred = torch.argmax(logits, dim=1).view(-1)
#     label = label.view(-1)
#     accuracy = 100 * pred.eq(label).float().mean()
#     return accuracy

def evaluate_server_global_model2(model, model_without_ddp, original_model, 
                    criterion, data_loader, optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, task_id=0, test_sizes=None, args = None,):
    
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    # acc_all_model=[]
    # for n in range(len(models)):  
    test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels = evaluate_till_now2(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, test_sizes=test_sizes, args=args)


    # acc_global_model=np.array(avg_stat[0])
    # print("Task: [" +str(task_id+1)+"] Server Global Model Acc@1: " + str(acc_global_model))
    print('-----------------------------------------------------------------------------------------------------')
    result_str = "[Global Model Average Top1-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[0], base_classes_stat[0], novel_classes_stat[0], twavg_stat[0], (base_classes_stat[0]+novel_classes_stat[0])/2)
    print(result_str)
    result_str = "[Global Model Average Top5-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[1], base_classes_stat[1], novel_classes_stat[1], twavg_stat[1], (base_classes_stat[1]+novel_classes_stat[1])/2)
    print(result_str)
    result_str = "[Global Model Average TestLoss till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[2], base_classes_stat[2], novel_classes_stat[2], twavg_stat[2], (base_classes_stat[2]+novel_classes_stat[2])/2)
    print(result_str)

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    print("Confussion Matrix at task-"+str(task_id+1))
    print(conf_mat.tolist())




@torch.no_grad()
def evaluate3(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, 
            all_global_prototype=None, all_global_prototype_var=None,args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    true_labels=[]
    pred_labels=[]

    max_label = args.base_classes + task_id*args.fs_classes

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            # if (task_id ==0 or all_global_prototype==None):
            if (all_global_prototype==None):
                output = model(input, task_id=task_id, cls_features=cls_features,train=False)
                logits = output['logits']
            else:
                pre_logists = model.forward_get_prelogits(input, task_id=task_id, cls_features=cls_features, train=False)
                # all_proto = torch.tensor(np.array(list(all_global_prototype.values())), dtype=input.dtype).to(device, non_blocking=True)
                all_proto = {}
                for k in all_global_prototype.keys():
                    all_proto[k] = torch.tensor(np.array(all_global_prototype[k]), dtype=input.dtype).to(device, non_blocking=True)

                logits = None
                isFirst = True
                for j in range(0,pre_logists.shape[0]):
                    # logit = classify(pre_logists[j], all_proto)
                    # logit = classify_with_proto(pre_logists[j], all_proto)
                    logit = classify_with_proto(pre_logists[j], all_proto, max_label)
                    if isFirst:
                        logits = logit
                        isFirst= False
                    else:
                        logits=torch.cat((logits,logit), dim=0)
            #     print("Finish 1 batch testing")


            # print("Finish 1 set testing")
            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            # print("After masking")
            # print(logits.shape)
            # print(target.shape)
            # print(target)

            loss = criterion(logits, target)
            # print("After loss computng")
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            # print("After acc coputing")
            
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, true_labels, pred_labels


@torch.no_grad()
def evaluate3a(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, 
            all_global_prototype=None, all_global_prototype_var=None,args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    true_labels=[]
    pred_labels=[]

    max_label = args.base_classes + task_id*args.fs_classes
    all_proto = None
    is_all_proto_created = False

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            # if (task_id ==0 or all_global_prototype==None):
            if not is_all_proto_created:
                all_proto = {}
                for k in all_global_prototype.keys():
                    all_proto[k] = torch.tensor(np.array(all_global_prototype[k]), dtype=input.dtype).to(device, non_blocking=True)
                is_all_proto_created = False

            

            res = model.forward_features(input, task_id=task_id, cls_features=cls_features,train=False)
            pred_prompt_idx =  res['prompt_idx'][:,0].tolist()

            print("Check pred_prompt_idx")
            print(pred_prompt_idx)

            most_frequent_idx = max(pred_prompt_idx, key=pred_prompt_idx.count) 
            if(most_frequent_idx == 0):
                print("Classification with FC Layer")
                output = model.forward_head(res)
                # output = model(input, task_id=task_id, cls_features=cls_features,train=False)
                logits = output['logits']
            else:
                print("Classification with Prototoype")
                logits = None
                isFirst = True
                pre_logits = model.forward_head_prelogits(res)['pre_logits']

                for j in range(0,pre_logits.shape[0]):
                    # logit = classify(pre_logists[j], all_proto)
                    # logit = classify_with_proto(pre_logists[j], all_proto)
                    logit = classify_with_proto(pre_logits[j], all_proto, max_label)
                    if isFirst:
                        logits = logit
                        isFirst= False
                    else:
                        logits=torch.cat((logits,logit), dim=0)



            # if (all_global_prototype==None):
            #     output = model(input, task_id=task_id, cls_features=cls_features,train=False)
            #     logits = output['logits']
            # else:
            #     logits = None
            #     isFirst = True
            #     pre_logists = model.forward_get_prelogits(input, task_id=task_id, cls_features=cls_features, train=False)
            #     # all_proto = torch.tensor(np.array(list(all_global_prototype.values())), dtype=input.dtype).to(device, non_blocking=True)
                
            #     for j in range(0,pre_logists.shape[0]):
            #         # logit = classify(pre_logists[j], all_proto)
            #         # logit = classify_with_proto(pre_logists[j], all_proto)
            #         logit = classify_with_proto(pre_logists[j], all_proto, max_label)
            #         if isFirst:
            #             logits = logit
            #             isFirst= False
            #         else:
            #             logits=torch.cat((logits,logit), dim=0)
            # #     print("Finish 1 batch testing")


            # print("Finish 1 set testing")
            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            # print("After masking")
            # print(logits.shape)
            # print(target.shape)
            # print(target)

            loss = criterion(logits, target)
            # print("After loss computng")
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            # print("After acc coputing")
            
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, true_labels, pred_labels


@torch.no_grad()
def evaluate3b(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, 
            all_global_prototype=None, all_global_prototype_var=None,args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    true_labels=[]
    pred_labels=[]

    max_label = args.base_classes + task_id*args.fs_classes
    all_proto = None
    is_all_proto_created = False

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            # if (task_id ==0 or all_global_prototype==None):
            if not is_all_proto_created:
                all_proto = {}
                for k in all_global_prototype.keys():
                    all_proto[k] = torch.tensor(np.array(all_global_prototype[k]), dtype=input.dtype).to(device, non_blocking=True)
                is_all_proto_created = False

            

            res = model.forward_features(input, task_id=task_id, cls_features=cls_features,train=False)
            pred_prompt_idx =  res['prompt_idx'][:,0].tolist()

            print("Check pred_prompt_idx")
            print(pred_prompt_idx)

            most_frequent_idx = max(pred_prompt_idx, key=pred_prompt_idx.count) 
            
            output = model.forward_head(res)
            logits = output['logits'][:,0:max_label]

            logits2 = None
            isFirst = True
            pre_logits = model.forward_head_prelogits(res)['pre_logits']

            for j in range(0,pre_logits.shape[0]):
                logit = classify_with_proto(pre_logits[j], all_proto, max_label)
                if isFirst:
                    logits2 = logit
                    isFirst= False
                else:
                    logits2=torch.cat((logits2,logit), dim=0)

            logits=torch.nn.functional.softmax(logits, dim=1)
            logits2=torch.nn.functional.softmax(logits2, dim=1)

            for j in range(0,logits.shape[0]):
                if torch.max(logits[j,:]) < torch.max(logits2[j,:]):
                      logits[j,:] = logits2[j,:]

            # print("Finish 1 set testing")
            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask



            loss = criterion(logits, target)
            # print("After loss computng")
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            # print("After acc coputing")
            
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, true_labels, pred_labels






@torch.no_grad()
def evaluate_till_now3(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, test_sizes=None, 
                    all_global_prototype=None, all_global_prototype_var=None, args=None,):
    # stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    stat_matrix = np.zeros((3, task_id+1))

    test_sizes = np.array(test_sizes[0:task_id+1])
    # test_sizes = test_sizes[0:task_id+1]

    true_labels=[]
    pred_labels=[]
    for i in range(task_id+1):
        test_stats, true_labels_t, pred_labels_t = evaluate3(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, 
                            all_global_prototype=all_global_prototype, all_global_prototype_var=all_global_prototype_var,args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        
        true_labels.extend(true_labels_t)
        pred_labels.extend(pred_labels_t)
    # avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    avg_stat = np.divide(np.sum(stat_matrix*test_sizes, axis=1), np.sum(test_sizes))
    twavg_stat = np.divide(np.sum(stat_matrix, axis=1),  task_id+1)

    if task_id > 0:
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = np.divide(np.sum(stat_matrix[:,1:]*test_sizes[1:], axis=1), np.sum(test_sizes[1:]))
    else :
        base_classes_stat = stat_matrix[:,0]
        novel_classes_stat = stat_matrix[:,0]

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)


    return test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels


# def classify_with_proto(data, paras):
#     scale_factor =  10
#     logits = scale_factor * torch.nn.functional.cosine_similarity(
#         data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
#         paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
#         dim=-1)
#     return logits


# def classify(data, paras):
#     scale_factor =  10
#     logits = scale_factor * torch.nn.functional.cosine_similarity(
#         data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
#         paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
#         dim=-1)
#     return logits
# def count_accuracy(logits, label):
#     pred = torch.argmax(logits, dim=1).view(-1)
#     label = label.view(-1)
#     accuracy = 100 * pred.eq(label).float().mean()
#     return accuracy

def evaluate_server_global_model3(model, model_without_ddp, original_model, 
                    criterion, data_loader, optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, task_id=0, test_sizes=None, 
                    all_global_prototype=None, all_global_prototype_var=None, args = None,):
    
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    # acc_all_model=[]
    # for n in range(len(models)):  
    test_stats, avg_stat, base_classes_stat, novel_classes_stat, twavg_stat, true_labels, pred_labels = evaluate_till_now3(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, test_sizes=test_sizes, 
                                all_global_prototype=all_global_prototype, all_global_prototype_var=all_global_prototype_var, args=args)


    # acc_global_model=np.array(avg_stat[0])
    # print("Task: [" +str(task_id+1)+"] Server Global Model Acc@1: " + str(acc_global_model))
    print('-----------------------------------------------------------------------------------------------------')
    result_str = "[Global Model Average Top1-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[0], base_classes_stat[0], novel_classes_stat[0], twavg_stat[0], (base_classes_stat[0]+novel_classes_stat[0])/2)
    print(result_str)
    result_str = "[Global Model Average Top5-Acc till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[1], base_classes_stat[1], novel_classes_stat[1], twavg_stat[1], (base_classes_stat[1]+novel_classes_stat[1])/2)
    print(result_str)
    result_str = "[Global Model Average TestLoss till task{}]\t Avg: {:.4f}\tBaseClasses: {:.4f}\tNovelClasses: {:.4f}\tTWAvg: {:.4f}\tHarmonic: {:.4f}".format(task_id+1, 
        avg_stat[2], base_classes_stat[2], novel_classes_stat[2], twavg_stat[2], (base_classes_stat[2]+novel_classes_stat[2])/2)
    print(result_str)

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    print("Confussion Matrix at task-"+str(task_id+1))
    print(conf_mat.tolist())



def mixup_batch_representations(x, y, alpha=1.0, use_cuda=True, aug_prob=0.1):
    '''Returns mixed inputs, pairs of targets, and lambda
    source: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''
    batch_size = x.size()[0]

    aug_indexes = torch.rand(batch_size) < aug_prob

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if use_cuda:
        random_index = torch.randperm(batch_size).cuda()
    else:
        random_index = torch.randperm(batch_size)

    x[torch.where(aug_indexes)] = lam*x[torch.where(aug_indexes)] + (1 - lam)*x[random_index[torch.where(aug_indexes)]]

    y_a = y
    y_b = copy.deepcopy(y)
    y_b[torch.where(aug_indexes)] = y[random_index[torch.where(aug_indexes)]]
    return x, y_a, y_b, lam, aug_indexes