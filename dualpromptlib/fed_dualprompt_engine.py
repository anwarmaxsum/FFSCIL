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
    

    # min_c = task_id*10
    # max_c = (task_id+1)*10
    # all_classes = [item for item in range(min_c, max_c)]
    # print(all_classes)
    # available_classes = random.sample(all_classes, 6)
    # print(available_classes)

    iter=0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        
        # if task_id > 0:
        # print(set(target))
        # tmp = copy.deepcopy(target)
        # filter = tmp.apply_(lambda x: x in available_classes).bool()
        # # print(target)
        # # print(filter)
        # input=input[filter.tolist()]
        # target=target[filter.tolist()]
        # print(input.shape)

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
        min_c = task_id*args.classes_per_task
        max_c = (task_id+1)*args.classes_per_task
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
                # print("Original model not None")
                # print("forward original model")
                output = original_model(input)
                cls_features = output['pre_logits']
                # cls_features = torch.cat([cls_features,cls_features],dim=0)
                # print("cls_features shape")
                # print(cls_features.shape)
            else:
                # print("Original model None")
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

@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats, avg_stat

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
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
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):
            dloader=data_loader[task_id]['train']
                      
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')



def train_and_evaluate_pertask(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, args = None,):
    

    # create matrix to save end-of-task accuracies 
    print("Train and Evaluate per Task [TaskID]: " + str(task_id))
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
                            models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            models[n].e_prompt.prompt.grad.zero_()
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
                        models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 
    # Create new optimizer for each task to clear optimizer status
    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])
    
    for epoch in range(args.epochs):
        for n in range(len(models)):            
            train_stats = train_one_epoch(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[0][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], args=args,)

            # train_stats = train_one_epoch(model=model2, original_model=original_model2, criterion=criterion, 
            #                             data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
            #                             device=device, epoch=epoch, max_norm=args.clip_grad, 
            #                             set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            # if lr_schedulers[n]:
            #     lr_schedulers[n].step(epoch)

    acc_all_models=[]
    for n in range(len(models)):  
        test_stats, avg_stat = evaluate_till_now(model=models[n], original_model=original_models[n], data_loader=data_loaders[0], device=device, 
                                    task_id=task_id, class_mask=class_masks[n], acc_matrix=acc_matrix, args=args)
        acc_all_models.append(avg_stat[0])
    # if args.output_dir and utils.is_main_process():
    #     Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
        
    #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
    #     state_dict = {
    #             'model': model_without_ddp.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'args': args,
    #         }
    #     if args.sched is not None and args.sched != 'constant':
    #         state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        
    #     utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,'n-client_idx:':n,}

    acc_all_models=np.array(acc_all_models)
    print("Task: [" +str(task_id)+"] All clients summary:")
    print(range(len(models)))
    print(acc_all_models)
    print("Task: [" +str(task_id)+"] Avg Acc of All clients:"+str(np.mean(acc_all_models)))


        

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')



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
    min_c = task_id*args.classes_per_task
    max_c = (task_id+1)*args.classes_per_task
    all_classes = [item for item in range(min_c, max_c)]
    for n in range(len(models)):
        available_classes = random.sample(all_classes, args.available_classes)
        available_classes_all.append(available_classes)
    
    for epoch in range(args.epochs):
        for n in range(len(models)):            
            # train_stats = train_one_epoch(model=models[n], original_model=original_models[n], criterion=criterion, 
            #                             data_loader=data_loaders[0][task_id]['train'], optimizer=optimizers[n], 
            #                             device=device, epoch=epoch, max_norm=args.clip_grad, 
            #                             set_training_mode=True, task_id=task_id, class_mask=class_masks[n], args=args,)
            # print(f"Train client {}")
            train_stats = train_one_epoch_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[0][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
                                        available_classes=available_classes_all[n], args=args,)
        

    # if args.output_dir and utils.is_main_process():
    #     with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
    #         f.write(json.dumps(log_stats) + '\n')




def train_pertask_full(models, models_without_ddp, original_models, 
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
                            models[n].module.e_prompt.prompt.grad.zero_()
                            models[n].module.e_prompt.prompt[cur_idx] = models[n].module.e_prompt.prompt[prev_idx]
                            optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                        else:
                            models[n].e_prompt.prompt.grad.zero_()
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
                        models[n].module.e_prompt.prompt_key.grad.zero_()
                        models[n].module.e_prompt.prompt_key[cur_idx] = models[n].module.e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].module.parameters()
                    else:
                        models[n].e_prompt.prompt_key.grad.zero_()
                        models[n].e_prompt.prompt_key[cur_idx] = models[n].e_prompt.prompt_key[prev_idx]
                        optimizers[n].param_groups[0]['params'] = models[n].parameters()
 
    # Create new optimizer for each task to clear optimizer status
    if task_id > 0 and args.reinit_optimizer:
        for n in range(len(models)):
            optimizers[n] = create_optimizer(args, models[n])

    available_classes_all=[]

    
    for epoch in range(args.epochs):
        for n in range(len(models)):            
            train_stats = train_one_epoch(model=models[n], original_model=original_models[n], criterion=criterion, 
                                        data_loader=data_loaders[n][task_id]['train'], optimizer=optimizers[n], 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_masks[n], args=args,)

            # train_stats = train_one_epoch_with_available_classes(model=models[n], original_model=original_models[n], criterion=criterion, 
            #                             data_loader=data_loaders[0][task_id]['train'], optimizer=optimizers[n], 
            #                             device=device, epoch=epoch, max_norm=args.clip_grad, 
            #                             set_training_mode=True, task_id=task_id, class_mask=class_masks[n], 
            #                             available_classes=available_classes_all[n], args=args,)
        


def evaluate_all_clients(models, models_without_ddp, original_models, 
                    criterion, data_loaders, optimizers, lr_schedulers, device: torch.device, 
                    class_masks=None, task_id=0, args = None,):
    
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    acc_all_models=[]
    for n in range(len(models)):  
        test_stats, avg_stat = evaluate_till_now(model=models[n], original_model=original_models[n], data_loader=data_loaders[0], device=device, 
                                    task_id=task_id, class_mask=class_masks[n], acc_matrix=acc_matrix, args=args)
        acc_all_models.append(avg_stat[0])

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        #     'epoch': epoch,'n-client_idx:':n,}

    acc_all_models=np.array(acc_all_models)
    print("Task: [" +str(task_id+1)+"] All clients summary:")
    print(range(len(models)))
    print(acc_all_models)
    print("Task: [" +str(task_id+1)+"] Avg Acc of All clients:"+str(np.mean(acc_all_models)))


def evaluate_server_global_model(model, model_without_ddp, original_model, 
                    criterion, data_loader, optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, task_id=0, args = None,):
    
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    # acc_all_model=[]
    # for n in range(len(models)):  
    test_stats, avg_stat = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
    # acc_all_models.append(avg_stat[0])

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        #     'epoch': epoch,'n-client_idx:':n,}

    acc_global_model=np.array(avg_stat[0])
    print("Task: [" +str(task_id+1)+"] Server Global Model Accuracy: " + str(acc_global_model))

