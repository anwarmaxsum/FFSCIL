#from GLFC import GLFC_model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import yaml
# import json
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

# from dualpromptlib.datasets import build_continual_dataloader
from data.ffscil_datasets import build_continual_dataloader
# from dualpromptlib.engine import *
# from codapromptlib.fed_codaprompt_engine import *
# import codapromptlib.models as models
import dualpromptlib.utils as utils

import random

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

# from fed_codaprompt_utils import * 
import copy


import codapromptlib.learners as learners

import codapromptlib.dataloaders as dataloaders
from codapromptlib.dataloaders.utils import *
from torch.utils.data import DataLoader

from fed_codap_utils import * 

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    
    data_loaders=[]
    class_masks=[]

    models=[]
    original_models=[]
    models_without_ddp=[]
    optimizers=[]
    lr_schedulers=[]

    data_loader, class_mask = build_continual_dataloader(args)
    print("Class Mask")
    for i in range(len(class_mask)):
        print(class_mask[i])


    data_loaders.append(data_loader)
    class_masks.append(class_mask)
    #data_loader2, class_mask2 = build_continual_dataloader(args)
    for i in range(1, args.num_clients):
        # dl =  copy.deepcopy(data_loader)
        # cm =  copy.deepcopy(class_mask)
        # random.shuffle(dl)
        dl, cm = build_continual_dataloader(args)
        data_loaders.append(dl)
        class_masks.append(cm)

    print("Check Train Labels: \n==========================")
    for t in range(0,args.num_tasks):
        data_loader0=data_loader[t]['train']
        # print(vars(data_loader0.dataset))
        set_labels=set()
        print("Task : " + str(t))
        label_list = []
        for input, target in data_loader0:
            set_labels.update(set(target.numpy()))
            label_list.extend(target)
            # break
        print(list(set_labels))
        print(len(label_list))


    print("Check Validation Labels: \n==========================")
    test_sizes = []
    for t in range(0,args.num_tasks):
        data_loader0=data_loader[t]['val']
        set_labels=set()
        print("Task : " + str(t))
        label_list = []
        for input, target in data_loader0:
            # print(input.shape)
            # print(target.shape)
            set_labels.update(set(target.numpy()))
            label_list.extend(target)
            # break
        print(list(set_labels))
        print(len(label_list))
        test_sizes.append(len(label_list))
    

    print(args)

    # num_classes = args.base_classes + (args.num_tasks-1)* args.fs_classes

    learner_config = {'num_classes': args.num_classes,
                'lr': args.lr,
                'debug_mode': args.debug_mode == 1,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'schedule': args.schedule,
                'schedule_type': args.schedule_type,
                'model_type': args.model_type,
                'model_name': args.model_name,
                'optimizer': args.optimizer,
                'gpuid': args.gpuid,
                'memory': args.memory,
                'temp': args.temp,
                'out_dim': args.num_classes,
                'overwrite': args.overwrite == 1,
                'DW': args.DW,
                'batch_size': args.batch_size,
                'upper_bound_flag': args.upper_bound_flag,
                # 'tasks': tasks_logits,
                'tasks': class_mask,
                'top_k': 1,
                'prompt_param':[args.num_tasks,args.prompt_param]
                }

    learner_type, learner_name = args.learner_type, args.learner_name
    learner = learners.__dict__[learner_type].__dict__[learner_name](learner_config)
    server_model = learner

    # learner.add_valid_output_dim(args.classes_per_task)
    # learner.add_valid_output_dim(args.num_classes)
    print("create server model")


    if len(args.gpuid) > 1: 
        server_prompt = server_model.model.module.prompt
        server_head = server_model.model.module.last
    else:       
        server_prompt = server_model.model.prompt
        server_head = server_model.model.last

    # for l in range(server_prompt.e_p_length):
    for l in range(5):    
        # print("Layer: " + str(l))
        K = getattr(server_prompt,f'e_k_{l}')
        A = getattr(server_prompt,f'e_a_{l}')
        p = getattr(server_prompt,f'e_p_{l}')
        # print(K.shape)
        # print(A.shape)
        # print(p.shape)


    for n, p in learner.model.named_parameters():
        if n.startswith(tuple(args.freeze)):
            p.requires_grad = False
        # elif i==0:
        #     print(n)

    for n, p in learner.named_parameters():
         if p.requires_grad:
            print(n)

    for i in range(args.num_clients):
        print(f"Creating model : for client {i}")
        # original_model = original_model
        # client_prompt =  copy.deepcopy(server_prompt).to(device)
        # client_head =  copy.deepcopy(server_head).to(device)
        # model = (client_prompt,client_head)
        model = copy.deepcopy(server_model)
        models.append(model)
        # original_models.append(original_model)

    print("create model success")

    start_time = time.time()
    # avg_train_time = learner.learn_batch(0,data_loader[0]['train'], data_loader[0]['train'].dataset, './model_ckpt/', data_loader[0]['val'])

    # for task in range(args.num_tasks):
    #     for c in range(args.num_clients):
    #         clearner = models[c]
    #         clearner.add_valid_output_dim(args.classes_per_task)   
    FedAvgWithHead(server_model, models ,args.distributed)
    FedDistributeWithHead(server_model,models,args.distributed)

    all_time_round=0
    

    for task in range(args.num_tasks):
        if (task > 0):
            args.lr = args.lr_fs
            args.epochs = args.fs_epochs
            for c in range(args.num_clients):
                # models[c].init_optimizer()
                models[c].config['lr'] = args.lr_fs
                # models[c].optimizer.param_groups[0]['lr'] = args.lr_fs

        if (task == 0):
            server_model.add_valid_output_dim(args.base_classes)
            for c in range(args.num_clients):
                models[c].add_valid_output_dim(args.base_classes)
        else:
            server_model.add_valid_output_dim(args.fs_classes)
            for c in range(args.num_clients):
                models[c].add_valid_output_dim(args.fs_classes)
        

        # print("==== Training on Task [" + str(task) + "]")
        RT = args.rounds_per_task
        for n_round in range(0,RT):
            print("Task ["+str(task+1)+"] Global Round : "+str(all_time_round+1))
 

            clients_index = random.sample(range(args.num_clients), args.local_clients)
            print(clients_index)

            # FedDistributeWithHead(server_model,[models[i] for i in clients_index],args.distributed)
            FedDistributeWithHead(server_model,models,args.distributed)

            # for c in clients_index:
            #     print("Client: " +str(c) + " Validation before training")
            #     models[c].validation_till_task(task, data_loader, model=None, task_metric='acc',  verbal = True, task_global=False)

            for c in clients_index:
                learner = models[c]

                print("Client ID: " + str(c))

                # avg_train_time = learner.learn_per_task(task, args.epochs, data_loader[task]['train'], data_loader[task]['train'].dataset, './model_ckpt/', data_loader[task]['val'])
                if task==0:
                    avg_train_time = learner.learn_per_task2(task, args.epochs, data_loader[task]['train'], data_loader[task]['train'].dataset, './model_ckpt/', args, data_loader[task]['val'])
                else:
                    avg_train_time = learner.learn_per_task2(task, args.epochs, data_loaders[c][task]['train'], data_loaders[c][task]['train'].dataset, './model_ckpt/', args, data_loaders[c][task]['val'])
                
                # print("Client " +str(c)+ " Validation")
                # learner.validation_till_task(task, data_loader, model=None, task_metric='acc',  verbal = True, task_global=False)

            FedAvgWithHead(server_model, [models[i] for i in clients_index] ,args.distributed)
            
            # print("Server Validation")
            # server_model.validation_till_task(task, data_loader, model=None, task_metric='acc',  verbal = True, task_global=False)

            all_time_round =  all_time_round + 1

        FedDistributeWithHead(server_model,models,args.distributed)

        print("Server Validation on Task: " + str(task+1))
        server_model.validation_till_task(task, data_loader, model=None, task_metric='acc',  verbal = True, task_global=False)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # parser.add_argument('--output_dir', default='./output/', type=str, help='output dir')
    # parser.add_argument('--gpuid', nargs="+", type=int, default=[0,1],
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='CODAPrompt', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    # parser.add_argument('--prompt_param', nargs="+", type=float, default=[100, 8,0.0],
    #                      help="e prompt pool size, e prompt length, g prompt length")
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[10, 8,0.0],
                         help="e prompt pool size, e prompt length, g prompt length")
    
    # parser.add_argument('--num_classes', type=int, default=100, help="num_classes")
    # parser.add_argument('--first_split_size', type=int, default=10, help="first task split")
    # parser.add_argument('--other_split_size', type=int, default=10, help="other task split")

    parser.add_argument('--schedule', nargs="+", type=int, default=[2], help="train schedule")
    parser.add_argument('--schedule_type', type=str, default='cosine', help="Schedule type")
    # parser.add_argument('--batch_size', type=int, default=128, help="train batch_size")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer")
    # parser.add_argument('--lr', type=float, default=0.002, help="train batch_size")
    parser.add_argument('--momentum', type=float, default=0.9, help="train batch_size")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="train batch_size")
    parser.add_argument('--model_type', type=str, default='zoo', help="Model type")
    parser.add_argument('--model_name', type=str, default='vit_pt_imnet', help="Model name")
    parser.add_argument('--max_task', type=int, default=-1, help="train batch_size")
    parser.add_argument('--dataroot', type=str, default='data', help="data root")
    # parser.add_argument('--workers', type=int, default=4, help="workers")
    parser.add_argument('--validation', default=False, action='store_true', help='validation')
    parser.add_argument('--train_aug', default=True, action='store_true', help='train aug')
    parser.add_argument('--rand_split', default=True, action='store_true', help='rand split')
    # parser.add_argument('--seed', type=int, default=2021, help="seed")

    config = parser.parse_known_args()[-1][0]

    # config = 'cifar100_dualprompt'
    # config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_codaprompt':
        print("chek config cifar100")
        from dualpromptlib.configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 codaprompt configs')
        # print()

    if config == 'cifar100_dualprompt':
        print("chek config cifar100")
        from dualpromptlib.configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
        # print()

    elif config == 'ffscil_cifar100_codap_9tasks_60bases_5ways':
        print("Check ffscil_cifar100_codap_9tasks_60bases_5ways")
        from configs.ffscil_cifar100_codap_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cifar100_codap_9tasks_60bases_5ways', help='Split-CIFAR100 FFSCIL configs')

    elif config == 'ffscil_imagenetsubset_codap_9tasks_60bases_5ways':
        print("Check ffscil_imagenetsubset_codap_9tasks_60bases_5ways")
        from configs.ffscil_imagenetsubset_codap_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_imagenetsubset_codap_9tasks_60bases_5ways', help='Split-ImagenetSubset FFSCIL configs')


    elif config == 'ffscil_cub200_codap_11tasks_100bases_10ways':
        print("Check ffscil_cub200_codap_11tasks_100bases_10ways")
        from configs.ffscil_cub200_codap_11tasks_100bases_10ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cub200_codap_11tasks_100bases_10ways', help='Split-CUB200 FFSCIL configs')



    elif config == 'imr_codaprompt':
        from codapromptlib.configs.imr_codaprompt import get_args_parser
        config_parser = subparser.add_parser('imr_codaprompt', help='Split-ImageNet-R codaprompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    parser.add_argument('--output_dir', default='./output/', type=str, help='output dir')
    args = parser.parse_args()
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)
