import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from data.ffscil_datasets import build_continual_dataloader
# from dualpromptlib.engine import *
from piplib.ffscil_xpip_engine1v_nodproto import *

import dualpromptlib.modelsX as models
# import dualpromptlib.models2 as models
import dualpromptlib.utils as utils


import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

from fed_pip_utils_nogp import * 
import copy

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    args.use_g_prompt = False
    args.use_prefix_tune_for_g_prompt = False
    args.e_prompt_layer_idx = [0,1,2,3,4]
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


    # print(f"Creating server original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    original_model.to(device)

    
    print(f"Creating server model: {args.model}")
    server_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
    )
    server_model.to(device)
    

    # print(vars(server_model.head))

    for i in range(args.num_clients):
        print(f"Creating model : {args.model} for client {i}")
        # original_model = original_model
        model =   copy.deepcopy(server_model).to(device)

        models.append(model)
        original_models.append(original_model)
    

    if args.freeze:
        # all parameters are frozen for original vit model
        for i in range(0,args.num_clients):
            for p in original_models[i].parameters():
                p.requires_grad = False
            
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in models[i].named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False
            
            # for n, p in models[i].named_parameters():
            #     if n.startswith("fs_head"):
            #         p.requires_grad = False

    print(args)


    server_model_without_ddp = server_model
    for i in range(0,args.num_clients):
        models_without_ddp.append(models[i])
        

    if args.distributed:
        server_model_without_ddp = server_model.module
        for i in range(0,args.num_clients):
            models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            models_without_ddp[i] = models[i].module
            

        # model2 = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model_without_ddp2 = model2.module
    for n, p in models[0].named_parameters():
        if (p.requires_grad):
            print(n)
            print(str(p.numel()))

    n_parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    server_optimizer = create_optimizer(args, server_model_without_ddp)
    for i in range(0,args.num_clients):
        optimizer = create_optimizer(args, models_without_ddp[i])
        optimizers.append(optimizer)

    # optimizer2 = create_optimizer(args, model_without_ddp2)

    if args.sched != 'constant':
        for i in range(0,args.num_clients):
            lr_scheduler, _ = create_scheduler(args, optimizers[i])
            lr_schedulers.append(lr_scheduler)
    elif args.sched == 'constant':
        for i in range(0,args.num_clients):
            lr_scheduler = None
            lr_schedulers.append(lr_scheduler)
        

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()


    all_time_round=0
    global_prototype =  None
    global_prototype_var =  None
    all_global_prototype =  {}
    all_global_prototype_var =  {}


    FedAvgWithHead(server_model,models,args.distributed)
    

    for task_id in range(0,args.num_tasks):
    # for task_id in range(1,args.num_tasks):
        if task_id > 0:
            args.classes_per_task = args.fs_classes
            args.available_classes = args.available_fs_classes
        else:
            args.classes_per_task = args.base_classes
            args.available_classes = args.available_base_classes

        clients_participations = [0] * args.num_clients

        if task_id > 0:
            args.epochs = args.fs_epochs

        RT = args.rounds_per_task
        # if task_id == 0:
        #     RT = 1
        if task_id == 1:
            # args.lr = 0.07
            # for n, p in server_model.head.named_parameters():
            #         p.requires_grad = False

            # for c in range(0,len(models)):
            #     for n, p in models[c].head.named_parameters():
            #         p.requires_grad = False

            args.lr = args.lr_fs
            optimizers = []
            for i in range(0,args.num_clients):
                optimizer = create_optimizer(args, models_without_ddp[i])
                optimizers.append(optimizer)

        for n_round in range(0,RT):
            print("Task ["+str(task_id+1)+"] Global Round : "+str(all_time_round+1))

            clients_index = random.sample(range(args.num_clients), args.local_clients)
            print(clients_index)

            # if (n_round==0):
                # FedDistribute(server_model,[models[i] for i in clients_index],args.distributed)
            # else:
            # FedDistributeWithHead(server_model,[models[i] for i in clients_index],args.distributed)
            # train_and_evaluate_pertask(models, models_without_ddp, original_models,
            #                 criterion, data_loaders, optimizers, lr_schedulers,
            #                 device, class_masks, task_id, args)
                        
            idx_notrain = [x for x in clients_index if clients_participations[x]==0]
            idx_trained = [x for x in clients_index if clients_participations[x]>0]
            FedDistribute(server_model,[models[i] for i in idx_trained],args.distributed)
            if(task_id==0):
                FedDistributeWithHead(server_model,[models[i] for i in idx_notrain],args.distributed)
            else:
                FedDistribute(server_model,[models[i] for i in idx_notrain],args.distributed)



            for i in clients_index:
                clients_participations[i] =  clients_participations[i] + 1

            clients_weight =  [clients_participations[i] for i in clients_index]


            if (task_id == 0):
                if (n_round) == 0:
                    clients_prototype, clients_prototype_var = train_pertask([models[i] for i in clients_index], 
                                    [models_without_ddp[i] for i in clients_index], 
                                    [original_models[i] for i in clients_index],
                                    criterion, 
                                    [data_loaders[i] for i in clients_index],
                                    # [data_loaders[0]],
                                    [optimizers[i] for i in clients_index], 
                                    [lr_schedulers[i] for i in clients_index],
                                    device, 
                                    [class_masks[i] for i in clients_index], 
                                    task_id, args)

                else:
                    clients_prototype, clients_prototype_var  = train_pertask_v2([models[i] for i in clients_index], 
                                    [models_without_ddp[i] for i in clients_index], 
                                    [original_models[i] for i in clients_index],
                                    criterion, 
                                    [data_loaders[i] for i in clients_index],
                                    # [data_loaders[0]],
                                    [optimizers[i] for i in clients_index], 
                                    [lr_schedulers[i] for i in clients_index],
                                    device, 
                                    [class_masks[i] for i in clients_index], 
                                    task_id, global_prototype, global_prototype_var, args)
                    print("Train 1 round with prototype done")

            else:
                if (n_round) == 0:
                    clients_weight =  [clients_participations[i] for i in clients_index]
                    clients_prototype, clients_prototype_var = train_pertask([models[i] for i in clients_index], 
                                    [models_without_ddp[i] for i in clients_index], 
                                    [original_models[i] for i in clients_index],
                                    criterion, 
                                    [data_loaders[i] for i in clients_index],
                                    # [data_loaders[0]],
                                    [optimizers[i] for i in clients_index], 
                                    [lr_schedulers[i] for i in clients_index],
                                    device, 
                                    [class_masks[i] for i in clients_index], 
                                    task_id, args)

                    global_prototype, global_prototype_var =  FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args)
                    for k in global_prototype.keys():
                        all_global_prototype[k] =  global_prototype[k]
                        all_global_prototype_var[k] = global_prototype_var[k]
                    print("All Global prototypes:")
                    print(all_global_prototype.keys())

                else:
                    clients_prototype, clients_prototype_var  = train_pertask_v2([models[i] for i in clients_index], 
                                    [models_without_ddp[i] for i in clients_index], 
                                    [original_models[i] for i in clients_index],
                                    criterion, 
                                    [data_loaders[i] for i in clients_index],
                                    # [data_loaders[0]],
                                    [optimizers[i] for i in clients_index], 
                                    [lr_schedulers[i] for i in clients_index],
                                    device, 
                                    [class_masks[i] for i in clients_index], 
                                    task_id, global_prototype, global_prototype_var, args)
                    print("Train 1 round with prototype done")    

                print("Train 1 round with prototype done")

         
            # global_prototype, global_prototype_var =  FedAvgPrototype2(clients_prototype,clients_prototype_var,clients_weight,task_id)
            # global_prototype, global_prototype_var =  FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args.classes_per_task)
            global_prototype, global_prototype_var =  FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args)
            for k in global_prototype.keys():
                    all_global_prototype[k] =  global_prototype[k]
                    all_global_prototype_var[k] = global_prototype_var[k]  
            print("All Global prototypes:")
            print(all_global_prototype.keys())   

        # train_and_evaluate_pertask(models, models_without_ddp, original_models,
        #                 criterion, data_loaders, optimizers, lr_schedulers,
        #                 device, class_masks, 0, args)

            print("Global prototypes catalouge:")
            print(list(global_prototype.keys()))

            if (n_round < (args.rounds_per_task-1)):
                FedWeightedAvg(server_model, [models[i] for i in clients_index], clients_weight, args.distributed)
            else:
                if (task_id == 0):
                    FedWeightedAvgWithHead(server_model, [models[i] for i in clients_index], clients_weight, args.distributed)
                else:
                    FedWeightedAvg(server_model, [models[i] for i in clients_index], clients_weight, args.distributed)
                # idx_notrain = [ x for x in range(0,args.num_clients) if clients_participations[x]==0 ]
                # FedDistributeWithHead(server_model,[models[i] for i in idx_notrain],args.distributed)

            all_time_round =  all_time_round + 1

        FedDistribute(server_model,models,args.distributed)

            # server_model.g_prompt = copy.deepcopy(fixed_g_prompt)
            # print(server_model.head.state_dict())
            # print(server_model_without_ddp.head.state_dict())
            # print(server_model.g_prompt)
        # evaluate_server_global_model(server_model, server_model_without_ddp, original_model, global_prototype, global_prototype_var,
        #                     criterion, data_loaders[0], server_optimizer, None,
        #                     device, None, task_id,  args)

        # evaluate_server_global_model(server_model, server_model_without_ddp, original_model,
        #                     criterion, data_loaders[0], server_optimizer, None,
        #                     device, None, task_id, test_sizes, args)
        evaluate_server_global_model3(server_model, server_model_without_ddp, original_model,
                            criterion, data_loaders[0], server_optimizer, None,
                            device, None, task_id, test_sizes, 
                            all_global_prototype, all_global_prototype_var, args)

        print("Clients Participation on Task: [" +str(task_id+1)+"]")
        print(clients_participations)
        


        # evaluate_all_clients(models, models_without_ddp, original_models,
        #                     criterion, data_loaders, optimizers, lr_schedulers,
        #                     device, class_masks, task_id, args)


        # FedAvg(server_model, models,args.distributed)
        # FedDistribute(server_model,models,args.distributed)


    # train_and_evaluate_pertask(models, models_without_ddp, original_models,
    #                 criterion, data_loaders, optimizers, lr_schedulers,
    #                 device, class_masks, 1, args)

    # FedAvg(server_model, models,args.distributed)
    # FedDistribute(server_model,models,args.distributed)

    # train_and_evaluate(models[0], models_without_ddp[0], original_models[0],
    #                 criterion, data_loaders[0], optimizers[0], lr_schedulers[0],
    #                 device, class_masks[0], args)

    # train_and_evaluate_pertask(models, models_without_ddp, original_models,
    #                 criterion, data_loaders, optimizers, lr_schedulers,
    #                 device, class_masks, 2, args)

    # train_and_evaluate2(model, model_without_ddp, original_model,
    #                 model2, model_without_ddp2, original_model2,
    #                 criterion, train_loader, optimizer, lr_scheduler,
    #                 device, class_mask, args)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # parser.add_argument('--output_dir', default='./output/', type=str, help='output dir')

    config = parser.parse_known_args()[-1][0]

    # config = 'cifar100_dualprompt'
    # config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        print("chek config cifar100")
        from dualpromptlib.configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
        # print()

    elif config == 'ffscil_cifar100_9tasks_60bases_5ways':
        print("Check ffscil_cifar100_9tasks_60bases_5ways")
        from configs.ffscil_cifar100_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cifar100_9tasks_60bases_5ways', help='Split-CIFAR100 FFSCIL configs')

    elif config == 'ffscil_imagenetsubset_9tasks_60bases_5ways':
        print("Check ffscil_imagenetsubset_9tasks_60bases_5ways")
        from configs.ffscil_imagenetsubset_9tasks_60bases_5ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_imagenetsubset_9tasks_60bases_5ways', help='Split-ImagenetSubset FFSCIL configs')


    elif config == 'ffscil_cub200_11tasks_100bases_10ways':
        print("Check ffscil_cub200_11tasks_100bases_10ways")
        from configs.ffscil_cub200_11tasks_100bases_10ways import get_args_parser
        config_parser = subparser.add_parser('ffscil_cub200_11tasks_100bases_10ways', help='Split-CUB200 FFSCIL configs')


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
