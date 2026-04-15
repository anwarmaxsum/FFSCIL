import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
import random

def FedAvg(server, models, distributed):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    

    with torch.no_grad():
        if distributed:
            server.g_prompt.data = models[0].module.g_prompt.data
            # server.g_prompt = copy.deepcopy(models[0].module.g_prompt)
        else:
            server.g_prompt.data = models[0].g_prompt.data
            # server.g_prompt = copy.deepcopy(models[0].g_prompt)

        for i in range(1, len(models)):
            if distributed:
                server.g_prompt.data += models[i].module.g_prompt.data
            else:
                server.g_prompt.data += models[i].g_prompt.data
        # model_agg.g_prompt = torch.nn.Parameter(torch.div(model_agg.g_prompt, len(models)))
        server.g_prompt.data = torch.div(server.g_prompt.data, len(models)).clone()

    print("Agregating g_prompt done")


    #Aggregate the E_prompt
    with torch.no_grad():
        if distributed:
            server.e_prompt.prompt_key.data = models[0].module.e_prompt.prompt_key.data
            server.e_prompt.prompt.data = models[0].module.e_prompt.prompt.data
            # server.e_prompt = copy.deepcopy(models[0].module.e_prompt)
        else:
            server.e_prompt.prompt_key.data = models[0].e_prompt.prompt_key.data
            server.e_prompt.prompt.data = models[0].e_prompt.prompt.data
            # server.e_prompt = copy.deepcopy(models[0].e_prompt)

        for i in range(1, len(models)):
            if distributed:
                server.e_prompt.prompt_key.data += models[i].module.e_prompt.prompt_key.data
                server.e_prompt.prompt.data += models[i].module.e_prompt.prompt.data
            else:
                server.e_prompt.prompt_key.data += models[i].e_prompt.prompt_key.data
                server.e_prompt.prompt.data += models[i].e_prompt.prompt.data
        # model_agg.e_prompt.prompt_key = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt_key , len(models)))
        # model_agg.e_prompt.prompt = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt , len(models)))
        server.e_prompt.prompt.data = torch.div(server.e_prompt.prompt.data , len(models)).clone()
        server.e_prompt.prompt_key.data = torch.div(server.e_prompt.prompt_key.data , len(models)).clone()
    
    print("Agregating e_prompt done")
    # return model_agg


def FedAvgWithHead(server, models, distributed):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        if distributed:
            # models[0].module.head.grad.zero_()
            # server.head.load_state_dict(models[0].module.head.state_dict())
            server.head.weight.data = models[0].module.head.weight.data
            server.head.bias.data = models[0].module.head.bias.data
        else:
            # models[0].head.grad.zero_()
            # server.head.load_state_dict(models[0].head.state_dict())
            server.head.weight.data = models[0].head.weight.data
            server.head.bias.data = models[0].head.bias.data


        # for name, param in server.head.named_parameters():
        for i in range(1, len(models)):
            if distributed:
                server.head.weight.data += models[i].module.head.weight.data
                server.head.bias.data += models[i].module.head.bias.data
            else:
                server.head.weight.data += models[i].head.weight.data
                server.head.bias.data += models[i].head.bias.data
        # server.head.weight = torch.div(server.head.weight, len(models)).clone()
        # server.head.bias = torch.div(server.head.bias, len(models)).clone()

        server.head.weight.data = torch.div(server.head.weight, len(models)).clone()
        server.head.bias.data = torch.div(server.head.bias, len(models)).clone()



        # for i in range(1, len(models)):
        #     if distributed:
        #         server.head.data += models[i].module.head.data
        #     else:
        #         server.head.data += models[i].head.data
        # # model_agg.g_prompt = torch.nn.Parameter(torch.div(model_agg.g_prompt, len(models)))
        # server.head.data = torch.div(server.head.data, len(models)).clone()

    print("Agregating head done")

    with torch.no_grad():
        if distributed:
            server.g_prompt.data = models[0].module.g_prompt.data
        else:
            server.g_prompt.data = models[0].g_prompt.data

        for i in range(1, len(models)):
            if distributed:
                server.g_prompt.data += models[i].module.g_prompt.data
            else:
                server.g_prompt.data += models[i].g_prompt.data
        # model_agg.g_prompt = torch.nn.Parameter(torch.div(model_agg.g_prompt, len(models)))
        server.g_prompt.data = torch.div(server.g_prompt.data, len(models))

    print("Agregating g_prompt done")


    #Aggregate the E_promp
    with torch.no_grad():
        if distributed:
            server.e_prompt.prompt_key.data = models[0].module.e_prompt.prompt_key.data
            server.e_prompt.prompt.data = models[0].module.e_prompt.prompt.data
        else:
            server.e_prompt.prompt_key.data = models[0].e_prompt.prompt_key.data
            server.e_prompt.prompt.data = models[0].e_prompt.prompt.data
        
        for i in range(1, len(models)):
            if distributed:
                server.e_prompt.prompt_key.data += models[i].module.e_prompt.prompt_key.data
                server.e_prompt.prompt.data += models[i].module.e_prompt.prompt.data
            else:
                server.e_prompt.prompt_key.data += models[i].e_prompt.prompt_key.data
                server.e_prompt.prompt.data += models[i].e_prompt.prompt.data
        # model_agg.e_prompt.prompt_key = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt_key , len(models)))
        # model_agg.e_prompt.prompt = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt , len(models)))
        server.e_prompt.prompt.data = torch.div(server.e_prompt.prompt.data , len(models)).clone()
        server.e_prompt.prompt_key.data = torch.div(server.e_prompt.prompt_key.data , len(models)).clone()
    
    print("Agregating e_prompt done")



def FedDistribute(server,clients,distributed):
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        for i in range(0, len(clients)):
            if distributed:
                clients[i].module.g_prompt.data = server.g_prompt.data.clone()
                clients[i].module.e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
                clients[i].module.e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()
            else:
                clients[i].g_prompt.data = server.g_prompt.data.clone()
                clients[i].e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
                clients[i].e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()
    print("Distributing g_prompt and e_prompt done")




def FedDistributeWithHead(server,clients,distributed):
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        for i in range(0, len(clients)):
            if distributed:
                clients[i].module.g_prompt.data = server.g_prompt.data.clone()
                clients[i].module.e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
                clients[i].module.e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()
                
                clients[i].module.head.weight.data = server.head.weight.data.clone()
                clients[i].module.head.bias.data = server.head.bias.data.clone() 
                # clients[i].module.head.load_state_dict(server.head.state_dict())

                # clients[i].module.g_prompt.requires_grad=True
                # clients[i].module.e_prompt.prompt.requires_grad=True
                # clients[i].module.e_prompt.prompt_key.requires_grad=True
                # clients[i].module.head.requires_grad=True
            else:
                clients[i].g_prompt.data = server.g_prompt.data.clone()
                clients[i].e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
                clients[i].e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()

                clients[i].head.weight.data = server.head.weight.data.clone()
                clients[i].head.bias.data = server.head.bias.data.clone() 
                # clients[i].head.load_state_dict(server.head.state_dict())

                # clients[i].g_prompt.requires_grad=True
                # clients[i].e_prompt.prompt.requires_grad=True
                # clients[i].e_prompt.prompt_key.requires_grad=True
                # clients[i].head.requires_grad=True


    print("Distributing g_prompt and e_prompt done")
    # model_agg.g_prompt[k] = torch.div(model_agg.g_prompt[k], len(models))

    # for k in model_agg.g_prompt.prompt_key:
    #     for i in range(1, len(models)):
    #         model_agg.g_prompt.prompt_key[k] += models[i].g_prompt[k]
    #     model_agg.g_prompt[k] = torch.div(model_agg.g_prompt[k], len(models))
    # return model_agg



def FedAvgPrototype(clients_prototype,task_id,classes_per_task):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    min_c = task_id*classes_per_task
    max_c = (task_id+1)*classes_per_task
    
    global_prototype = {}
    proto_size = list(clients_prototype[0].values())[0].shape[0]
    for c in range(min_c,max_c+1):
        counter = 0
        proto_c = np.zeros(proto_size)
        for n in range(len(clients_prototype)):
            if c in clients_prototype[n]:
                proto_c = proto_c + clients_prototype[n][c]
                counter = counter + 1

        if counter > 0:
            proto_c = proto_c / counter
            global_prototype[c] = proto_c
    
    print("Agregating prototype done")
    return global_prototype




def FedAvgPrototype2(clients_prototype, clients_prototype_var, clients_weight, task_id, classes_per_task):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    min_c = task_id*classes_per_task
    max_c = (task_id+1)*classes_per_task
    
    global_prototype = {}
    global_prototype_var = {}
    proto_size = list(clients_prototype[0].values())[0].shape[0]

    for c in range(min_c,max_c+1):
        
        counter = 0
        total_weight = 0
        proto_c = np.zeros(proto_size)
        proto_var_c = np.zeros(proto_size)
        
        for n in range(len(clients_prototype)):
            if c in clients_prototype[n]:
                proto_c = proto_c + (clients_weight[n]*clients_prototype[n][c])
                peoro_var_c = proto_var_c + ((clients_prototype_var[n][c]+(clients_prototype[n][c]*clients_prototype[n][c]))*clients_weight[n])

                total_weight =  total_weight + clients_weight[n]
                counter = counter + 1

        if counter > 0:
            # proto_c = proto_c / counter
            proto_c = proto_c / total_weight
            proto_var_c = (proto_var_c/total_weight) - (proto_c*proto_c)
            global_prototype[c] = proto_c
            global_prototype_var[c] = proto_var_c
    
    print("Agregating prototype done")
    return global_prototype, global_prototype_var