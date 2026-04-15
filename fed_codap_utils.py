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


# def FedAvg(server, models, distributed):
    
#     # model_agg = copy.deepcopy(models[0])
#     #Aggregate the G_prompt
    

#     with torch.no_grad():
#         if distributed:
#             server.g_prompt.data = models[0].module.g_prompt.data
#             # server.g_prompt = copy.deepcopy(models[0].module.g_prompt)
#         else:
#             server.g_prompt.data = models[0].g_prompt.data
#             # server.g_prompt = copy.deepcopy(models[0].g_prompt)

#         for i in range(1, len(models)):
#             if distributed:
#                 server.g_prompt.data += models[i].module.g_prompt.data
#             else:
#                 server.g_prompt.data += models[i].g_prompt.data
#         # model_agg.g_prompt = torch.nn.Parameter(torch.div(model_agg.g_prompt, len(models)))
#         server.g_prompt.data = torch.div(server.g_prompt.data, len(models)).clone()

#     print("Agregating g_prompt done")


#     #Aggregate the E_prompt
#     with torch.no_grad():
#         if distributed:
#             server.e_prompt.prompt_key.data = models[0].module.e_prompt.prompt_key.data
#             server.e_prompt.prompt.data = models[0].module.e_prompt.prompt.data
#             # server.e_prompt = copy.deepcopy(models[0].module.e_prompt)
#         else:
#             server.e_prompt.prompt_key.data = models[0].e_prompt.prompt_key.data
#             server.e_prompt.prompt.data = models[0].e_prompt.prompt.data
#             # server.e_prompt = copy.deepcopy(models[0].e_prompt)

#         for i in range(1, len(models)):
#             if distributed:
#                 server.e_prompt.prompt_key.data += models[i].module.e_prompt.prompt_key.data
#                 server.e_prompt.prompt.data += models[i].module.e_prompt.prompt.data
#             else:
#                 server.e_prompt.prompt_key.data += models[i].e_prompt.prompt_key.data
#                 server.e_prompt.prompt.data += models[i].e_prompt.prompt.data
#         # model_agg.e_prompt.prompt_key = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt_key , len(models)))
#         # model_agg.e_prompt.prompt = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt , len(models)))
#         server.e_prompt.prompt.data = torch.div(server.e_prompt.prompt.data , len(models)).clone()
#         server.e_prompt.prompt_key.data = torch.div(server.e_prompt.prompt_key.data , len(models)).clone()
    
#     print("Agregating e_prompt done")
#     # return model_agg


def FedAvgWithHead(server, models, distributed):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        if distributed:
            server.model.last.weight.data = models[0].module.model.last.weight.data
            server.model.last.bias.data = models[0].module.model.last.bias.data
        else:
            server.model.last.weight.data = models[0].model.last.weight.data
            server.model.last.bias.data = models[0].model.last.bias.data


        # for name, param in server.head.named_parameters():
        for i in range(1, len(models)):
            if distributed:
                server.model.last.weight.data += models[i].module.model.last.weight.data
                server.model.last.bias.data += models[i].module.model.last.bias.data
            else:
                server.model.last.weight.data += models[i].model.last.weight.data
                server.model.last.bias.data += models[i].model.last.bias.data
        # server.model.last.weight = torch.div(server.model.last.weight, len(models)).clone()
        # server.model.last.bias = torch.div(server.model.last.bias, len(models)).clone()

        server.model.last.weight.data = torch.div(server.model.last.weight, len(models)).clone()
        server.model.last.bias.data = torch.div(server.model.last.bias, len(models)).clone()



    #Aggregate the E_promp
    with torch.no_grad():
        if distributed:
           for param1, param2 in zip(server.model.prompt.parameters(), models[0].module.model.prompt.parameters()):
                param1.data = param2.data
        else:
            # server.e_prompt.prompt_key.data = models[0].e_prompt.prompt_key.data
            # server.e_prompt.prompt.data = models[0].e_prompt.prompt.data
            # model[0].prompt.e_p_0.d = model[0].prompt.e_p_0.data
            # model[0].prompt.e_k_0.data
            # model[0].prompt.e_a_0.data
            # model[0].prompt.e_p_1.data
            # model[0].prompt.e_k_1.data
            # model[0].prompt.e_a_1.data
            # model[0].prompt.e_p_2.data
            # model[0].prompt.e_k_2.data
            # model[0].prompt.e_a_2.data
            # model[0].prompt.e_p_3.data
            # model[0].prompt.e_k_3.data
            # model[0].prompt.e_a_3.data
            # model[0].prompt.e_p_4.data
            # model[0].prompt.e_k_4.data
            # model[0].prompt.e_a_4.data
            # for c in range(1, n):
            for param1, param2 in zip(server.model.prompt.parameters(), models[0].model.prompt.parameters()):
                param1.data = param2.data
                # print(param2)
                # param1.data = param2.data
                # param1.data = (param1.data * 1.0) + (param2.data * 1.0) 

        
        for i in range(1, len(models)):
            if distributed:
                for param1, param2 in zip(server.model.prompt.parameters(), models[i].module.model.prompt.parameters()):
                    param1.data += param2.data
            else:
                for param1, param2 in zip(server.model.prompt.parameters(), models[i].model.prompt.parameters()):
                    param1.data += param2.data
        # model_agg.e_prompt.prompt_key = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt_key , len(models)))
        # model_agg.e_prompt.prompt = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt , len(models)))
        # server.e_prompt.prompt.data = torch.div(server.e_prompt.prompt.data , len(models)).clone()
        for param1 in server.model.prompt.parameters():
                param1.data = torch.div(param1.data,len(models)).clone()

    print("Agregating e_prompt done")


def FedAvgWithHead2(server, models, distributed):
    
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        if distributed:
            server.model.last.weight.data = models[0].module.model.last.weight.data
            server.model.last.bias.data = models[0].module.model.last.bias.data
        else:
            server.model.last.weight.data = models[0].model.last.weight.data
            server.model.last.bias.data = models[0].model.last.bias.data


        # for name, param in server.head.named_parameters():
        for i in range(1, len(models)):
            if distributed:
                server.model.last.weight.data += models[i].module.model.last.weight.data
                server.model.last.bias.data += models[i].module.model.last.bias.data
            else:
                server.model.last.weight.data += models[i].model.last.weight.data
                server.model.last.bias.data += models[i].model.last.bias.data
        # server.model.last.weight = torch.div(server.model.last.weight, len(models)).clone()
        # server.model.last.bias = torch.div(server.model.last.bias, len(models)).clone()

        server.model.last.weight.data = torch.div(server.model.last.weight, len(models)).clone()
        server.model.last.bias.data = torch.div(server.model.last.bias, len(models)).clone()



    #Aggregate the E_promp
    with torch.no_grad():
        if distributed:
           for param1, param2 in zip(server.model.prompt.parameters(), models[0].module.model.prompt.parameters()):
                param1.data = param2.data
        else:
            server.model.prompt.e_p_0.data = models[0].model.prompt.e_p_0.data
            server.model.prompt.e_k_0.data = models[0].model.prompt.e_k_0.data
            server.model.prompt.e_a_0.data = models[0].model.prompt.e_a_0.data
            server.model.prompt.e_p_1.data = models[0].model.prompt.e_p_1.data
            server.model.prompt.e_k_1.data = models[0].model.prompt.e_k_1.data
            server.model.prompt.e_a_1.data = models[0].model.prompt.e_a_1.data
            server.model.prompt.e_p_2.data = models[0].model.prompt.e_p_2.data
            server.model.prompt.e_k_2.data = models[0].model.prompt.e_k_2.data
            server.model.prompt.e_a_2.data = models[0].model.prompt.e_a_2.data
            server.model.prompt.e_p_3.data = models[0].model.prompt.e_p_3.data
            server.model.prompt.e_k_3.data = models[0].model.prompt.e_k_3.data
            server.model.prompt.e_a_3.data = models[0].model.prompt.e_a_3.data
            server.model.prompt.e_p_4.data = models[0].model.prompt.e_p_4.data
            server.model.prompt.e_k_4.data = models[0].model.prompt.e_k_4.data
            server.model.prompt.e_a_4.data = models[0].model.prompt.e_a_4.data
            # for c in range(1, n):
       
        
        for i in range(1, len(models)):
            if distributed:
                for param1, param2 in zip(server.model.prompt.parameters(), models[i].module.model.prompt.parameters()):
                    param1.data += param2.data
            else:
                server.model.prompt.e_p_0.data += models[i].model.prompt.e_p_0.data
                server.model.prompt.e_k_0.data += models[i].model.prompt.e_k_0.data
                server.model.prompt.e_a_0.data += models[i].model.prompt.e_a_0.data
                server.model.prompt.e_p_1.data += models[i].model.prompt.e_p_1.data
                server.model.prompt.e_k_1.data += models[i].model.prompt.e_k_1.data
                server.model.prompt.e_a_1.data += models[i].model.prompt.e_a_1.data
                server.model.prompt.e_p_2.data += models[i].model.prompt.e_p_2.data
                server.model.prompt.e_k_2.data += models[i].model.prompt.e_k_2.data
                server.model.prompt.e_a_2.data += models[i].model.prompt.e_a_2.data
                server.model.prompt.e_p_3.data += models[i].model.prompt.e_p_3.data
                server.model.prompt.e_k_3.data += models[i].model.prompt.e_k_3.data
                server.model.prompt.e_a_3.data += models[i].model.prompt.e_a_3.data
                server.model.prompt.e_p_4.data += models[i].model.prompt.e_p_4.data
                server.model.prompt.e_k_4.data += models[i].model.prompt.e_k_4.data
                server.model.prompt.e_a_4.data += models[i].model.prompt.e_a_4.data
        # model_agg.e_prompt.prompt_key = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt_key , len(models)))
        # model_agg.e_prompt.prompt = torch.nn.Parameter(torch.div(model_agg.e_prompt.prompt , len(models)))
        # server.e_prompt.prompt.data = torch.div(server.e_prompt.prompt.data , len(models)).clone()
        
        # for param1 in server.model.prompt.parameters():
        #         param1.data = torch.div(param1.data,len(models)).clone()
        server.model.prompt.e_p_0.data = torch.div(server.model.prompt.e_p_0.data,len(models)).clone()
        server.model.prompt.e_k_0.data = torch.div(server.model.prompt.e_k_0.data,len(models)).clone()
        server.model.prompt.e_a_0.data = torch.div(server.model.prompt.e_a_0.data,len(models)).clone()
        server.model.prompt.e_p_1.data = torch.div(server.model.prompt.e_p_1.data,len(models)).clone()
        server.model.prompt.e_k_1.data = torch.div(server.model.prompt.e_k_1.data,len(models)).clone()
        server.model.prompt.e_a_1.data = torch.div(server.model.prompt.e_a_1.data,len(models)).clone()
        server.model.prompt.e_p_2.data = torch.div(server.model.prompt.e_p_2.data,len(models)).clone()
        server.model.prompt.e_k_2.data = torch.div(server.model.prompt.e_k_2.data,len(models)).clone()
        server.model.prompt.e_a_2.data = torch.div(server.model.prompt.e_a_2.data,len(models)).clone()
        server.model.prompt.e_p_3.data = torch.div(server.model.prompt.e_p_3.data,len(models)).clone()
        server.model.prompt.e_k_3.data = torch.div(server.model.prompt.e_k_3.data,len(models)).clone()
        server.model.prompt.e_a_3.data = torch.div(server.model.prompt.e_a_3.data,len(models)).clone()
        server.model.prompt.e_p_4.data = torch.div(server.model.prompt.e_p_4.data,len(models)).clone()
        server.model.prompt.e_k_4.data = torch.div(server.model.prompt.e_k_4.data,len(models)).clone()
        server.model.prompt.e_a_4.data = torch.div(server.model.prompt.e_a_4.data,len(models)).clone()   

    print("Agregating e_prompt done")



# def FedDistribute(server,clients,distributed):
#     # model_agg = copy.deepcopy(models[0])
#     #Aggregate the G_prompt
#     with torch.no_grad():
#         for i in range(0, len(clients)):
#             if distributed:
#                 clients[i].module.g_prompt.data = server.g_prompt.data.clone()
#                 clients[i].module.e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
#                 clients[i].module.e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()
#             else:
#                 clients[i].g_prompt.data = server.g_prompt.data.clone()
#                 clients[i].e_prompt.prompt.data = server.e_prompt.prompt.data.clone()
#                 clients[i].e_prompt.prompt_key.data = server.e_prompt.prompt_key.data.clone()
#     print("Distributing g_prompt and e_prompt done")




def FedDistributeWithHead(server,clients,distributed):
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    with torch.no_grad():
        for i in range(0, len(clients)):
            if distributed:
                for param1, param2 in zip(server.model.prompt.parameters(), clients[i].module.model.prompt.parameters()):
                    param2.data = param1.data
                
                clients[i].module.model.last.weight.data = server.model.last.weight.data.clone()
                clients[i].module.model.last.bias.data = server.model.last.bias.data.clone() 
     
            else:
                for param1, param2 in zip(server.model.prompt.parameters(), clients[i].model.prompt.parameters()):
                    param2.data = param1.data.clone()

                clients[i].model.last.weight.data = server.model.last.weight.data.clone()
                clients[i].model.last.bias.data = server.model.last.bias.data.clone() 
                # clients[i].head.load_state_dict(server.head.state_dict())

                # clients[i].g_prompt.requires_grad=True
                # clients[i].e_prompt.prompt.requires_grad=True
                # clients[i].e_prompt.prompt_key.requires_grad=True
                # clients[i].head.requires_grad=True


    print("Distributing e_prompt done")
    # model_agg.g_prompt[k] = torch.div(model_agg.g_prompt[k], len(models))

    # for k in model_agg.g_prompt.prompt_key:
    #     for i in range(1, len(models)):
    #         model_agg.g_prompt.prompt_key[k] += models[i].g_prompt[k]
    #     model_agg.g_prompt[k] = torch.div(model_agg.g_prompt[k], len(models))
    # return model_agg


def FedDistributeWithHead2(server,clients,distributed):
    # model_agg = copy.deepcopy(models[0])
    #Aggregate the G_prompt
    # models = clients
    with torch.no_grad():
        for i in range(0, len(clients)):
            if distributed:
                for param1, param2 in zip(server.model.prompt.parameters(), clients[i].module.model.prompt.parameters()):
                    param2.data = param1.data
                
                clients[i].module.model.last.weight.data = server.model.last.weight.data.clone()
                clients[i].module.model.last.bias.data = server.model.last.bias.data.clone() 
     
            else:
                # clients[i].model.prompt.e_p_0.data = server.model.prompt.e_p_0.data.clone()
                # clients[i].model.prompt.e_k_0.data = server.model.prompt.e_k_0.data.clone()
                # clients[i].model.prompt.e_a_0.data = server.model.prompt.e_a_0.data.clone()
                # clients[i].model.prompt.e_p_1.data = server.model.prompt.e_p_1.data.clone()
                # clients[i].model.prompt.e_k_1.data = server.model.prompt.e_k_1.data.clone()
                # clients[i].model.prompt.e_a_1.data = server.model.prompt.e_a_1.data.clone()
                # clients[i].model.prompt.e_p_2.data = server.model.prompt.e_p_2.data.clone() 
                # clients[i].model.prompt.e_k_2.data = server.model.prompt.e_k_2.data.clone()
                # clients[i].model.prompt.e_a_2.data = server.model.prompt.e_a_2.data.clone() 
                # clients[i].model.prompt.e_p_3.data = server.model.prompt.e_p_3.data.clone() 
                # clients[i].model.prompt.e_k_3.data = server.model.prompt.e_k_3.data.clone() 
                # clients[i].model.prompt.e_a_3.data = server.model.prompt.e_a_3.data.clone() 
                # clients[i].model.prompt.e_p_4.data = server.model.prompt.e_p_4.data.clone()
                # clients[i].model.prompt.e_k_4.data = server.model.prompt.e_k_4.data.clone()
                # clients[i].model.prompt.e_a_4.data = server.model.prompt.e_a_4.data.clone()

                # clients[i].model.last.weight.data = server.model.last.weight.data.clone()
                # clients[i].model.last.bias.data = server.model.last.bias.data.clone() 
                # clients[i].head.load_state_dict(server.head.state_dict())

                for param1, param2 in zip(server.model.parameters(), clients[i].model.parameters()):
                    param2.data = param1.data

                # clients[i].g_prompt.requires_grad=True
                # clients[i].e_prompt.prompt.requires_grad=True
                # clients[i].e_prompt.prompt_key.requires_grad=True
                # clients[i].head.requires_grad=True


    print("Distributing e_prompt done")


# def FedAvgPrototype(clients_prototype,task_id,classes_per_task):
    
#     # model_agg = copy.deepcopy(models[0])
#     #Aggregate the G_prompt
#     min_c = task_id*classes_per_task
#     max_c = (task_id+1)*classes_per_task
    
#     global_prototype = {}
#     proto_size = list(clients_prototype[0].values())[0].shape[0]
#     for c in range(min_c,max_c+1):
#         counter = 0
#         proto_c = np.zeros(proto_size)
#         for n in range(len(clients_prototype)):
#             if c in clients_prototype[n]:
#                 proto_c = proto_c + clients_prototype[n][c]
#                 counter = counter + 1

#         if counter > 0:
#             proto_c = proto_c / counter
#             global_prototype[c] = proto_c
    
#     print("Agregating prototype done")
#     return global_prototype




# def FedAvgPrototype2(clients_prototype, clients_prototype_var, clients_weight, task_id, classes_per_task):
    
#     # model_agg = copy.deepcopy(models[0])
#     #Aggregate the G_prompt
#     min_c = task_id*classes_per_task
#     max_c = (task_id+1)*classes_per_task
    
#     global_prototype = {}
#     global_prototype_var = {}
#     proto_size = list(clients_prototype[0].values())[0].shape[0]

#     for c in range(min_c,max_c+1):
        
#         counter = 0
#         total_weight = 0
#         proto_c = np.zeros(proto_size)
#         proto_var_c = np.zeros(proto_size)
        
#         for n in range(len(clients_prototype)):
#             if c in clients_prototype[n]:
#                 proto_c = proto_c + (clients_weight[n]*clients_prototype[n][c])
#                 peoro_var_c = proto_var_c + ((clients_prototype_var[n][c]+(clients_prototype[n][c]*clients_prototype[n][c]))*clients_weight[n])

#                 total_weight =  total_weight + clients_weight[n]
#                 counter = counter + 1

#         if counter > 0:
#             # proto_c = proto_c / counter
#             proto_c = proto_c / total_weight
#             proto_var_c = (proto_var_c/total_weight) - (proto_c*proto_c)
#             global_prototype[c] = proto_c
#             global_prototype_var[c] = proto_var_c
    
#     print("Agregating prototype done")
#     return global_prototype, global_prototype_var