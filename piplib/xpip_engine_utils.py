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



#CLassift single data with set of prorotypes
def classify_with_proto(data, paras, max_label):
    # print("Testing with prototype-based cosine cosine similarity")
    scale_factor =  10
    logits = []
  
    all_proto_keys = list(paras.keys())
    for i in range(0,max_label):
        if i in all_proto_keys:
            proto = paras[i]
        else:
            if(len(all_proto_keys)>1):
                idx =  random.sample(all_proto_keys, 2)
            else:
                idx = all_proto_keys * 2
            proto = (paras[idx[0]]+paras[idx[1]]) * 0.5 
        logit = scale_factor * torch.nn.functional.cosine_similarity(
                data,
                proto,
                dim=-1)
        logits.append(logit)

    logits = torch.stack(logits).unsqueeze(0)
    # print(logits.shape)
        # logits = scale_factor * torch.nn.functional.cosine_similarity(
        #     data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
        #     paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
        #     dim=-1)
    
    # print("finish cosine-based logits")
    # print(logits.shape)
    
    return logits


def classify_with_proto2(data, paras, min_label, max_label):
    # print("Testing with prototype-based cosine cosine similarity")
    scale_factor =  10
    logits = []
  
    all_proto_keys = list(paras.keys())
    for i in range(min_label,max_label):
        if i in all_proto_keys:
            proto = paras[i]
        else:
            if(len(all_proto_keys)>1):
                idx =  random.sample(all_proto_keys, 2)
            else:
                idx = all_proto_keys * 2
            proto = (paras[idx[0]]+paras[idx[1]]) * 0.5 
        logit = scale_factor * torch.nn.functional.cosine_similarity(
                data,
                proto,
                dim=-1)
        logits.append(logit)

    logits = torch.stack(logits).unsqueeze(0)
    # print(logits.shape)
        # logits = scale_factor * torch.nn.functional.cosine_similarity(
        #     data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
        #     paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
        #     dim=-1)
    
    # print("finish cosine-based logits")
    # print(logits.shape)
    
    return logits



def classify_with_proto_inbatch(data, paras, max_label):
    # print("Testing with prototype-based cosine cosine similarity")
    scale_factor =  10
    logits = None
    isFirst = True

    all_proto_keys = list(paras.keys())
    all_proto = []
    for i in range(0,max_label):
        if i in all_proto_keys:
            proto = paras[i]
        else:
            if(len(all_proto_keys)>1):
                idx =  random.sample(all_proto_keys, 2)
            else:
                idx = all_proto_keys * 2
            proto = (paras[idx[0]]+paras[idx[1]]) * 0.5

        all_proto.append(proto) 


    for i in range(data.shape[0]):
        logit = []
        for j in range(len(all_proto)):
            d = scale_factor * torch.nn.functional.cosine_similarity(
                    data[i],
                    all_proto[j],
                    dim=-1)
            logit.append(d)

        logit = torch.stack(logit).unsqueeze(0)
        if isFirst:
            logits = logit
            isFirst = False
        else:
            logits = torch.cat((logits,logit), dim=0)

        

    # logits = torch.stack(logits).unsqueeze(0)
    # print(logits.shape)
        # logits = scale_factor * torch.nn.functional.cosine_similarity(
        #     data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
        #     paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
        #     dim=-1)
    
    # print("finish cosine-based logits")
    # print(logits.shape)
    
    return logits


def classify(data, paras):
    # print("data shape")
    # print(data.shape)
    # print(paras.shape)
    scale_factor =  10
    logits = scale_factor * torch.nn.functional.cosine_similarity(
        data.unsqueeze(2).expand(-1, -1, paras.shape[1], -1),
        paras.unsqueeze(1).expand(-1, data.shape[1], -1, -1),
        dim=-1)
    return logits


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy