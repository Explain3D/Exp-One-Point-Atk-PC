#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:19:34 2021

@author: 
"""

import torch

def euclidean_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    return torch.sum(torch.diagonal(torch.cdist(a,b,p=2)))

def pairwise_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    a = a.permute(0,2,1)
    b = b.permute(0,2,1)
    return torch.cdist(a,b,p=2)
    

def chamfer(a, b):
    M = pairwise_distances(a, b)
    return ((M.min(1)[0].sum(1))/a.shape[1] + (M.min(2)[0].sum(1))/b.shape[1])[0]


def sgd_hausdorff_dis(a,b):
    M = pairwise_distances(a, b)
    SHD = torch.max(M[0].min(1)[0])
    return SHD
    
    
def bid_hausdorff_dis(a,b):
    d_ab = sgd_hausdorff_dis(a, b)
    d_ba = sgd_hausdorff_dis(b, a)
    return torch.max(d_ab,d_ba)
    
#a = torch.Tensor([[[1,1,1],[1,1,1],[1,1,1]]])
#b = torch.Tensor([[[2,2,3],[2,2,2],[2,2,2]]])
#print(euclidean_distances(a,b))
#print(chamfer(a,b),'Cham')
#print(sgd_hausdorff_dis(a, b))
#print(bid_hausdorff_dis(a, b))
