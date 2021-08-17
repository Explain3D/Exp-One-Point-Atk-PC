#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:19:34 2021

@author: 
"""

import torch

def pairwise_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    a = a.permute(0,2,1)
    b = b.permute(0,2,1)
    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", b.shape)
    return (a.unsqueeze(2) - b.unsqueeze(1)).abs().pow(p).sum(3)


def chamfer(a, b):
    M = pairwise_distances(a, b)
    return (M.min(1)[0].sum(1) + M.min(2)[0].sum(1))[0]
