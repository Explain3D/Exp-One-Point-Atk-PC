#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:19:34 2021

@author: 
"""

import numpy as np
from scipy.spatial import distance_matrix

# =============================================================================
# def point_dis(a,b):
#     dis = 0
#     for i in range(a.shape[0]):
#         dis += np.square(a[i]-b[i])
#     return np.sqrt(dis)
# =============================================================================

def pairwise_distances(a, b):  #(1024,3)
# =============================================================================
#     dis_mtx = np.zeros((a.shape[0],b.shape[0]))
#     for i in range(a.shape[0]):
#         for j in range(b.shape[0]):
#             dis_mtx[i][j] = point_dis(a[i],b[j])
# =============================================================================
    return distance_matrix(a,b)
    
    

def chamfer(a, b):
    M = pairwise_distances(a, b)
    chamfer_dis = np.mean(np.min(M,axis=1)) + np.mean(np.min(M,axis=0))
    return(chamfer_dis)


def sgd_hausdorff_dis(a,b):
    M = pairwise_distances(a, b)
    SHD = np.max(np.min(M,axis=1))
    return(SHD)
    
    
def bid_hausdorff_dis(a,b):
    d_ab = sgd_hausdorff_dis(a, b)
    d_ba = sgd_hausdorff_dis(b, a)
    return np.max([d_ab,d_ba])
    
# =============================================================================
# a = np.array([[1,1,1],[1,1,1],[1,1,1]])
# b = np.array([[2,2,2],[2,2,2],[2,2,2]])
# print(chamfer(a,b),'Cham')
# print(sgd_hausdorff_dis(a, b))
# print(bid_hausdorff_dis(a, b))
# =============================================================================
