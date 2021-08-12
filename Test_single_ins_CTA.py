#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:55 2021

@author: tan
"""

import CTA 
import torch
import os
import importlib
import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
import dis_utils_numpy as dun

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


num_class = 40
ori_class = 0 #Car
target_class = 0 #Plane
experiment_dir = 'log/classification/pointnet_cls_msg'
model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 

classifier = MODEL.get_model(num_class,normal_channel=False)
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
print(classifier.load_state_dict(checkpoint['model_state_dict']))
layer_name = list(classifier.state_dict().keys())
selected_layer = 'fc3'



def check_num_pc_changed(adv,ori):
    logits_mtx = np.logical_and.reduce(adv==ori,axis=1)
    return np.sum(logits_mtx==False)

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled



prototype =  np.expand_dims((sampling(np.loadtxt('data/modelnet40_normal_resampled/airplane/airplane_0046.txt',delimiter=',')[:,0:3], 1024)),0)
np.save('visu/Ori_pt.npy',np.squeeze(prototype))
ori_pt = o3d.geometry.PointCloud()
ori_pt.points = o3d.utility.Vector3dVector(prototype[0,:,0:3])
o3d.io.write_point_cloud('visu/Ori_pt.ply', ori_pt)


ipt = torch.from_numpy(prototype).float()
ipt = ipt.permute(0,2,1)
ipt.requires_grad_(True)

activation_dictionary = {}
classifier.fc3.register_forward_hook(CTA.layer_hook(activation_dictionary, selected_layer))

IG_steps = 25
alpha = torch.tensor(1e-6)
delta = torch.tensor(1)
n_points = 1
verbose = True              # print activation every step
using_softmax_neuron = False  # whether to optimize the unit after softmax
penalize_dis = True
sec_act_noise = False
beta = torch.tensor(1e-5)    #Weighting for penalizing distance
optimizer = 'Adam'
state,output,ori_logits,max_other_logits = CTA.act_max(network=classifier,
                input=ipt,
                layer_activation=activation_dictionary,
                layer_name=selected_layer,
                ori_cls=ori_class,
                IG_steps=IG_steps,
                alpha=alpha,
                beta=beta,
                n_points=n_points,
                verbose=verbose,
                using_softmax_neuron=using_softmax_neuron,
                penalize_dis=penalize_dis,
                optimizer=optimizer,
                target_att='random'
                )
res = output.permute(0,2,1)
if torch.cuda.is_available() == True:
    res = res.cpu().detach().numpy()
else:
    res = res.detach().numpy()
res = res[0]

if state == 'Suc':
    np.save('visu/AM.npy',res)
    plt.plot(ori_logits)
    plt.plot(max_other_logits)
    plt.savefig('visu/unit.jpg')
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(res[:,0:3])
    o3d.io.write_point_cloud('visu/AM.ply', pc)
    print("Generate point cloud AM.ply successful!")
    
    res = np.float32(res)
    target_pro = np.float32(np.squeeze(prototype))
    Hausdorff_dis2 = dun.bid_hausdorff_dis(res, target_pro)
    cham_dis = dun.chamfer(res, target_pro)
    num_preturbed_pc = check_num_pc_changed(res,target_pro)
    
    print('Finding one-point advserial example Successful!')
    print('Hausdorff distance: ', "%e"%Hausdorff_dis2)
    print('Chamfer distance: ', "%e"%cham_dis)
    print('Number of points changed: ', num_preturbed_pc)

else:
    print('Finding one-point advserial example failed!')
