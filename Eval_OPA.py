#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:55 2021

@author: tan
"""

import OPA 
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

def check_num_pc_changed(adv,ori):
    logits_mtx = np.logical_and.reduce(adv==ori,axis=1)
    return np.sum(logits_mtx==False)

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

def generate_adv(prototype, ori_class, filename):
# =============================================================================
#     np.save('visu/Ori_pt.npy',np.squeeze(prototype))
#     ori_pt = o3d.geometry.PointCloud()
#     ori_pt.points = o3d.utility.Vector3dVector(prototype[0,:,0:3])
#     o3d.io.write_point_cloud('visu/Ori_pt.ply', ori_pt)
# =============================================================================
    ipt = torch.from_numpy(prototype).float()
    ipt = ipt.permute(0,2,1)
    ipt.requires_grad_(True)
    activation_dictionary = {}
    classifier.fc3.register_forward_hook(OPA.layer_hook(activation_dictionary, selected_layer))
    #steps = 10              # perform 100 iterations                  # flamingo class of Imagenet
    IG_steps = 50
    if torch.cuda.is_available() == True:
        alpha = torch.tensor(1e-6).cuda()
        beta = alpha*16
    else:
        alpha = torch.tensor(1e-6)
        beta = alpha*16
    verbose = True              # print activation every step
    using_softmax_neuron = False  # whether to optimize the unit after softmax
    penalize_dis = False
    optimizer = 'Adam'
    target_att = False
    state,output,ori_logits,max_other_logits = OPA.act_max(network=classifier,
                    input=ipt,
                    layer_activation=activation_dictionary,
                    layer_name=selected_layer,
                    ori_cls=ori_class,
                    IG_steps=IG_steps,
                    alpha=alpha,
                    beta=beta,
                    verbose=verbose,
                    using_softmax_neuron=using_softmax_neuron,
                    penalize_dis=penalize_dis,
                    optimizer=optimizer,
                    
                    )
    res = output.permute(0,2,1)
    if torch.cuda.is_available() == True:
        res = res.detach().cpu().numpy()
    else:
        res = res.detach().numpy()
    res = res[0]
    res = np.float32(res)
    
    ##########################################
    #Save npy for trasferability test
    #if state == 'Suc':
        #trans_path = 'transferability/pn_med/'
        #np.save(trans_path+filename,res)
    ##########################################

    #adv_pc = o3d.geometry.PointCloud()
    #adv_pc.points = o3d.utility.Vector3dVector(res[:,0:3])
    #o3d.io.write_point_cloud('visu/output/'+ filename+'.ply', adv_pc)
    
    target_pro = np.float32(np.squeeze(prototype))
    Hausdorff_dis2 = dun.bid_hausdorff_dis(res, target_pro)
    cham_dis = dun.chamfer(res, target_pro)
    num_perturbed_pc = check_num_pc_changed(res,target_pro)
    return state,Hausdorff_dis2, cham_dis, num_perturbed_pc

def get_pred(points,classifier):
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    if torch.cuda.is_available() == True:
        points = points.cuda()
        classifier = classifier.cuda()
    classifier = classifier.eval()
    pred, bf_sftmx , _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    return pred_choice, SHAPE_NAMES[pred_choice]


test_file = 'data/modelnet40_normal_resampled/modelnet40_test_adv.txt'
num_class = 40
target_class = 0 #Plane
experiment_dir = 'log/classification/pointnet_cls_msg'
data_dir = 'data/modelnet40_normal_resampled'
model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 
classifier = MODEL.get_model(num_class,normal_channel=False)
if torch.cuda.is_available() == True:
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
else:
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
print(classifier.load_state_dict(checkpoint['model_state_dict']))
layer_name = list(classifier.state_dict().keys())
selected_layer = 'fc3'

total_num_ins = 0
total_suc_num = 0
total_avg_Hausdorff_dis = 0
total_avg_cham_dis = 0
class_num_ins = np.zeros((num_class))
class_suc_num = np.zeros((num_class))
class_avg_Hausdorff_dis = np.zeros((num_class))
class_avg_cham_dis = np.zeros((num_class))
avg_number_points_changed = 0
for line in open(test_file):
    ori_class = line[:line.rfind('_')]
    cur_file = data_dir + '/' + ori_class + '/' +  line[:-1] + '.txt'
    prototype =  np.expand_dims((sampling(np.loadtxt(cur_file,delimiter=',')[:,0:3], 1024)),0)
    cur_cls_num, pred_class = get_pred(prototype,classifier)
    if ori_class == pred_class:    #Only generate instances being classified correctly
            class_num_ins[cur_cls_num] += 1
            total_num_ins += 1
            save_name = ori_class + str(int(class_num_ins[cur_cls_num]))
            saving_path = 'visu/output/'
            #if (save_name + '.ply') in os.listdir(saving_path):
                #print('Already processed, skip!')
                #continue
            if torch.cuda.is_available() == True:
                state, Hausdorff_dis, cham_dis, num_perturbed_pc = generate_adv(prototype,cur_cls_num.detach().cpu().numpy()[0],save_name)
            else:
                state, Hausdorff_dis, cham_dis, num_perturbed_pc = generate_adv(prototype,cur_cls_num.detach().numpy()[0],save_name)
            if state == 'Suc':
                class_suc_num[cur_cls_num] += 1
                total_suc_num += 1
                class_avg_Hausdorff_dis[cur_cls_num] += Hausdorff_dis
                class_avg_cham_dis[cur_cls_num] += cham_dis
                total_avg_Hausdorff_dis += Hausdorff_dis
                total_avg_cham_dis += cham_dis
                avg_number_points_changed += num_perturbed_pc
                print('Hausdorff distance: ', "%e"%Hausdorff_dis)
                print('Chamfer distance: ', "%e"%cham_dis)
                print('Number of points changed: ', num_perturbed_pc)
            elif state == 'Fail':
                print('Finding one-point adversarial example failed!')
class_suc_rate = class_suc_num / class_num_ins
class_avg_Hausdorff_dis = class_avg_Hausdorff_dis / class_suc_num
class_avg_cham_dis = class_avg_cham_dis / class_suc_num
print('\n')
print('*****************************************************************')
print('Average Class Hausdorff Distance :', class_avg_Hausdorff_dis)
print('Average Class Chamfer Distance :', class_avg_cham_dis)
print('Class Success percentage:', class_suc_rate)
print('*****************************************************************')
print('\n')
total_avg_Hausdorff_dis /= total_suc_num
total_avg_cham_dis /= total_suc_num
total_suc_rate = total_suc_num/total_num_ins
avg_number_points_changed = avg_number_points_changed/total_suc_num
print('\n')
print('##################################################################')
print('Total number of instance tested: ', total_num_ins)
print('Total Average Hausdorff Distance :', total_avg_Hausdorff_dis)
print('Total Average Chamfer Distance :', total_avg_cham_dis)
print('Total Average number of points perturbed :', avg_number_points_changed)
print('Total Success percentage:', total_suc_rate)
print('##################################################################')
print('\n')


