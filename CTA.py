import torch
import torch.nn as nn
import torchvision.models as models


# Reading images
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile
import torch.nn.functional as F
import numpy as np
import dis_utils_torch
from integrated_gradients import IntegratedGradients
import point_cloud_utils as pcu
import os
import random

stop_threshold = 5e-1
noise_weight = 1e-2
SHAPE_NAMES = [line.rstrip() for line in \
    open('data/shape_names.txt')] 
    
"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""

def get_IG(input_tensor,ori_cls,network,IG_steps=25, baseline = 'black'):
    IG = IntegratedGradients(network.eval())
    mask = IG.get_mask(image_tensor=input_tensor,target_class=ori_cls,baseline=baseline,steps=IG_steps)
    return mask

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook


def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""
def act_max(network, 
    input, 
    layer_activation, 
    layer_name, 
    ori_cls,
    alpha,
    beta,
    target_att=False,
    IG_steps=25,
    n_points=1,
    verbose=False,
    using_softmax_neuron=False,
    penalize_dis=False,
    optimizer='Adam'
    ):
    
    if torch.cuda.is_available() == True:
        input = input.cuda()
    
    state = 'Suc'
    prototype = input.clone()
    tmp_ptt = input.clone()
    tmp_ptt = tmp_ptt.detach()
    
    best_img = input
    ori_logits = []
    max_other_logits = []
    step = 0   
    mask = get_IG(tmp_ptt, ori_cls, network, IG_steps, baseline='black')
    contri = np.sum(mask,axis=1)
    contr_index = np.argsort(contri, axis=-1, kind='quicksort', order=None)
    print(contr_index.shape)
    print('Number of positive points: ',np.sum(contri>0))
    
    #Parameters for Momentum
    v = torch.zeros_like(input)
    v_adam = torch.zeros_like(input)
    s_adam = torch.zeros_like(input)
    
    
    if target_att == 'random':
        tar_cls = ori_cls
        while (tar_cls == ori_cls):
            tar_cls = random.randint(0,39)
        print(ori_cls,':ori')
        print('Target class is ', SHAPE_NAMES[tar_cls])
    elif target_att == 'second':
        if torch.cuda.is_available() == True:
            tar_cls = torch.topk(layer_activation[layer_name][0], 2).indices[-1].cpu().detach().numpy()
        else:
            tar_cls = torch.topk(layer_activation[layer_name][0], 2).indices[-1].detach().numpy()
        print('The second max activation is ,', SHAPE_NAMES[tar_cls])

    elif target_att == 'least':
        if torch.cuda.is_available() == True:
            tar_cls = torch.topk(layer_activation[layer_name][0], 40).indices[-1].cpu().detach().numpy()
        else:
            tar_cls = torch.topk(layer_activation[layer_name][0], 40).indices[-1].detach().numpy()
        print('Lowest activation is,', SHAPE_NAMES[tar_cls])
    elif target_att == False:
        print('Untarget attack!')
        
     

    for num_p_per in range(n_points,np.sum(contri>0)):
        cur_step = 0
        mean_recorder_last_ori = float('inf')
        if target_att != False:
            mean_recorder_last_tar = -float('inf')
        activation_recorder_ori = []
        activation_recorder_tar = []
        ori_logits = []
        max_other_logits = []
        input = prototype.clone()
        while(True):
            step += 1
            cur_step += 1
            input.retain_grad() # non-leaf tensor
            #network.zero_grad()
            # Propogate image through network,
            if torch.cuda.is_available() == True:
                network.eval()
            else:
                network.eval()
            network(input)
            layer_out = layer_activation[layer_name]
            unit_to_opt = ori_cls
                
            if using_softmax_neuron == True:
                sftmax = torch.nn.functional.softmax(layer_out[0])
                ls = torch.log(sftmax)
                gradient_optimizer = alpha * ls[unit_to_opt]
    
            elif using_softmax_neuron  == False:
                if target_att != False:
                    gradient_optimizer = alpha * (layer_out[0][unit_to_opt] - layer_out[0][tar_cls])
                elif target_att == False:
                    #gradient_optimizer = alpha * (layer_out[0][unit_to_opt] - layer_out[0][torch.topk(layer_out[0], 2).indices[-1]])
                    gradient_optimizer = alpha * (layer_out[0][unit_to_opt])
                
            
            if penalize_dis == True:
                total_dis = 0
                for pa in range(num_p_per):
                    #total_dis = dis_utils_torch.chamfer(input,prototype)
                    #total_dis = dis_utils_torch.euclidean_distances(input[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1),prototype[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1))
                    total_dis += dis_utils_torch.sgd_hausdorff_dis(input[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1),prototype[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1))
                gradient_optimizer +=  beta * total_dis
            
    ##############################################################################
    #Add gradients
            gradient_optimizer.backward(retain_graph=True)
            input_grad = input.grad.clone()      
            input_grad_masked = torch.zeros_like(input_grad)
            for pa in range(num_p_per):
                input_grad_masked[0,:,contr_index[pa]] = input_grad[0,:,contr_index[pa]]
            
            if optimizer == 'Momentum':      
                #Momentum
                M = 0.9
                v = M * v - input_grad_masked
                input = torch.add(input, v)
            elif optimizer == 'Adam':
                a1 = 1
                #Adam
                b1 = 0.9
                b2 = 0.999
                xi = 1e-8
                v_adam = b1 * v_adam + (1 - b1) * input_grad_masked
                s_adam = b2 * s_adam + (1 - b2) * torch.square(input_grad_masked)
                input = torch.add(input, -a1 * v_adam/torch.sqrt(s_adam + xi))
            
    
            input.requires_grad_(True)
    
            if verbose:
                if torch.cuda.is_available() == True:
                    to_display_ori = layer_out[0][ori_cls].cpu().detach().numpy()
                elif torch.cuda.is_available() == False:
                    to_display_ori = layer_out[0][ori_cls].detach().numpy()
                if target_att == False:
                    print("\r",'step: ', step, 'cur step: ', cur_step, 'ori_cls activation: ', to_display_ori,end="",flush=True)
                elif target_att != False:
                    if torch.cuda.is_available() == True:
                        to_display_tar = layer_out[0][tar_cls].cpu().detach().numpy()
                    elif torch.cuda.is_available() == False:
                        to_display_tar = layer_out[0][tar_cls].detach().numpy()
                    print("\r",'step: ', step, 'cur step: ', cur_step, 'ori_cls activation: ', to_display_ori,
                          'target activation: ', to_display_tar,end="",flush=True)
                
            best_img = input
            if torch.cuda.is_available() == True:
                ori_logits.append(layer_out[0][unit_to_opt].detach().cpu().numpy())
            else:
                ori_logits.append(layer_out[0][unit_to_opt].detach().numpy())
            tmp = layer_out[0].clone()
            tmp[unit_to_opt] *= -1
            if torch.cuda.is_available() == True:
                max_other_logits.append(torch.max(tmp).detach().cpu().numpy())
            else:
                max_other_logits.append(torch.max(tmp).detach().numpy())
            #When original unit is surpassed or failed, stop
            if torch.cuda.is_available() == True:
                activation_recorder_ori.append(layer_out[0][unit_to_opt].detach().cpu().numpy())
                if target_att != False:
                    activation_recorder_tar.append(layer_out[0][tar_cls].detach().cpu().numpy())
            else:
                activation_recorder_ori.append(layer_out[0][unit_to_opt].detach().numpy())
                if target_att != False:
                    activation_recorder_tar.append(layer_out[0][tar_cls].detach().numpy())
    
            if torch.cuda.is_available() == True:
                cur_class = torch.argmax(layer_out[0]).detach().cpu().numpy()
            else:
                cur_class = torch.argmax(layer_out[0]).detach().numpy()
            
            if target_att == False:
                if cur_class != ori_cls:
                    print('\nAttack success, current prediction class: ', SHAPE_NAMES[cur_class])
                    return state,best_img,ori_logits,max_other_logits
            elif target_att != False:
                if cur_class == tar_cls:
                    print('\nCurrent prediction: ', SHAPE_NAMES[cur_class],'Target class: ', SHAPE_NAMES[tar_cls],' Success!')
                    return state,best_img,ori_logits,max_other_logits
                
            if (cur_step >= 25 and cur_step % 25 == 0):
                activation_recorder_set_ori = activation_recorder_ori[-25:]
                if target_att != False:
                    activation_recorder_set_tar = activation_recorder_tar[-25:]
                mean_recorder_new_ori = np.mean(np.asarray(activation_recorder_set_ori))
                if target_att != False:
                    mean_recorder_new_tar = np.mean(np.asarray(activation_recorder_set_tar))
                if target_att != False:
                    print('ori last: ', mean_recorder_last_ori, 'ori new: ', mean_recorder_new_ori,'tar last: ', mean_recorder_last_tar, 'tar new: ', mean_recorder_new_tar)
                elif target_att == False:
                    print('ori last: ', mean_recorder_last_ori, 'ori new: ', mean_recorder_new_ori,'tar last: ')
                
                if target_att != False:
                    if (mean_recorder_new_ori >= mean_recorder_last_ori) or (mean_recorder_new_tar <= mean_recorder_last_tar) or (cur_step >= 1500):
                        print('\n', num_p_per, ' points adversarial searching failed, perturbed points +1\n')
                        break
                elif target_att == False:
                    if (mean_recorder_new_ori >= mean_recorder_last_ori) or (cur_step >= 1500):
                        print('\n', num_p_per, ' points adversarial searching failed, perturbed points +1\n')
                        break
                mean_recorder_last_ori = mean_recorder_new_ori
                if target_att != False:
                    mean_recorder_last_tar = mean_recorder_new_tar
        if step >= 15000:
            state = 'Fail'
            print('Adversarial searching failed!')
            return state,best_img,ori_logits,max_other_logits

