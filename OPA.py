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

stop_threshold = 1e-5
noise_weight = 10**(-0.5)

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
    IG_steps=25,
    n_points=1,
    verbose=False,
    using_softmax_neuron=False,
    penalize_dis=False,
    sec_act_noise=False,
    optimizer='Adam'
    ):

    print("Layer name: ", layer_name)
    print("Target class: ", ori_cls)
    print("Alpha: ", alpha)
    print("Distance Penalizing: ", penalize_dis)
    print("Beta: ", beta)
    print("Optimizer: ", optimizer)
    
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
    contr_index = np.argsort(contri, axis=-1, kind='quicksort', order=None)[ : :-1]
    print(contr_index.shape)
    print('Number of positive points: ',np.sum(contri>0))
    
    #Parameters for Momentum
    v = torch.zeros_like(input)
    v_adam = torch.zeros_like(input)
    s_adam = torch.zeros_like(input)
    
    activation_recorder = []
    mean_recorder_last = float('inf')
    mean_noiser_last = float('inf')
    
    tar_cls = 1
    while(True):
        step += 1
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
        
# =============================================================================
#         alpha_weight = torch.abs(layer_out[0][unit_to_opt]-layer_out[0][torch.argsort(layer_out[0])[-2]]).detach()
#         alpha_weight = torch.exp(torch.sqrt(alpha_weight))
#         print(alpha_weight)
# =============================================================================
            
        if using_softmax_neuron == True:
            sftmax = torch.nn.functional.softmax(layer_out[0])
            ls = torch.log(sftmax)
            gradient_optimizer = alpha * ls[unit_to_opt]

        elif using_softmax_neuron  == False:
            if sec_act_noise == True:
                gradient_optimizer = alpha * (layer_out[0][unit_to_opt] - layer_out[0][torch.topk(layer_out[0], 2).indices[-1]])
            elif sec_act_noise == False:
                gradient_optimizer = alpha * (layer_out[0][unit_to_opt])
            
        
        if penalize_dis == True:
            total_dis = 0
            for pa in range(n_points):
                #total_dis = dis_utils_torch.chamfer(input,prototype)
                total_dis = dis_utils_torch.euclidean_distances(input[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1),prototype[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1))
                #total_dis += dis_utils_torch.sgd_hausdorff_dis(input[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1),prototype[0,:,contr_index[pa]].unsqueeze(0).unsqueeze(-1))
            gradient_optimizer +=  beta * total_dis
            
# =============================================================================
#         if sec_act_noise == True:
#             sec_act_unit = alpha * layer_out[0][torch.topk(layer_out[0], 2).indices[-1]]
# =============================================================================
            
        
##############################################################################
#Add gradients
        gradient_optimizer.backward(retain_graph=True)
        input_grad = input.grad.clone()      
        input_grad_masked = torch.zeros_like(input_grad)
# =============================================================================
#         if sec_act_noise == True:
#             input.grad.zero_()
#             sec_act_unit.backward(retain_graph=True)
#             sec_grad = input.grad.clone()
# =============================================================================
        for pa in range(n_points):
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
        
        # Add some noise to escape from getting stuck at local optimal
        noise_threshold = 5e-1
        if (len(activation_recorder)) >= 25 and (step % 25 == 0):
            print('Noiser detecting')
            noiser = activation_recorder[-10:]
            var_noiser = np.var(np.asarray(noiser))
            mean_noiser_new = np.mean(np.asarray(noiser))
            print('var: ',var_noiser, 'last: ',mean_noiser_last, 'new: ', mean_noiser_new)
            if var_noiser <= noise_threshold and mean_noiser_new >= mean_noiser_last:
                print('var: ',var_noiser, 'last: ',mean_noiser_last, 'new: ', mean_noiser_new)
# =============================================================================
#                 if sec_act_noise == True:
#                     sec_grad_masked = torch.zeros_like(sec_grad)
#                     for pa in range(n_points):
#                         sec_grad_masked[0,:,contr_index[pa]] = sec_grad[0,:,contr_index[pa]]
#                         #Momentum
#                         M_sec = 0.9
#                         v_sec = M_sec * v_sec - sec_grad_masked
#                         input = torch.add(input, v_sec)
#                         print('Activate stopped, Second max activation noise added')
#                         print(sec_grad_masked[0,:,contr_index[pa]])
#                 else:
# =============================================================================
                gauss_noise = torch.zeros_like(input_grad)
                for pa in range(n_points):
                    gauss_noise[0,:,contr_index[pa]] = noise_weight * torch.randn(1) 
                    input = torch.add(input, gauss_noise)
                    print('Activate stopped, add noise')
            mean_noiser_last = mean_noiser_new

        input.requires_grad_(True)

        if verbose:
            print("\r",'step: ', step, 'ori_cls activation: ', layer_out[0][ori_cls].detach(),end="",flush=True)
            
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
            activation_recorder.append(layer_out[0][unit_to_opt].detach().cpu().numpy())
        else:
            activation_recorder.append(layer_out[0][unit_to_opt].detach().numpy())

        if torch.cuda.is_available() == True:
            cur_class = torch.argmax(layer_out[0]).detach().cpu().numpy()
        else:
            cur_class = torch.argmax(layer_out[0]).detach().numpy()
        
        
        if cur_class != ori_cls:
            break
            
        if (len(activation_recorder) >= 1000 and (step % 1000 == 0)):
            activation_recorder = activation_recorder[-1000:]
            var_recorder = np.var(np.asarray(activation_recorder))
            mean_recorder_new = np.mean(np.asarray(activation_recorder))
            print('last: ', mean_recorder_last, 'new: ', mean_recorder_new, 'var: ', var_recorder)
            if (var_recorder <= stop_threshold and mean_recorder_new >= mean_recorder_last) or (step >= 2000):
                state = 'Fail'
                return state,best_img,ori_logits,max_other_logits
            mean_recorder_last = mean_recorder_new
            
    return state,best_img,ori_logits,max_other_logits

