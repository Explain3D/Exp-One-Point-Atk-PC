Explainability-Aware One Point Attack for Point Cloud Neural Networks
===================
Pytorch implementation for Explainability-Aware One Point Attack for Point Cloud Neural Networks.
Point cloud neural networks are based on [this repo](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Please follow the instructions to train networks before attacking them.

Environments
---------------
Python >= 3.6 Pytorch >= 1.6.0

Usage
------------
Before running the code, move the test data file list modelnet40_test_adv.txt to ./data/modelnet40_normal_resampled/ or generate user-defined number of test files using sample_adv_test_data.py (also should be placed in the ./data/modelnet40_normal_resampled/ path):

    python sample_adv_test_data.py
    
Create visualization path:

    mkdir visu
    cd visu
    mkdir output
Test and visualize one instance randomly picked up from dataset with OPA and CTA respectively:

    python Test_single_ins_OPA.py
    python Test_single_ins_CTA.py
    
![Image text](https://github.com/Explain3D/Exp-One-Point-Atk-PC/blob/main/pic/exp_opa.png?raw=true)
Quantitatively evaluate the attack performance:

    python Eval_OPA.py
    python Eval_CTA.py
    
[![Image text]](https://github.com/Explain3D/Exp-One-Point-Atk-PC/blob/main/pic/exp_cta.png?raw=true)
