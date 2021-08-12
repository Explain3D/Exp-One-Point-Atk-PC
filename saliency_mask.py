import torch

class SaliencyMask(object):
    def __init__(self, model):
        if torch.cuda.is_available() == True:
            self.model = model.cuda()
        else: 
            self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()

    def get_mask(self, image_tensor, target_class=None):
        raise NotImplementedError('A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
