"""

"""
import torch
import numpy as np


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        #self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            print(grad_in[0].shape)
            self.gradients = grad_in[0]

        # Register hook to the first layer
        layers = list(self.model.modules())
        first_layer = layers[0]
        layers2 = list(first_layer._modules)
        first_layer2 = layers2[0]
        layer3 = self.model._modules[first_layer2]
        first_layer3 = list(layer3._modules)[0]
        layer4 = self.model._modules[first_layer2]._modules[first_layer3]
        first_layer4 = list(layer4._modules)[0]
        layer5 = self.model._modules[first_layer2]._modules[first_layer3]._modules[first_layer4]
        print(layer5)
        layer5.register_backward_hook(hook_function)


    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output,_,_ = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list: #[1,3,1024]
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        return integrated_grads[0]


# =============================================================================
# if __name__ == '__main__':
#     # Get params
#     target_example = 0  # Snake
#     (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#         get_example_params(target_example)
#     # Vanilla backprop
#     IG = IntegratedGradients(pretrained_model)
#     # Generate gradients
#     integrated_grads = IG.generate_integrated_gradients(prep_img, target_class, 100)
#     # Convert to grayscale
#     grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
#     # Save grayscale gradients
#     save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_gray')
#     print('Integrated gradients completed.')
# =============================================================================
