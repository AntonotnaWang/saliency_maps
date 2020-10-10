import numpy as np
import copy
import types
from torch.nn import ReLU
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch

from func.utils import convert_to_grayscale

# Guided-Backpropagation
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, which_layer_to_hook=0):
        print("Guided Backpropagation")
        self.model = model
        self.which_forward_layer_to_hook = which_layer_to_hook
        self.which_backward_layer_to_hook = which_layer_to_hook
        
        #print('model structure')
        #print(self.model.features)
        
        self.gradients = None
        self.feature_map = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_forward_layers()
        self.hook_backward_layers()
    
    def toString(self):
        return "Guided Backpropagation"
    
    def hook_backward_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0][0]
        
        # Register hook
        backward_hook_layer = list(self.model.features._modules.items())[self.which_backward_layer_to_hook][1]
        backward_hook_layer.register_backward_hook(hook_function)
        # print("backward_hook_layer: "+str(self.which_backward_layer_to_hook)+\
        #      " "+str(backward_hook_layer))
    
    def hook_forward_layers(self):
        def hook_function(module, ten_in, ten_out):
            self.feature_map = ten_in[0][0]
            
        # Register hook
        forward_hook_layer = list(self.model.features._modules.items())[self.which_forward_layer_to_hook][1]
        forward_hook_layer.register_forward_hook(hook_function)
        # print("forward_hook_layer: "+str(self.which_forward_layer_to_hook)+\
        #      " "+str(forward_hook_layer))
    
    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_explanation(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Convert Pytorch variable to numpy array
        feature_map_as_arr = copy.deepcopy(self.feature_map.data.numpy())
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel
        gradients_as_arr = copy.deepcopy(self.gradients.data.numpy())
        return gradients_as_arr, feature_map_as_arr

# Gradient (normal backpropagation)
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, which_layer_to_hook=0):
        print("Vanilla Backpropagation")
        self.model = model
        self.which_forward_layer_to_hook = which_layer_to_hook
        self.which_backward_layer_to_hook = which_layer_to_hook
        
        self.gradients = None
        self.feature_map = None
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Hook layer
        self.hook_backward_layers()
        self.hook_forward_layers()
    
    def toString(self):
        return "Vanilla Backpropagation"
    
    def hook_backward_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0][0]

        # Register hook
        hook_layer = list(self.model.features._modules.items())[self.which_backward_layer_to_hook][1]
        hook_layer.register_backward_hook(hook_function)
        # print("backward_hook_layer: "+str(self.which_backward_layer_to_hook)+\
        #      " "+str(hook_layer))
    
    def hook_forward_layers(self):
        def hook_function(module, ten_in, ten_out):
            self.feature_map = ten_in[0][0]
        
        # Register hook
        hook_layer = list(self.model.features._modules.items())[self.which_forward_layer_to_hook][1]
        hook_layer.register_forward_hook(hook_function)
        # print("forward_hook_layer: "+str(self.which_forward_layer_to_hook)+\
        #       " "+str(hook_layer))

    def generate_explanation(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Convert Pytorch variable to numpy array
        feature_map_as_arr = copy.deepcopy(self.feature_map.data.numpy())
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = copy.deepcopy(self.gradients.data.numpy())
        return gradients_as_arr, feature_map_as_arr

# SmoothGrad
class SmoothGrad():
    def __init__(self, pretrained_model, which_layer_to_hook=0, is_guidedbackprop=False):
        print("SmoothGrad")
        if is_guidedbackprop==False:
            self.backprop = VanillaBackprop(pretrained_model, which_layer_to_hook)
        else:
            self.backprop = GuidedBackprop(pretrained_model, which_layer_to_hook)
    
    def toString(self):
        return "SmoothGrad"
    
    def generate_explanation(self, input_image, target_class, param_n = 50, param_sigma_multiplier = 4):
        smooth_grad, smooth_feature_map = self.get_smooth_grad(self.backprop, input_image, target_class, param_n, param_sigma_multiplier)
        return smooth_grad, smooth_feature_map
    
    def get_smooth_grad(self, Backprop, prep_img, target_class, param_n, param_sigma_multiplier):
        """
            Generates smooth gradients of given Backprop type. You can use this with both vanilla
            and guided backprop
        Args:
            Backprop (class): Backprop type
            prep_img (torch Variable): preprocessed image
            target_class (int): target class of imagenet
            param_n (int): Amount of images used to smooth gradient
            param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
        """
        # Calculate gradients
        grads, feature_map = Backprop.generate_explanation(prep_img, target_class)

        # Generate an empty image/matrix
        smooth_grad = np.zeros(grads.shape)
        smooth_feature_map = np.zeros(feature_map.shape)

        mean = 0
        sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
        for x in range(param_n):
            print("progress: "+str(x/param_n), end='\r')
            # Generate noise
            noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
            # Add noise to the image
            noisy_img = prep_img + noise
            # Calculate gradients
            grads, feature_map = Backprop.generate_explanation(noisy_img, target_class)
            # Add gradients to smooth_grad
            smooth_grad = smooth_grad + grads
            smooth_feature_map = smooth_feature_map + feature_map
        # Average it out
        smooth_grad = smooth_grad / param_n
        smooth_feature_map = smooth_feature_map / param_n
        return smooth_grad, smooth_feature_map

# Grad-CAM
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCAM():
    """
        Produces class activation map
    """
    def __init__(self, model, which_layer_to_hook=0):
        print("GradCAM")
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, which_layer_to_hook)
    
    def toString(self):
        return "GradCAM"
    
    def generate_explanation(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        conv_output = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(conv_output.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * conv_output[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        
        '''
        cam_resize = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam_resize = np.uint8(Image.fromarray(cam_resize).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        '''
        cam = np.expand_dims(cam, axis=0)
        
        return cam, conv_output

# Guided Grad-CAM
class GuidedGradCAM():
    def __init__(self, pretrained_model, which_layer_to_hook=0):
        print("Guided GradCAM")
        self.gradcam=GradCAM(pretrained_model, which_layer_to_hook)
        self.gbp=GuidedBackprop(pretrained_model, which_forward_layer_to_hook=which_layer_to_hook, which_backward_layer_to_hook=which_layer_to_hook)
    
    def toString(self):
        return "Guided GradCAM"
    
    def generate_explanation(self, input_image, target_class=None):
        cam, feature_map = self.gradcam.generate_explanation(input_image, target_class)
        guided_grads, _ = self.gbp.generate_explanation(input_image, target_class)
        grayscale_guided_grads = convert_to_grayscale(guided_grads)[0]
        cam_gb = np.multiply(cam, grayscale_guided_grads)
        return cam_gb, feature_map

# IntegratedGradients
class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, which_layer_to_hook=0):
        print("IntegratedGradients")
        self.model = model
        self.gradients = None
        self.feature_map = None
        self.which_forward_layer_to_hook = which_layer_to_hook
        self.which_backward_layer_to_hook = which_layer_to_hook
        
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_backward_layers()
        self.hook_forward_layers()
    
    def toString(self):
        return "IntegratedGradients"
    
    def hook_backward_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0][0]

        # Register hook
        hook_layer = list(self.model.features._modules.items())[self.which_backward_layer_to_hook][1]
        hook_layer.register_backward_hook(hook_function)
        
    def hook_forward_layers(self):
        def hook_function(module, ten_in, ten_out):
            self.feature_map = ten_in[0][0]
        
        # Register hook
        hook_layer = list(self.model.features._modules.items())[self.which_forward_layer_to_hook][1]
        hook_layer.register_forward_hook(hook_function)
        
    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        feature_map_as_arr = copy.deepcopy(self.feature_map.data.numpy())
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = copy.deepcopy(self.gradients.data.numpy())
        return gradients_as_arr, feature_map_as_arr

    def generate_explanation(self, input_image, target_class, steps = 100):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        
        # init grad
        grad, feature_map = self.generate_gradients(input_image, target_class)
        
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(grad.shape)
        
        for idx, xbar_image in enumerate(xbar_list):
            print("progress: "+str(idx/len(xbar_list)), end="\r")
            # Generate gradients from xbar images
            single_integrated_grad, _ = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        
        return integrated_grads, feature_map

# Gradient * Input
class GradientxInput(VanillaBackprop):
    def __init__(self, pretrained_model, which_layer_to_hook=0):
        print("Gradient * Input")
        super(GradientxInput, self).__init__(pretrained_model, which_layer_to_hook)
    
    def toString(self):
        return "Gradient * Input"
    
    def generate_explanation(self, processed_img_tensor, target_class_index):
        vanilla_grads, feature_map = super(GradientxInput, self).generate_explanation(processed_img_tensor, target_class_index)
        return np.multiply(vanilla_grads, feature_map), feature_map

# DeepLIFT
class DeepLIFT(GradientxInput):
    def __init__(self, pretrained_model, which_layer_to_hook=0):
        print("DeepLIFT")
        super(DeepLIFT, self).__init__(pretrained_model, which_layer_to_hook)
        self._prepare_reference()
        self.baseline_inp = None
        self._override_backward()
    
    def toString(self):
        return "DeepLIFT"
    
    def _prepare_reference(self):
        def init_refs(m):
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.ref_inp_list = []
                m.ref_out_list = []

        def ref_forward(self, x):
            self.ref_inp_list.append(x.data.clone())
            out = F.relu(x)
            self.ref_out_list.append(out.data.clone())
            return out

        def ref_replace(m):
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.forward = types.MethodType(ref_forward, m)

        self.model.apply(init_refs)
        self.model.apply(ref_replace)

    def _reset_preference(self):
        def reset_refs(m):
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.ref_inp_list = []
                m.ref_out_list = []

        self.model.apply(reset_refs)

    def _baseline_forward(self, inp):
        if self.baseline_inp is None:
            self.baseline_inp = inp.data.clone()
            self.baseline_inp.fill_(0.0)
            self.baseline_inp = Variable(self.baseline_inp)
        else:
            self.baseline_inp.fill_(0.0)
        # get ref
        _ = self.model(self.baseline_inp)

    def _override_backward(self):
        def new_backward(self, grad_out):
            ref_inp, inp = self.ref_inp_list
            ref_out, out = self.ref_out_list
            delta_out = out - ref_out
            delta_in = inp - ref_inp
            g1 = (delta_in.abs() > 1e-5).float() * grad_out * \
                 delta_out / delta_in
            mask = ((ref_inp + inp) > 0).float()
            g2 = (delta_in.abs() <= 1e-5).float() * 0.5 * mask * grad_out

            return g1 + g2

        def backward_replace(m):
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.backward = types.MethodType(new_backward, m)

        self.model.apply(backward_replace)

    def generate_explanation(self, inp, ind):
        self._reset_preference()
        self._baseline_forward(inp)
        deeplift_explanation, feature_map = super(DeepLIFT, self).generate_explanation(inp, ind)

        return deeplift_explanation, feature_map