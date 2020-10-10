import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
import torch
from torch.autograd import Variable
from torchvision import models
from torch.nn import ReLU

# some functions for image processing, pre-trained model loading, and a unified way to conduct saliency map methods

# a unified way to conduct saliency map methods
def conduct_saliency_map_method(METHOD, processed_img_tensor, target_class_index, pretrained_model, which_layer_to_hook=0):
    saliency_map_method = METHOD(pretrained_model, which_layer_to_hook=0)
    explanation, feature_map = saliency_map_method.generate_explanation(processed_img_tensor, target_class_index)
    return explanation, feature_map

# load pre-trained model
def get_pretrained_model(model_name):
    if model_name == "alexnet":
        return models.alexnet(pretrained=True)
    
    elif model_name == "vgg11":
        return models.vgg11(pretrained=True)
    elif model_name == "vgg11_bn":
        return models.vgg11_bn(pretrained=True)
    elif model_name == "vgg13":
        return models.vgg13(pretrained=True)
    elif model_name == "vgg13_bn":
        return models.vgg13_bn(pretrained=True)
    elif model_name == "vgg16":
        return models.vgg16(pretrained=True)
    elif model_name == "vgg16_bn":
        return models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        return models.vgg19(pretrained=True)
    elif model_name == "vgg19_bn":
        return models.vgg19_bn(pretrained=True)
    
    elif model_name == "resnet18":
        return models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        return models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        return models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=True)

# process an ImageNet raw image to input it into a CNN
def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")
            
    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)
        
    im_as_arr = np.float32(pil_im)
    
    im_resize = np.array(im_as_arr, dtype = np.int)
    
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    
    return im_as_var, im_resize

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max>im_min:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
    '''
    grayscale_im = np.sum(im_as_arr, axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
    '''

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../grabcheck_results'):
        os.makedirs('../grabcheck_results')
    
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    
    # Save image
    path_to_file = os.path.join('../grabcheck_results', file_name + '.jpg')
    save_image(gradient, path_to_file)
'''
def get_positive_negative(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency
'''
def get_positive_negative(input_img, is_one_channel_output = True):
        
    pos = np.maximum(0, input_img)
    neg = np.maximum(0, -input_img)
        
    if is_one_channel_output==True:
        pos = convert_to_grayscale(pos)
        neg = convert_to_grayscale(neg)
    else:
        pos_max = np.percentile(pos, 99)
        pos_min = np.min(pos)
        if pos_max>pos_min:
            pos = (np.clip((pos - pos_min) / (pos_max - pos_min), 0, 1))

        neg_max = np.percentile(neg, 99)
        neg_min = np.min(neg)
        if neg_max>neg_min:
            neg = (np.clip((neg - neg_min) / (neg_max - neg_min), 0, 1))
    
    return pos, neg