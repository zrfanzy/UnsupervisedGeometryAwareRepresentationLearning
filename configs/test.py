import matplotlib.pyplot as plt

import sys
import torch
import numpy as np
import numpy.linalg as la
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import plotting as utils_plt
from utils import skeleton as utils_skel

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events

from matplotlib.widgets import Slider, Button

import matplotlib.pyplot as plt

from datasets import collected_dataset
import sys, os, shutil

import numpy as np
#import pickle
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import training as utils_train
from utils import plot_dict_batch as utils_plot_batch

from models import unet_encode3D
from losses import generic as losses_generic
from losses import images as losses_images

import math
import torch
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models_tv
import glob
import data_generation
import param
import cv2
import pickle
import skimage.io

def tensor_to_npimg(torch_array):
    return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

def npimg_to_tensor(torch_array):
    return (np.swapaxes(np.swapaxes(torch_array, 0, 1), 0, 2))

def denormalize(np_array):
    return np_array * np.array(config_dict['img_std']) + np.array(config_dict['img_mean'])

def normalize(np_array):
    return (np_array.astype('float') - np.array(config_dict['img_mean'])) / np.array(config_dict['img_std'])

# extract image
def tensor_to_img(output_tensor):
    output_img = tensor_to_npimg(output_tensor)
    output_img = denormalize(output_img)
    output_img = np.clip(output_img, 0, 1)
    return output_img

def img_to_tensor(input_tensor):
    input_img = normalize(input_tensor)
    input_img = npimg_to_tensor(input_img)
    return input_img

def rotationMatrixXZY(theta, phi, psi):
    Ax = np.matrix([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    Ay = np.matrix([[np.cos(phi), 0, -np.sin(phi)],
                    [0, 1, 0],
                    [np.sin(phi), 0, np.cos(phi)]])
    Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1], ])
    return Az * Ay * Ax

def load_network(config_dict):
        output_types= config_dict['output_types']
        
        use_billinear_upsampling = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear']
        lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'
        
        from_latent_hidden_layers = config_dict.get('from_latent_hidden_layers', 0)
        num_encoding_layers = config_dict.get('num_encoding_layers', 4)
        
        num_cameras = 4
        if config_dict['active_cameras']: # for H36M it is set to False
            num_cameras = len(config_dict['active_cameras'])
        
        if lower_billinear:
            use_billinear_upsampling = False
        network_single = unet_encode3D.unet(dimension_bg=config_dict['latent_bg'],
                                            dimension_fg=config_dict['latent_fg'],
                                            dimension_3d=config_dict['latent_3d'],
                                            feature_scale=config_dict['feature_scale'],
                                            shuffle_fg=config_dict['shuffle_fg'],
                                            shuffle_3d=config_dict['shuffle_3d'],
                                            latent_dropout=config_dict['latent_dropout'],
                                            in_resolution=config_dict['inputDimension'],
                                            encoderType=config_dict['encoderType'],
                                            is_deconv=not use_billinear_upsampling,
                                            upper_billinear=upper_billinear,
                                            lower_billinear=lower_billinear,
                                            from_latent_hidden_layers=from_latent_hidden_layers,
                                            n_hidden_to3Dpose=config_dict['n_hidden_to3Dpose'],
                                            num_encoding_layers=num_encoding_layers,
                                            output_types=output_types,
                                            subbatch_size=config_dict['useCamBatches'],
                                            implicit_rotation=config_dict['implicit_rotation'],
                                            skip_background=config_dict['skip_background'],
                                            num_cameras=num_cameras,
                                            )
        
        if 'pretrained_network_path' in config_dict.keys(): # automatic
            if config_dict['pretrained_network_path'] == 'MPII2Dpose':
                pretrained_network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/output_save/CVPR18_H36M/TransferLearning2DNetwork/h36m_23d_crop_relative_s1_s5_aug_from2D_2017-08-22_15-52_3d_resnet/models/network_000000.pth'
                print("Loading weights from MPII2Dpose")
                pretrained_states = torch.load(pretrained_network_path)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0, add_prefix='encoder.') # last argument is to remove "network.single" prefix in saved network
            else:
                print("Loading weights from config_dict['pretrained_network_path']")
                pretrained_network_path = config_dict['pretrained_network_path']            
                pretrained_states = torch.load(pretrained_network_path)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0) # last argument is to remove "network.single" prefix in saved network
                print("Done loading weights from config_dict['pretrained_network_path']")
        
        if 'pretrained_posenet_network_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_posenet_network_path']")
            pretrained_network_path = config_dict['pretrained_posenet_network_path']            
            pretrained_states = torch.load(pretrained_network_path)
            utils_train.transfer_partial_weights(pretrained_states, network_single.to_pose, submodule=0) # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_posenet_network_path']")
        return network_single


def predict():
    #nonlocal output_dict
    #model.eval()
    with torch.no_grad():
        input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
        output_dict_cuda = model(input_dict_cuda)
        output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
    
if __name__ == "__main__":
    params = param.get_general_params()
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    config_dict['batch_size_test'] = 2
    config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)

    # load data
    device='cuda'

    data_loader = pickle.load(open('examples/test_set.pickl',"rb"))
    ex = data_loader[0][0]
    

    # load model
    model = load_network(config_dict)
    model = model.to(device)

    imgfiles = glob.glob('/scratch/rzhou/posewarp-cvpr2018/poses/All/*.jpg')
    imgfiles = sorted(imgfiles)

    count = 0
    for i in range(69):
        for j in range(15):
            img = imgfiles[count]
            base = os.path.basename(img)
            filename = os.path.splitext(base)[0]
            print(str(count) + " : filename: " + filename)
            matfile = '/scratch/rzhou/posewarp-cvpr2018/poses/pose/' + filename + '.mat'

            if (not os.path.isfile(matfile)):
                count = count + 1
                print('missing people ' + str(i) + ', pose ' + str(j))
                continue
            count = count + 1
            X, Y = data_generation.create_feed_ours(params, img, matfile)
            #print(np.shape(X[0,:,:,:]))
            input_dict = {}
            for k in range(15):
                tmpy = img_to_tensor(Y[k,:,:,:])
                tmpx = img_to_tensor(X[k,:,:,:])
                tmp = np.concatenate(([tmpy], [tmpx]))
                tmp = torch.from_numpy(tmp).float().cuda()
                input_dict['img_crop'] = tmp

                tmp1 = torch.from_numpy(np.concatenate(([np.eye(3)], [np.eye(3)]))).float().cuda()
                input_dict['extrinsic_rot'] = tmp1
                input_dict['extrinsic_rot_inv'] = tmp1

                input_dict['bg_crop'] = input_dict['img_crop']
                # white background
                input_dict['bg_crop'] = torch.from_numpy(np.ones((2,3,128,128))*3).float().cuda()
                input_dict['shuffled_appearance'] =torch.from_numpy(np.array([1,0]).astype('int'))
                # test on example
                # input_dict = ex

                input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().cuda()
                #input_dict['external_rotation_global'] = torch.from_numpy(rotationMatrixXZY(theta=0, phi=0, psi=2.0)).float().cuda()
                input_dict['external_rotation_cam'] = torch.from_numpy(np.eye(3)).float().cuda()

                output_dict = None
                label_dict = None
                model.eval()
                with torch.no_grad():
                    input_dict_cuda = utils_data.nestedDictToDevice((input_dict), device=device)
                    output_dict_cuda = model(input_dict_cuda)
                    output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
                    skimage.io.imsave('whiteresults/' + filename + '_' + str(k) + '.png', tensor_to_img(output_dict['img_crop'][0]))
                    #skimage.io.imsave('redin.png', X[i,:,:,:])
                    #skimage.io.imsave('results/' + str(i) + '.png', Y[i,:,:,:])
                    #pickle
            #pred = model.predict(X[:-2])

            #print(Y[0])
            #skimage.io.imsave('test1.png', Y[0])
            #print(np.shape(Y[0]))
            #testimg = tensor_to_npimg(npimg_to_tensor(Y[0]))
            #skimage.io.imsave('test.png', testimg)
            #break
        #break 
