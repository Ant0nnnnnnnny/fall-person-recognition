import os
from collections import namedtuple
import cv2
import torch
import torchvision
import torch.nn as nn
from .evaluate import get_max_preds
import math
from .vis import display_pose

import matplotlib.pyplot as plt
import numpy as np
import dsntnn
def save_checkpoint(states, is_best, output_dir,model_name,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir,model_name,filename))
    if not os.path.exists( os.path.join(output_dir,model_name)):
        os.mkdir( os.path.join( output_dir,model_name))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, model_name, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return summary,details

def inference(model,args,x,index, meta,mode = 'offline',use_dataset = False):
    '''
    mode: `real_time` or `offline`
    '''
    nrow = 8
    padding = 2
    if mode == 'offline' :
        if use_dataset:
            if  args.inference_dir is None:
                raise TypeError(
                "Output dir should not be none in offline mode."
            )
            else:
                model.eval()
                x = torch.Tensor.float(x).to(args.device)
                y = model(x,None,None)
                if args.model_name == 'mfnet':
                    y = dsntnn.dsnt(y)
                    display_pose(x,y)
                preds,_ = get_max_preds(y.cpu().detach().numpy())
                preds *=8
                grid_image = torchvision.utils.make_grid(x, nrow, padding, True)
                ndarr = grid_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                ndarr = ndarr.copy()

                nmaps = x.size(0)
                xmaps = min(nrow, nmaps)
                ymaps = int(math.ceil(float(nmaps) / xmaps))
                height = int(x.size(2) + padding)
                width = int(x.size(3) + padding)
                k = 0
                for y in range(ymaps):
                    for x in range(xmaps):
                        if k >= nmaps:
                            break
                        joints = preds[k]

                        for joint in joints:
                            joint[0] = x * width + padding + joint[0]
                            joint[1] = y * height + padding + joint[1]
                            cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                        k = k + 1
        
                plt.figure(figsize=(18,10))
                plt.imshow(ndarr)
                plt.savefig(os.path.join(args.inference_dir, 'inf-'+str(index)+'.png'))
                # plt.show()
        else:
            model.eval()
            x = torch.tensor(x).to(args.device)
            x = torch.Tensor.float(x)
            x = x.unsqueeze(0)
            x = x.permute(0,3,1,2)
            y = model(x,None,None)
            if args.model_name == 'mfnet':
                y = dsntnn.dsnt(y)
                print(x)
                display_pose(x[0][:3,:,:]/256,y[0])
                return 
            preds,_ = get_max_preds(y.cpu().detach().numpy())
            preds *=8
            joints = preds[0]
            height = int(x.size(2))
            width = int(x.size(3))
            grid_image = torchvision.utils.make_grid(x, nrow, padding, True)
            ndarr = grid_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            ndarr = ndarr.copy()
            ndarr = show_skeleton(ndarr, joints)
            return ndarr
    elif mode == 'online':
        model.eval()
        x = torch.tensor(x).to(args.device)
        x = torch.Tensor.float(x)
        x = x.unsqueeze(0)
        x = x.permute(0,3,1,2)
        y = model(x,None,None)
        preds,_ = get_max_preds(y.cpu().detach().numpy())
        preds *=8
        joints = preds[0]
        height = int(x.size(2))
        width = int(x.size(3))
        grid_image = torchvision.utils.make_grid(x, nrow, padding, True)
        ndarr = grid_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        ndarr = ndarr.copy()
        ndarr = show_skeleton(ndarr, joints)
        return ndarr

def show_skeleton(img,kpts,color=(255,128,128)):
    skelenton = [[10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [8, 9], [7, 8], [2, 6],
                 [3, 6], [1, 2], [1, 0], [3, 4], [4, 5],[6,7]]
    points_num = [num for num in range(1,16)]
    for sk in skelenton:
        pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
        pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points-1][0]),int(kpts[points-1][1]))
        if pos[0] > 0 and pos[1] > 0 :
            cv2.circle(img, pos,4,(0,0,255),-1) #为肢体点画红色实心圆
    return img
