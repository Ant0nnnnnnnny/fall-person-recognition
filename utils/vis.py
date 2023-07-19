import math
import os

import numpy as np
import torch
import torchvision
import cv2
import logging
from utils.evaluate import get_max_preds

import matplotlib.pyplot as plt

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name , ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(args, x, meta, target, joints_pred, output,
                      prefix):
    if not args.debug_mode:
        return
    x = torch.Tensor.float(x)
    output = torch.Tensor.byte(output)
    if args.save_batch_image_gt:
        save_batch_image_with_joints(
            x, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if args.save_batch_image_pred:
        save_batch_image_with_joints(
            x, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if args.save_batch_heatmap_gt:
        save_batch_heatmaps(
            x, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if args.save_batch_heatmap_pred:
        save_batch_heatmaps(
            x, output, '{}_hm_pred.jpg'.format(prefix)
        )

def display_pose( img, pose):
    
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    pose  = pose.data.cpu().numpy()
    img = img.cpu().numpy().transpose(1,2,0)
    colors = ['g', 'g', 'g', 'g', 'g', 'g', 'm', 'm', 'r', 'r', 'y', 'y', 'y', 'y','y','y']
    pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
    colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
    img = np.clip(img*std+mean, 0.0, 1.0)
    img_width, img_height,_ = img.shape
    pose = ((pose + 1)* np.array([img_width, img_height])-1)/2 # pose ~ [-1,1]
    ax = plt.gca()
    plt.imshow(img)
    for idx in range(len(colors)):
        plt.plot(pose[idx,0], pose[idx,1], marker='o', color=colors[idx])
    for idx in range(len(colors_skeleton)):
        plt.plot(pose[pairs[idx],0], pose[pairs[idx],1],color=colors_skeleton[idx])

    xmin = np.min(pose[:,0])
    ymin = np.min(pose[:,1])
    xmax = np.max(pose[:,0])
    ymax = np.max(pose[:,1])

    bndbox = np.array(expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height))
    coords = (bndbox[0], bndbox[1]), bndbox[2]-bndbox[0]+1, bndbox[3]-bndbox[1]+1
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=1))

def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]