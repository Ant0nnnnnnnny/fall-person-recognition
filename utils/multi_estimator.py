import cv2
import dsntnn
import numpy as np
import torch

class MultiEstimator:
    
    def __init__(self,model, args):
        
        self.img_shape = args.img_shape[0]
        
        self.model = model.to('mps')
        
        self.model.eval()
   
    def rescale(self, img, output_size):

        img_ = img/256.0
        h, w = img_.shape[:2]
        
        im_scale = min(float(output_size[0]) / float(h), float(output_size[1]) / float(w))
        new_h = int(img_.shape[0] * im_scale)
        new_w = int(img_.shape[1] * im_scale)
        image = cv2.resize(img_, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left_pad =int( (output_size[1] - new_w) / 2.0)
        top_pad = int((output_size[0] - new_h) / 2.0)
        mean=np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, top_pad), (left_pad, left_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c])for c in range(3)], axis=2)
        pose_fun = lambda x: ((((x.reshape([-1,2])+np.array([1.0,1.0]))/2.0*np.array(output_size)-[left_pad, top_pad]) * 1.0 /np.array([new_w, new_h])*np.array([w,h])))
        
        return {'image': image, 'pose_fun': pose_fun}

    def to_tensor(self, image):
     
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])    
        image = torch.from_numpy(((image-mean)/std).transpose((2, 0, 1))).float()
        return image

    def inference(self, imgs, boxes):
        multi_keypoints = []
        for i in boxes:
            img = imgs[int(i[1]):int(i[3]), int(i[0]):int(i[2]),:]
            
            height = img.shape[0]
            width = img.shape[1]

            rescale_out = self.rescale(img, (self.img_shape, self.img_shape))

            image = rescale_out['image']
            image = self.to_tensor(image).to('mps')
            image = image.unsqueeze(0)
            pose_fun = rescale_out['pose_fun']

            keypoints = self.model(image,0,0)

            keypoints = dsntnn.dsnt(keypoints)

            keypoints = keypoints[0].detach().cpu().numpy()

            keypoints = pose_fun(keypoints).astype(int) + i[:2]
            
            multi_keypoints.append(keypoints.astype(int))
            
        return np.array(multi_keypoints)

    def vis(self,npimg, multi_keypoints, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for single_keypoints in multi_keypoints:
            colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255]]

            pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
            colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
            colors_skeleton = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255]]

            for idx in range(len(colors)):
                cv2.circle(npimg, (single_keypoints[idx,0], single_keypoints[idx,1]), 3, colors[idx], thickness=3, lineType=8, shift=0)
            for idx in range(len(colors_skeleton)):
                npimg = cv2.line(npimg, (single_keypoints[pairs[idx][0],0], single_keypoints[pairs[idx][0],1]), (single_keypoints[pairs[idx][1],0], single_keypoints[pairs[idx][1],1]), colors_skeleton[idx], 3)

        return npimg