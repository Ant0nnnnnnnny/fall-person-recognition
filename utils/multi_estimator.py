import cv2
import dsntnn
import numpy as np
import torch
import onnxruntime


class MultiEstimator:

    def __init__(self, args):

        self.img_shape = args.img_shape

        self.session = onnxruntime.InferenceSession(args.estimator_onnx_path)

        self.input_name = self.session.get_inputs()[0].name

        self.output_name_list = [o.name for o in self.session.get_outputs()]


        self.skeleton_info = {
            0:
            dict(index=(15, 13),   color=[0, 255, 0]),
            1:
            dict(index=(13, 11),   color=[0, 255, 0]),
            2:
            dict(index=(16, 14),   color=[255, 128, 0]),
            3:
            dict(index=(14, 12),   color=[255, 128, 0]),
            4:
            dict(index=(11, 12),   color=[51, 153, 255]),
            5:
            dict(index=(5, 11),   color=[51, 153, 255]),
            6:
            dict(index=(6, 12),   color=[51, 153, 255]),
            7:
            dict(
                index=(5, 6),

                color=[51, 153, 255]),
            8:
            dict(index=(5, 7),   color=[0, 255, 0]),
            9:
            dict(
                index=(6, 8),   color=[255, 128, 0]),
            10:
            dict(index=(7, 9),   color=[0, 255, 0]),
            11:
            dict(index=(8, 10),   color=[255, 128, 0]),
            12:
            dict(index=(1, 2),   color=[51, 153, 255]),
            13:
            dict(index=(0, 1),   color=[51, 153, 255]),
            14:
            dict(index=(0, 2),   color=[51, 153, 255]),
            15:
            dict(index=(1, 3),   color=[51, 153, 255]),
            16:
            dict(index=(2, 4),   color=[51, 153, 255]),
            17:
            dict(index=(3, 5),   color=[51, 153, 255]),
            18:
            dict(
                index=(4, 6),   color=[51, 153, 255])
        }
        self.keypoint_info = {
            0: [51, 153, 255],
            1:
            [51, 153, 255],
            2:
            [51, 153, 255],
            3:
            [51, 153, 255],
            4:
            [51, 153, 255],
            5:
            [0, 255, 0],
            6:
            [255, 128, 0],
            7:
            [0, 255, 0],
            8:
            [255, 128, 0],
            9:
            [0, 255, 0],
            10:
            [255, 128, 0],
            11:
            [0, 255, 0],
            12:
            [255, 128, 0],
            13:
            [0, 255, 0],
            14:
            [255, 128, 0],
            15:
            [0, 255, 0],
            16:
            [255, 128, 0],
        }

    def rescale(self, img, output_size):

        img_ = img/256.0
        h, w = img_.shape[:2]

        im_scale = min(
            float(output_size[0]) / float(h), float(output_size[1]) / float(w))
        new_h = int(img_.shape[0] * im_scale)
        new_w = int(img_.shape[1] * im_scale)
        image = cv2.resize(img_, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        left_pad = int((output_size[1] - new_w) / 2.0)
        top_pad = int((output_size[0] - new_h) / 2.0)
        mean = np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, top_pad), (left_pad, left_pad))
        image = np.stack([np.pad(image[:, :, c], pad, mode='constant',
                         constant_values=mean[c])for c in range(3)], axis=2)

        def pose_fun(x): return ((((x.reshape([-1, 2])+np.array([1.0, 1.0]))/2.0*np.array(
            output_size)-[left_pad, top_pad]) * 1.0 / np.array([new_w, new_h])*np.array([w, h])))
        image = cv2.resize(
            image, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
        return {'image': image, 'pose_fun': pose_fun}

    def to_tensor(self, image):

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(
            ((image-mean)/std).transpose((2, 0, 1))).float()
        return image

    def inference(self, imgs, boxes):
        multi_keypoints = []
        for i in boxes:
            img = imgs[int(i[1]):int(i[3]), int(i[0]):int(i[2]), :]
            height = img.shape[0]
            width = img.shape[1]

            if height == 0 or width == 0:
                return np.array([])
            # 预处理
            rescale_out = self.rescale(img, self.img_shape)
            image = rescale_out['image']

            onnx_inputs = np.expand_dims(
                image.astype('float32').transpose(2, 0, 1), 0)

            # 推理

            px, py = self.session.run(self.output_name_list, 
                                        {self.input_name: onnx_inputs})

            # SimCC转换为2d坐标
            position2d = np.array(
                [[np.argmax(px[0, i]), np.argmax(py[0, i])]for i in range(17)],dtype='float32')

            # 归一化
            position2d /= np.array([384.0,512.0])
      
            multi_keypoints.append(position2d)

        return np.array(multi_keypoints)

    def vis(self, npimg, multi_keypoints, bboxs, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)

        for box, single_keypoints in zip(bboxs, multi_keypoints):

            roi_w = box[2] - box[0]
            roi_h = box[3] - box[1]

            single_keypoints[:, 0] *= roi_w
            
            single_keypoints[:, 1] *= roi_h

            offset  = np.array([box[0],box[1]])

            single_keypoints = (single_keypoints + offset).astype(int)
            
            cv2.rectangle(npimg, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (0, 0, 255), 4)
            for i in range(17):
                cv2.circle(npimg, (single_keypoints[i][0], single_keypoints[i][1]),
                           3, self.keypoint_info[i], thickness=3, lineType=8, shift=0)
            for i in range(19):
                cv2.line(npimg,
                         (single_keypoints[self.skeleton_info[i]['index'][0]][0],
                          single_keypoints[self.skeleton_info[i]['index'][0]][1]),
                         (single_keypoints[self.skeleton_info[i]['index'][1]][0],
                          single_keypoints[self.skeleton_info[i]['index'][1]][1]),
                         self.skeleton_info[i]['color'], 3)
        return npimg
