import json
import os
from scipy.io import loadmat
def save_joints(mat_path, save_path):
    mat = loadmat(mat_path)
    image_write = []
    train_wirte = []
    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):
        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            # only one person
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]


            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]
                    # build feed_dict
                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)
                    data = {
                        'filename': img_fn,
                        'train': train_flag,
                        'head_rect': head_rect,
                        'joint_pos': joint_pos
                    }
                    train_wirte.append(data)
        else:
            data = {
                'filename': img_fn,
                'train': train_flag,
            }
            image_write.append(data)
    print(image_write)
    print(train_wirte)
    test_dic = {"test":image_write}
    train_dic = {"train":train_wirte}
    with open(os.path.join(save_path,'train'+".json"), 'w') as f:
                json.dump(train_dic,f)

    with open(os.path.join(save_path,
                                   'test'+".json"), 'w') as f:
                json.dump(test_dic,f)
save_joints('dataset\\PoseData\\Annotation\\ann.mat',
            'dataset\\PoseData\\Annotation')
