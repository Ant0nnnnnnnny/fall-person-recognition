import argparse
import os


def parse_args():

    # ===============================Project ============================

    parser = argparse.ArgumentParser(
        description='Arguments for person pose estimation.')
    
    parser.add_argument('--model_name',type=str, default='st-gcn')
    
    parser.add_argument('--seed', type=int, default=7310, help='Random seed.')

    parser.add_argument('--dataset_root',type=str, default=os.path.join('dataset','PoseData',))
    parser.add_argument('--skeleton_dataset_dir',type=str,default=os.path.join('dataset','ActivityData','ntu120.pkl'))
    parser.add_argument('--ckpg_dir',type=str, default='checkpoints')
    parser.add_argument('--inference_dir',type=str,default=os.path.join('output','inference'))
    parser.add_argument('--log_dir',type=str,default='log')
    parser.add_argument('--output_dir',type=str, default='output')
    parser.add_argument('--debug_dir',type=str,default='debug')
    parser.add_argument('--config_dir', type = str, default= 'configs')

    parser.add_argument('--estimator_onnx_path',type = str, default=os.path.join('checkpoints','mfnet','checkpoint.onnx'))

    parser.add_argument('--auto_resume',type=bool,default=True)

    parser.add_argument('--print_steps', type=int, default=30)


    # =============================== Detector configs ================================

    parser.add_argument('--detector_prob_threshold', type=float, default=0.4)

    parser.add_argument('--detector_iou_threshold', type=float, default=0.3)
    
    parser.add_argument('--detector_weight_path', type=str, default= os.path.join('checkpoints','pico.onnx'))

    parser.add_argument('--threshold',type=float, default=0.25)
    
    # ===============================Data ================================

    parser.add_argument('--img_shape', type=list, default=[256,192],
                        help='the input images shape.')        

    parser.add_argument('--prefetch', default=2, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--num_workers', default=8, type=int,
                        help="num_workers for dataloader")

    parser.add_argument('--num_joints_half_body',type=int,default=8)

    parser.add_argument('--flip',type=bool,default=True)

    parser.add_argument('--rotation_factor',type=float,default=30)

    parser.add_argument('--scale_factor',type=float,default=0.25)

    parser.add_argument('--skeleton_label_dir',type=str,default=os.path.join('dataset','ActivityData','labels.txt'))

    parser.add_argument('--activity_classes',type=list,default=[42, 49, 50])

    # ============================= Model Configs =====================
    '''
    Moved to configs/
    '''
    # ============================ Hyperparameters ========================================

    parser.add_argument('--label_smoothing', type=float,
                        default=0)
    
    parser.add_argument('--adamw_betas', type=tuple,
                        default=(0.9, 0.999))
    parser.add_argument('--adamw_weight_decay', type=float,
                        default=1e-2)
    parser.add_argument('--adamw_amsgrad', type=bool, default=False)

    parser.add_argument('--learning_rate', type=float,
                        default=5e-4, help='learning rate.')

    parser.add_argument('--scheduler_min_lr', type=float,
                        default=1e-5, help='the min learning rate.')
    
    parser.add_argument('--max_epochs', type=int, default=300)

    parser.add_argument('--batch_size', default=4, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--val_batch_size', default=4,
                        type=int, help="use for validation duration per worker")

    #==============================Debug config =======================
    parser.add_argument('--post_processing',type=bool,default=True)
    parser.add_argument('--debug_mode',type=bool,default=True)
    parser.add_argument('--save_batch_image_gt',type=bool,default=True)
    parser.add_argument('--save_batch_image_pred',type=bool,default=True)
    parser.add_argument('--save_batch_heatmap_gt',type=bool,default=True)
    parser.add_argument('--save_batch_heatmap_pred',type=bool,default=True)

    return parser, parser.parse_args(args=[])
