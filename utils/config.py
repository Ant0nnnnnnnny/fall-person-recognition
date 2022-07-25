import argparse


def parse_args():

    # ===============================Project ============================

    parser = argparse.ArgumentParser(
        description='Arguments for person pose estimation.')

    parser.add_argument('--model_name',type=str, default='TCFormer Pose Estimation')
    parser.add_argument('--seed', type=int, default=7310, help='Random seed.')

    parser.add_argument('--dataset_root',type=str, default='dataset\\PoseData')

    parser.add_argument('--ckpg_dir',type=str, default='checkpoints')

    # ===============================Data ================================

    parser.add_argument('--img_shape', type=list, default=[192,256],
                        help='the input images shape.')        

    parser.add_argument('--prefetch', default=16, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--num_workers', default=0, type=int,
                        help="num_workers for dataloaders")

    parser.add_argument('--batch_size', default=8, type=int,
                        help="use for training duration per worker")

    parser.add_argument('--val_batch_size', default=8,
                        type=int, help="use for validation duration per worker")

    parser.add_argument('--test_batch_size', default=8,
                        type=int, help="use for testing duration per worker")

    parser.add_argument('--num_joints_half_body',type=int,default=8)

    parser.add_argument('--flip',type=bool,default=True)

    parser.add_argument('--rotation_factor',type=float,default=30)

    parser.add_argument('--scale_factor',type=float,default=0.25)
    # ================================ optimizer ================================
    parser.add_argument('--learning_rate', type=float,
                        default=5e-4, help='learning rate.')

    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='the factor in scheduler ReduceLROnPlateau.')

    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='the patient in scheduler ReduceLROnPlateau.')

    parser.add_argument('--scheduler_min_lr', type=float,
                        default=1e-5, help='the min learing rate.')

    # ============================= Model Configs =====================
    parser.add_argument('--num_keypoints', type=int,
                        default=16)

    parser.add_argument('--heatmap_size', type=list,
                        default=[48,64])      

    parser.add_argument('--sigma', type=float,
                        default=2.0)                    

    parser.add_argument('--mta_in_channels', type=list,
                        default=[64, 128, 320, 512])

    parser.add_argument('--mta_out_channels', type=int, default=256)

    parser.add_argument('--mta_start_level', type=int, default=0)

    parser.add_argument('--mta_add_extra_convs', type=str, default='on_input')

    parser.add_argument('--mta_num_heads', type = list, default=[4,4,4,4])

    parser.add_argument('--mta_mlp_ratios', type = list, default=[4,4,4,4])

    parser.add_argument('--mta_num_outs', type = int, default=4)

    parser.add_argument('--mta_use_sr_layer', type = bool, default=False)



    parser.add_argument('--head_in_channels', type = int, default=256)

    # parser.add_argument('--head_out_channels', type = int, default=16)

    parser.add_argument('--head_num_deconv_layers', type = int, default=0)
    
    parser.add_argument('--head_extra', type = dict, default=dict(final_conv_kernel=1, ))

    parser.add_argument('--head_loss_keypoint', type = dict, default=dict(type='JointsMSELoss', use_target_weight=True))

    # ============================ train ========================================

    parser.add_argument('--max_epochs', type=int, default=100)

    parser.add_argument('--print_steps', type=int, default=100)

    parser.add_argument('--log_dir',type=str,default='log')

    parser.add_argument('--auto_resume',type=bool,default=True)

    parser.add_argument('--output_dir',type=str, default='output')
    parser.add_argument('--debug_dir',type=str,default='debug')

    #==============================Debug config =======================
    parser.add_argument('--post_processing',type=bool,default=True)
    parser.add_argument('--debug_mode',type=bool,default=True)
    parser.add_argument('--save_batch_image_gt',type=bool,default=True)
    parser.add_argument('--save_batch_image_pred',type=bool,default=True)
    parser.add_argument('--save_batch_heatmap_gt',type=bool,default=True)
    parser.add_argument('--save_batch_heatmap_pred',type=bool,default=True)



    return parser.parse_args()
