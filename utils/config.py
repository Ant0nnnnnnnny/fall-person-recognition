import argparse


def parse_args():

    # ===============================Project ============================
    parser = argparse.ArgumentParser(
        description='Arguments for person pose estimation.')
    parser.add_argument('--seed', type=int, default=7310, help='Random seed.')

    parser.add_argument('--train_annotation', type=str,
                        default='dataset\\PoseData\\Annotation\\train.json')
    parser.add_argument('--test_annotation', type=str,
                        default='dataset\\PoseData\\Annotation\\test.json')
    parser.add_argument('--train_image_feat', type=str,
                        default='dataset\\PoseData\\zip\\images\\')

    # ===============================Data ================================
    parser.add_argument('--img_shape', type=int, default=224,
                        help='the input images shape.')

    parser.add_argument('--val_ratio', type=float, default=0.3)

    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")

    parser.add_argument('--num_workers', default=0, type=int, help="num_workers for dataloaders")

    parser.add_argument('--batch_size', default=4, type=int, help="use for training duration per worker")

    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")

    parser.add_argument('--test_batch_size', default=32, type=int, help="use for testing duration per worker")

    # ================================ optimizer ================================
    parser.add_argument('--learning_rate', type=float,
                        default=0.05, help='learning rate.')

    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='the factor in scheduler ReduceLROnPlateau.')

    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='the patient in scheduler ReduceLROnPlateau.')

    parser.add_argument('--scheduler_min_lr', type=float,
                        default=0.001, help='the min learing rate.')

    # ============================ train ========================================

    parser.add_argument('--max_epochs',type=int,default=20)

    parser.add_argument('--print_steps',type=int,default=20)

    return parser.parse_args()
