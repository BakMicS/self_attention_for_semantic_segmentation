import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

title_name_per_model = {'san12': 'SAN12',
                        'san19': 'SAN19',
                        'deeplab50': 'deeplab-resnet50',
                        'deeplab101': 'deeplab-resnet101',
                        'deeplab50_aspp_san': 'deeplab-resnet50 SAN ASPP',
                        'deeplab50_asppsan': 'deeplab-resnet50 SAN ASPP',
                        'deeplab101_aspp_san': 'deeplab-resnet101 SAN ASPP',
                        'deeplab101_asppsan': 'deeplab-resnet101 SAN ASPP',
                        'deeplabv3_lrnet50': 'deeplab-lrnet50'}


def get_image_size_from_dir_name(out_dir):
    parts_split_x = [part.split('x') for part in out_dir.split('_')]
    img_size_list = [part[1] for part in parts_split_x if len(part) == 2]
    if len(img_size_list) == 1:
        return img_size_list[0]
    else:
        return 'unkwn'


def get_model_name_from_dir_name(out_dir):
    model_backbone = [part for part in out_dir.split('_') if part.__contains__('deeplab')][0]
    full_name = model_backbone + '_asppsan' if out_dir.__contains__('asppsan') and not out_dir.__contains__('test') else model_backbone
    model_name = title_name_per_model[full_name]
    return model_name


def plot_and_save_learning_curves_from_log_file(out_dir, train_args=None, add=''):

    log_df = pd.read_csv(os.path.join(out_dir, 'log.csv'))

    best_miou = np.round(log_df['Test_mIOU'].max(), 3)
    if not train_args:
        dir_name = os.path.split(out_dir)[1]
        model_name = get_model_name_from_dir_name(dir_name)
        img_size = get_image_size_from_dir_name(dir_name)
        pretrained = 'pretrained' if dir_name.__contains__('freeze') else 'from_scratch'
        batch_size = [part.split('batch')[1] for part in dir_name.split('_') if part.__contains__('batch')][0]
    else:
        model_name = title_name_per_model[train_args.model]
        img_size = get_image_size_from_dir_name(train_args.out_dir)
        pretrained = 'pretrained' if train_args.out_dir.__contains__('freeze') else 'from_scratch'
        batch_size = train_args.batch_size

    suptitle = '{model_name} ({img_size}x{img_size},batch={batch_size},{pretrained}{add})\n best mIOU: {best_miou}'.format(
        model_name=model_name, img_size=img_size, batch_size=batch_size, pretrained=pretrained, add=add, best_miou=best_miou)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[7, 7])
    plt.suptitle(suptitle)
    log_df[['Test_loss', 'Train_loss']].plot(ax=axes[0])
    log_df[['Test_mIOU', 'Train_mIOU']].plot(ax=axes[1])
    plt.xlabel('epoch')

    plt.savefig(os.path.join(out_dir, 'learning_curvs.png'))
    plt.show()


if __name__ == "__main__":

    log_file_dir = './train_output/12-08-2021_08-08-56_10-08-2021_23-16-41_deeplab101_asppsan_conv1_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/04-09-2021_21-45-02_deeplab101_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/10-09-2021_08-53-22_deeplab101_asppsan_496x496_batch4_no_freeze'
    # log_file_dir = './train_output/09-09-2021_13-30-06_deeplab101_496x496_batch4_no_freeze'

    # Kernels 5 & 7
    log_file_dir = './train_output/05-10-2021_04-27-39_deeplab101_asppsan_conv1_kernel7_512x512_batch4_backbone_freeze'
    # log_file_dir = './train_output/04-10-2021_19-57-59_deeplab101_asppsan_conv1_kernel5_512x512_batch6_backbone_freeze'

    # both ASPPs
    log_file_dir = './train_output/04-10-2021_07-03-35_deeplab101_aspp2san_conv1_512x512_batch8_backbone_freeze'

    # Best regular rates
    log_file_dir = './train_output/05-10-2021_22-07-32_deeplab101_asppsan_noinBN_conv1_512x512_batch8_backbone_freeze'
    # log_file_dir = './train_output/06-10-2021_09-22-16_deeplab101_test_512x512_batch8_backbone_freeze'

    # Special rates
    # log_file_dir = './train_output/06-10-2021_20-21-36_deeplab101_asppsan_4_8_12_conv1_noinBN_512x512_batch8_backbone_freeze'
    # log_file_dir = './train_output/07-10-2021_07-36-12_deeplab101_asppsan_test_4_8_12_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/07-10-2021_17-16-05_deeplab101_asppsan_test_4_8_12_512x512_batch8_backbone_freeze'

    # 4 ASPP blocks
    # log_file_dir = './train_output/08-10-2021_08-05-52_deeplab101_asppsan_full_512x512_batch8_backbone_freeze'
    # log_file_dir = './train_output/10-10-2021_08-08-05_deeplab101_asppsan_full_12_24_36_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/10-10-2021_18-54-09_deeplab101_asppsan_full_4_8_12_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/11-10-2021_07-05-01_deeplab101_asppsan_full_12_24_36_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/11-10-2021_19-01-36_deeplab101_asppsan_test_12_24_36_512x512_batch8_backbone_freeze'
    log_file_dir = './train_output/12-10-2021_08-48-06_deeplab101_asppsan_full_k5_4_8_12_512x512_batch6_backbone_freeze'
    log_file_dir = './train_output/13-10-2021_00-02-48_deeplab101_asppsan_k5_6_12_18_512x512_batch6_backbone_freeze'

    add = ',k=5,rates=[6,12,18]'  # ',rates=[4,8,12]'
    plot_and_save_learning_curves_from_log_file(log_file_dir, add=add)
