import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from custom.label import color2label_komatsu600, rgb2label
from custom.analyze import print_avg_iou
from custom.visualize import visualize_class_distribution, visualize_comparison

dataset_dir = 'dataset/komatsu600'
segmentation_dir = os.path.join(dataset_dir, 'segmentation')
label_dir = os.path.join(dataset_dir, 'label')
list_dir = os.path.join(dataset_dir, 'list')
list_depth_dir = os.path.join(dataset_dir, 'list-depth')


def transform():
    """
    transform rgb-segmentation label to train-id label
    """
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    segmentation_paths = glob.glob(os.path.join(segmentation_dir, '*'))
    for path in tqdm(segmentation_paths):
        label = cv2.imread(path)[:, :, ::-1]
        new_label = rgb2label(label, color2label_komatsu600)
        cv2.imwrite(os.path.join(label_dir, os.path.basename(path)), new_label)


def makeList_depth_random(train_rate=0.8):
    if not os.path.exists(list_depth_dir):
        os.mkdir(list_depth_dir)
    train_list_path = os.path.join(list_depth_dir, 'train.txt')
    val_list_path = os.path.join(list_depth_dir, 'val.txt')
    depth_paths = sorted(glob.glob(os.path.join(dataset_dir, 'normalized_3C_depth', '*')))
    rgb_paths = sorted(glob.glob(os.path.join(dataset_dir, 'rgb', '*')))
    label_paths = sorted(glob.glob(os.path.join(dataset_dir, 'label', '*')))
    # remove dataset_dir
    depth_paths = [p.replace(dataset_dir + '/', '') for p in depth_paths]
    rgb_paths = [p.replace(dataset_dir + '/', '') for p in rgb_paths]
    label_paths = [p.replace(dataset_dir + '/', '') for p in label_paths]
    assert len(depth_paths) == len(rgb_paths) and len(depth_paths) == len(label_paths)
    total_num = len(label_paths)
    train_num = int(total_num * train_rate)
    idxes = np.arange(total_num)
    np.random.seed(0)
    np.random.shuffle(idxes)
    # shuffle data
    depth_paths = [depth_paths[i] for i in idxes]
    rgb_paths = [rgb_paths[i] for i in idxes]
    label_paths = [label_paths[i] for i in idxes]
    with open(train_list_path, 'w') as f:
        f.writelines(
            [' '.join([rgb, depth, label]) + '\n' for rgb, depth, label in
             zip(rgb_paths[:train_num], depth_paths[:train_num], label_paths[:train_num])])
    with open(val_list_path, 'w') as f:
        f.writelines(
            [' '.join([rgb, depth, label]) + '\n' for rgb, depth, label in
             zip(rgb_paths[train_num:], depth_paths[train_num:], label_paths[train_num:])])


def makeList_random(train_rate=0.8):
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)
    train_list_path = os.path.join(list_dir, 'train.txt')
    val_list_path = os.path.join(list_dir, 'val.txt')
    rgb_paths = sorted(glob.glob(os.path.join(dataset_dir, 'rgb', '*')))
    label_paths = sorted(glob.glob(os.path.join(dataset_dir, 'label', '*')))
    # remove dataset_dir
    rgb_paths = [p.replace(dataset_dir + '/', '') for p in rgb_paths]
    label_paths = [p.replace(dataset_dir + '/', '') for p in label_paths]
    assert len(rgb_paths) == len(label_paths)
    total_num = len(label_paths)
    train_num = int(total_num * train_rate)
    idxes = np.arange(total_num)
    np.random.seed(0)
    np.random.shuffle(idxes)
    # shuffle data
    rgb_paths = [rgb_paths[i] for i in idxes]
    label_paths = [label_paths[i] for i in idxes]
    with open(train_list_path, 'w') as f:
        f.writelines(
            [' '.join([rgb, label]) + '\n' for rgb, label in
             zip(rgb_paths[:train_num], label_paths[:train_num])])
    with open(val_list_path, 'w') as f:
        f.writelines(
            [' '.join([rgb, label]) + '\n' for rgb, label in
             zip(rgb_paths[train_num:], label_paths[train_num:])])


def normalize_depth():
    src_dir = 'dataset/komatsu600/depth_cm16bit'
    dst_dir = 'dataset/komatsu600/normalized_3C_depth'
    depth_paths = glob.glob(os.path.join(src_dir, '*'))
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for path in tqdm(depth_paths):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        normalized_depth = (depth / 65535 * 255).astype('uint8')
        normalized_3C_depth = np.stack((normalized_depth,) * 3, axis=-1)
        new_path = os.path.join(dst_dir, os.path.basename(path))
        cv2.imwrite(new_path, normalized_3C_depth)


def get_label_paths(split):
    assert split in ['all', 'val', 'train']
    if split == 'all':
        return sorted(glob.glob(os.path.join(label_dir, '*')))
    else:
        lines = open(os.path.join(list_dir, f'{split}.txt'), 'r').readlines()
        paths = list(map(lambda line: os.path.join(dataset_dir, line.split(' ')[-1].strip()), lines))
        return sorted(paths)


if __name__ == '__main__':
    class_names = list(map(lambda n: n.strip(), open('data/komatsu600/komatsu600_names.txt', 'r').readlines()))

    # # Generate list file
    # makeList_random()

    # # visualize class distribution
    # visualize_class_distribution(get_label_paths('all'), class_names, 'All')
    # visualize_class_distribution(get_label_paths('val'), class_names, 'Test')
    # visualize_class_distribution(get_label_paths('train'), class_names, 'Train')

    # # Show comparison between true segmentation and predicted segmentation
    # label_paths = get_label_paths('val')
    # rgb_paths = list(map(lambda path: path.replace('label', 'rgb'), label_paths))
    # depth_paths = list(map(lambda path: path.replace('label', 'normalized_3C_depth'), label_paths))
    # true_seg_paths = list(map(lambda path: path.replace('label', 'segmentation'), label_paths))
    # pred_seg_dir = 'exp/komatsu600/attention_v1_fusepspnet50/result/epoch_100/val/ss/color'
    # pred_seg_paths = list(map(lambda path: os.path.join(pred_seg_dir, os.path.basename(path)), label_paths))
    # visualize_comparison(rgb_paths, depth_paths, true_seg_paths, pred_seg_paths)

    # print iou
    log_paths = sorted(list(glob.glob('exp/komatsu600/pspnet50/result/*.log')))[:-1]
    print_avg_iou(log_paths)
