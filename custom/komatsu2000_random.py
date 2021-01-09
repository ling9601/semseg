import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from custom import tools
from custom.label import Label
from custom import visualize
import matplotlib.pyplot as plt

DATASET_DIR = 'dataset/komatsu2000_random'
RGB_DIR = os.path.join(DATASET_DIR, 'rgb')
SEG_DIR = os.path.join(DATASET_DIR, 'segmentation')
DEPTH_DIR = os.path.join(DATASET_DIR, 'depth_cm16bit')
SCENES = ['forward', 'backward']
DATA_NUM_PER_SCENE = 1000

LABEL_DIR = os.path.join(DATASET_DIR, 'label')
LIST_DIR = os.path.join(DATASET_DIR, 'list')

CLASS_NAMES = [line.strip() for line in open('data/komatsu2000/komatsu2000_names.txt', 'r').readlines()]


def resize_rgb_depth_seg():
    all_img_paths = glob.glob(os.path.join(DATASET_DIR, '*', '*', '*.png'))
    assert len(all_img_paths) == DATA_NUM_PER_SCENE * 6, len(all_img_paths)
    count_depth = 0
    count_rgb_seg = 0
    for path in tqdm(all_img_paths):
        if 'depth_cm16bit' in path:
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            count_depth += 1
        elif 'rgb' in path or 'segmentation' in path:
            img = cv2.imread(path)
            count_rgb_seg += 1
        else:
            raise KeyError
        assert img.shape[: 2] == (960, 1280), img.shape
        resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, resized_img)
    assert count_depth == DATA_NUM_PER_SCENE * 2, count_depth
    assert count_rgb_seg == DATA_NUM_PER_SCENE * 4, count_rgb_seg


def check_resize():
    for path in tqdm(glob.glob(os.path.join(DATASET_DIR, '*', '*', '*.png'))):
        if 'depth_cm16bit' in path:
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            assert img.shape == (480, 640)
            assert img.dtype == np.uint16
        elif 'rgb' in path or 'segmentation' in path:
            img = cv2.imread(path)
            assert img.shape == (480, 640, 3)
            assert img.dtype == np.uint8


def normalize_depth():
    normalized_depth_folder_name = 'normalized_3C_depth'
    normalized_depth_dir = os.path.join(DATASET_DIR, normalized_depth_folder_name)
    for scene in SCENES:
        directory = os.path.join(normalized_depth_dir, scene)
        if not os.path.exists(directory): os.makedirs(directory)
    for path in tqdm(glob.glob(os.path.join(DEPTH_DIR, '*', '*'))):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        new_depth = tools.normalize_depth(depth)
        cv2.imwrite(path.replace('depth_cm16bit', normalized_depth_folder_name), new_depth)


def transform(seg_dir, label_dir):
    seg_folder_name = os.path.basename(seg_dir)
    label_folder_name = os.path.basename(label_dir)
    label_dict = {
        (0, 0, 255): Label('Road', 0),
        (255, 255, 0): Label('RoughRoad', 1),
        (255, 0, 0): Label('Berm', 2),
        (173, 216, 230): Label('Puddle', 3),
        (0, 0, 0): Label('others', 4),
        (211, 211, 211): Label('Sky', 5),
        (255, 165, 0): Label('ConstructionVehicle', 6),
        (255, 192, 203): Label('Prop', 7),
        (221, 160, 221): Label('Building', 8),
        (34, 139, 34): Label('Foliage', 9),
        (240, 230, 140): Label('Rock', 10)
    }
    for scene in SCENES:
        directory = os.path.join(label_dir, scene)
        if not os.path.exists(directory): os.makedirs(directory)
    for path in tqdm(glob.glob(os.path.join(seg_dir, '*', '*'))):
        seg = cv2.imread(path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        label = tools.rgb2label(seg, label_dict)
        assert seg_folder_name in path
        cv2.imwrite(path.replace(seg_folder_name, label_folder_name), label)


def make_list():
    """
    train: [0, 999] from forward scene and [0, 249], [750, 999] from backward scene
    test: [250, 749] from backward scene
    """

    def write(list_file_name, ori_suffix, idx_list):
        with open(os.path.join(LIST_DIR, list_file_name), 'a') as f:
            for i in idx_list:
                suffix = ori_suffix.format(i)
                paths = [os.path.join(dtype, suffix) for dtype in ['rgb', 'normalized_3C_depth', 'label']]
                assert all([os.path.exists(os.path.join(DATASET_DIR, path)) for path in paths])
                f.write(' '.join(paths) + '\n')

    if not os.path.exists(LIST_DIR): os.mkdir(LIST_DIR)
    write('train_depth.txt', 'forward/{:05d}.png', list(range(1000)))
    write('train_depth.txt', 'backward/{:05d}.png', list(range(250)) + list(range(750, 1000)))
    write('val_depth.txt', 'backward/{:05d}.png', list(range(250, 750)))


def make_list_no_depth():
    def write(src_path, file_name):
        with open(os.path.join(LIST_DIR, file_name), 'w') as f:
            lines = open(src_path, 'r').readlines()
            for line in lines:
                splited_line = line.split(' ')
                new_line = ' '.join([splited_line[0], splited_line[2]])
                f.write(new_line)

    train_depth_list_path = os.path.join(LIST_DIR, 'train_depth.txt')
    val_depth_list_path = os.path.join(LIST_DIR, 'val_depth.txt')
    write(train_depth_list_path, 'train.txt')
    write(val_depth_list_path, 'val.txt')


def create_transferred_segmentation(color_list, folder_name, method, kernel_size):
    # class foliage
    dilated_seg_dir = os.path.join(DATASET_DIR, folder_name)
    for scene in SCENES:
        directory = os.path.join(dilated_seg_dir, scene)
        if not os.path.exists(directory): os.makedirs(directory)
    seg_paths = sorted(glob.glob(os.path.join(SEG_DIR, '*', '*')))
    # debug
    seg_paths = [path for path in seg_paths if any([num in path for num in ['00001.png', '00041.png', '00129.png', '00131.png']])]
    for path in tqdm(seg_paths):
        seg = cv2.imread(path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        transferred_seg = seg
        for color in color_list:
            transferred_seg = tools.morphological_transformation(transferred_seg, color, method, kernel_size,
                                                                 show=False)
        # debug
        fig = plt.figure()
        fig.add_subplot(121).title.set_text('ori')
        plt.imshow(seg)
        fig.add_subplot(122).title.set_text('transferred')
        plt.imshow(transferred_seg)
        fig.suptitle(os.path.basename(path))
        plt.show()
        transferred_seg = cv2.cvtColor(transferred_seg, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path.replace('segmentation', folder_name), transferred_seg)


def normalize_depth_24bit():
    normalized_depth_24bit_folder_name = 'normalized_3C_depth_24bit'
    normalized_depth_24bit_dir = os.path.join(DATASET_DIR, normalized_depth_24bit_folder_name)
    for scene in SCENES:
        directory = os.path.join(normalized_depth_24bit_dir, scene)
        if not os.path.exists(directory): os.makedirs(directory)
    for path in tqdm(glob.glob(os.path.join(DEPTH_DIR, '*', '*'))):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        new_depth = tools.normalize_depth_24bit(depth)
        cv2.imwrite(path.replace('depth_cm16bit', normalized_depth_24bit_folder_name), new_depth)


if __name__ == '__main__':
    # resize_rgb_depth_seg()
    # check_resize()
    # normalize_depth()
    # transform(SEG_DIR, LABEL_DIR)
    # make_list()
    # make_list_no_depth()

    # visualize class distribution
    # label_paths = [os.path.join(DATASET_DIR, line.split(' ')[-1].strip()) for line in
    #                open('dataset/komatsu2000_random/list/val.txt', 'r').readlines()]
    # visualize.visualize_class_distribution(label_paths, CLASS_NAMES, 'val')

    # 24bit depth
    # depth_path = "dataset/komatsu2000/depth_cm16bit/forward/00000.png"
    # depth_16bit = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # tools.normalize_depth_24bit(depth_16bit)
    # normalize_depth_24bit()

    # show comparison
    # idx_list = list(range(453, 458)) + list(range(626, 632)) + list(range(657, 665)) + list(range(721, 726))
    # idx_list = list(range(649, 674)) + list(range(708, 729))
    idx_list = [253, 276, 280, 454, 631, 661]
    file_names = ['{:0>5}.png'.format(idx) for idx in idx_list]
    rgb_paths = [os.path.join(RGB_DIR, 'backward', n) for n in file_names]
    depth_paths = [os.path.join(DATASET_DIR, 'normalized_3C_depth', 'backward', n) for n in file_names]
    seg_paths = [os.path.join(SEG_DIR, 'backward', n) for n in file_names]
    pred_dir = 'exp/komatsu2000_random/pspnet50/result/epoch_200/val/ss/color'
    pred_paths = [os.path.join(pred_dir, n) for n in file_names]
    pred_dir_closing = 'exp/komatsu2000_random_closing_foliage+rock/attention_v1_fusepspnet50/result/epoch_200/val/ss/color'
    pred_paths_closing = [os.path.join(pred_dir_closing, n) for n in file_names]
    visualize.visualize_comparison(rgb_paths, depth_paths, [('true', seg_paths), ('normal', pred_paths),
                                                            ('closing', pred_paths_closing)], overlay=True, out_dir='out')

    # # dilate foliage and rock
    # color_list = [(34, 139, 34), (240, 230, 140)]
    # create_dilated_segmentation(color_list, 'segmentation_dilated_foliage+rock')
    # transform(os.path.join(DATASET_DIR, 'segmentation_dilated_foliage+rock'), os.path.join(DATASET_DIR, 'label_dilated_foliage+rock'))
    # dilate rock only
    # color_list = [(240, 230, 140)]
    # create_transferred_segmentation(color_list, 'segmentation_dilated_rock', 'dilate')
    # transform(os.path.join(DATASET_DIR, 'segmentation_dilated_rock'), os.path.join(DATASET_DIR, 'label_dilated_rock'))
    # close flioage and rock with kernel size 3x3
    # color_list = [(34, 139, 34), (240, 230, 140)]
    # create_transferred_segmentation(color_list, 'segmentation_closing_foliage+rock', 'closing', (3, 3))
    # transform(os.path.join(DATASET_DIR, 'segmentation_closing_foliage+rock'), os.path.join(DATASET_DIR, 'label_closing_foliage+rock'))

    # close flioage and rock with kernel size 5x5
    # color_list = [(34, 139, 34), (240, 230, 140)]
    # create_transferred_segmentation(color_list, 'segmentation_closing_foliage+rock_5x5', 'closing', (5, 5))
    # transform(os.path.join(DATASET_DIR, 'segmentation_closing_foliage+rock_5x5'), os.path.join(DATASET_DIR, 'label_closing_foliage+rock_5x5'))
